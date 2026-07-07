# safeDDPG.py
# Multi-agent DDPG with per-bus actor/critic, attention mixer, and soft targets.
# Supports policy/value as lists or ModuleLists. Uses the actual env to get
# bus indices and one-hop neighbors for mixer context.

from typing import List, Tuple, Union
import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

Tensor = torch.Tensor
ModuleOrList = Union[nn.Module, nn.ModuleList, List[nn.Module]]

class ReplayBufferPI:
    """Simple replay buffer used in your project:
       push(state_vec, action_vec, last_action_vec, reward_scalar, next_state_vec, done_float)
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def push(self, s, a, la, r, s2, d):
        item = (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.float32),
                np.array(la, dtype=np.float32),
                float(r),
                np.array(s2, dtype=np.float32),
                float(d))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        s, a, la, r, s2, d = zip(*batch)
        return (np.stack(s, axis=0),
                np.stack(a, axis=0),
                np.stack(la, axis=0),
                np.array(r, dtype=np.float32).reshape(-1, 1),
                np.stack(s2, axis=0),
                np.array(d, dtype=np.float32).reshape(-1, 1))

def _as_list(mods: ModuleOrList) -> List[nn.Module]:
    if isinstance(mods, nn.ModuleList):
        return list(mods)
    if isinstance(mods, list):
        return mods
    # single module -> wrap
    return [mods]

def _soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.copy_(tp * (1.0 - tau) + sp * tau)

class DDPG:
    def __init__(
        self,
        policy_net: ModuleOrList,
        value_net: ModuleOrList,
        target_policy_net: ModuleOrList,
        target_value_net: ModuleOrList,
        mixers: ModuleOrList,
        env,                        # REQUIRED: used for bus maps / neighbors
        value_lr: float = 2e-4,
        policy_lr: float = 1e-4,
        gamma: float = 0.99,
        soft_tau: float = 1e-2,
        device: str = None,
    ):
        assert env is not None, "DDPG requires a real env (used for bus maps / neighbor context)."
        self.env = env
        self.gamma = gamma
        self.soft_tau = soft_tau

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Normalize inputs into lists (one per controlled bus / agent)
        self.policies       = _as_list(policy_net)
        self.values         = _as_list(value_net)
        self.target_pols    = _as_list(target_policy_net)
        self.target_vals    = _as_list(target_value_net)
        self.mixers         = _as_list(mixers)

        self.n_agents = len(self.policies)
        assert len(self.values)      == self.n_agents
        assert len(self.target_pols) == self.n_agents
        assert len(self.target_vals) == self.n_agents
        # mixers can be one-per-agent or a single shared mixer
        if len(self.mixers) == 1 and self.n_agents > 1:
            self.mixers = self.mixers * self.n_agents
        assert len(self.mixers) == self.n_agents

        # Send modules to device, set eval for targets
        for i in range(self.n_agents):
            self.policies[i].to(self.device)
            self.values[i].to(self.device)
            self.target_pols[i].to(self.device)
            self.target_vals[i].to(self.device)
            self.mixers[i].to(self.device)
            self.target_pols[i].eval()
            self.target_vals[i].eval()

        # Per-agent optimizers
        self.v_optim = [torch.optim.Adam(self.values[i].parameters(),  lr=value_lr)  for i in range(self.n_agents)]
        self.p_optim = [torch.optim.Adam(self.policies[i].parameters(), lr=policy_lr) for i in range(self.n_agents)]

        # Build bus index maps from env (used to fetch neighbor voltages)
        # env.injection_bus is typically a list/array of bus ids; env.bus_to_idx maps bus id -> column index
        # env.neighs maps bus id -> list of neighbor bus ids (1-hop)
        assert hasattr(env, "injection_bus"), "env must expose 'injection_bus'."
        # a robust map bus_id -> column index in the state vector
        self.bus_to_idx = dict(getattr(env, "bus_to_idx", {}))
        if not self.bus_to_idx:
            # fallback: assume the i-th agent corresponds to injection_bus[i]
            self.bus_to_idx = {int(b): i for i, b in enumerate(list(env.injection_bus))}
        self.neighs = dict(getattr(env, "neighs", {}))

    def _neighbor_idxs(self, agent_idx: int) -> List[int]:
        bus_id = int(self.env.injection_bus[agent_idx])
        nb = self.neighs.get(bus_id, [])
        return [self.bus_to_idx[b] for b in nb if b in self.bus_to_idx]

    def _to_t(self, x) -> Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def train_step(self, replay: ReplayBufferPI, batch_size: int):
        if len(replay) < batch_size:
            return

        S, A, LA, R, S2, D = replay.sample(batch_size)
        # Shapes: S,S2 [B,N]; A [B,N]; R,D [B,1]
        S  = self._to_t(S)
        A  = self._to_t(A)
        R  = self._to_t(R)
        S2 = self._to_t(S2)
        D  = self._to_t(D)

        with torch.no_grad():
            gamma_term = (1.0 - D) * self.gamma

        # ---- Update each agent independently ----
        for i in range(self.n_agents):
            # local signals
            v_i      = S[:, i:i+1]     # [B,1]
            a_i      = A[:, i:i+1]     # [B,1]
            v_i_next = S2[:, i:i+1]    # [B,1]

            # neighbor context (current and next)
            nb_idx = self._neighbor_idxs(i)
            if len(nb_idx) == 0:
                v_nb      = torch.zeros((S.size(0), 1, 1), device=self.device)
                v_nb_next = torch.zeros((S.size(0), 1, 1), device=self.device)
            else:
                v_nb      = S[:,  nb_idx].unsqueeze(-1)   # [B,K,1]
                v_nb_next = S2[:, nb_idx].unsqueeze(-1)   # [B,K,1]

            # target action for bootstrapping
            with torch.no_grad():
                m_next = self.mixers[i](v_i_next, v_nb_next)           # [B,CTX]
                u_next = self.target_pols[i](v_i_next, m_next)         # [B,1]
                q_tgt  = self.target_vals[i](torch.cat([v_i_next, u_next], dim=1))  # [B,1]
                y      = R + gamma_term * q_tgt

            # ----- Critic update -----
            q_pred = self.values[i](torch.cat([v_i, a_i], dim=1))
            v_loss = torch.nn.functional.mse_loss(q_pred, y)
            self.v_optim[i].zero_grad(set_to_none=True)
            v_loss.backward()
            clip_grad_norm_(self.values[i].parameters(), max_norm=5.0)
            self.v_optim[i].step()

            # ----- Actor update (deterministic policy gradient) -----
            m_cur = self.mixers[i](v_i, v_nb)
            u_pi  = self.policies[i](v_i, m_cur)                       # [B,1]
            q_pi  = self.values[i](torch.cat([v_i, u_pi], dim=1))
            # maximize q_pi => minimize -q_pi
            p_loss = -q_pi.mean()
            self.p_optim[i].zero_grad(set_to_none=True)
            p_loss.backward()
            clip_grad_norm_(self.policies[i].parameters(), max_norm=5.0)
            self.p_optim[i].step()

            # ----- Soft update targets -----
            _soft_update(self.target_vals[i], self.values[i], self.soft_tau)
            _soft_update(self.target_pols[i], self.policies[i], self.soft_tau)
