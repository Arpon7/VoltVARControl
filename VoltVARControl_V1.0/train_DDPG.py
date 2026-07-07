# train_DDPG.py — UPDATED (handles 3/4/5-tuple env returns)
# - Fix: step_Preward unpacking supports (s,r,d,info) and (s,r,d,trunc,info)
# - Matches safeDDPG.DDPG signature (no optimizer kwargs)
# - Context m has shape (1, ctx_dim)
# - Supports cases 13, 14, 123 with FAST/FINAL presets

import os, json, random, argparse
import numpy as np
import torch
import pandas as pd
from scipy.io import loadmat

from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from env_single_phase_14bus import IEEE14bus, create_14bus

from hybrid_safe_ddpg import AttentionMixer, SafePolicyNetworkWithCoord
from safeDDPG import DDPG, ReplayBufferPI

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- defaults / presets -----------------
VMIN, VMAX = 0.95, 1.05
DEF_CTX_DIM, DEF_HIDDEN_DIM = 12, 64
DEF_DQ_MAX = 0.25
DEF_VALUE_LR, DEF_POLICY_LR, DEF_SOFT_TAU = 3e-4, 1e-4, 1e-2
DEF_EPISODES, DEF_STEPS_PER_EP = 20, 180
DEF_REPLAY_CAPACITY, DEF_BATCH_SIZE = 20_000, 96
DEF_WARMUP_STEPS, DEF_EXP_NOISE_STD = 400, 0.05

PRESET_FINAL = {
    "13": dict(ctx_dim=16, hidden_dim=96, dq_max=0.30, value_lr=2e-4, policy_lr=7.5e-5, soft_tau=5e-3,
               episodes=80, steps_per_ep=240, batch_size=96, replay_capacity=80_000, warmup_steps=1500, exp_noise_std=0.04),
    "123": dict(ctx_dim=24, hidden_dim=128, dq_max=0.25, value_lr=2e-4, policy_lr=7.5e-5, soft_tau=7.5e-3,
                episodes=120, steps_per_ep=300, batch_size=128, replay_capacity=120_000, warmup_steps=2500, exp_noise_std=0.04),
    "14": dict(ctx_dim=16, hidden_dim=96, dq_max=0.30, value_lr=2e-4, policy_lr=7.5e-5, soft_tau=5e-3,
               episodes=80, steps_per_ep=240, batch_size=96, replay_capacity=80_000, warmup_steps=1500, exp_noise_std=0.04),
}

PRESET_FAST = {
    "13": dict(
        ctx_dim=12, hidden_dim=64, dq_max=0.25,
        value_lr=3e-4, policy_lr=1e-4, soft_tau=1e-2,
        episodes=20, steps_per_ep=180, batch_size=96,
        replay_capacity=20_000, warmup_steps=400, exp_noise_std=0.05
    ),
    "123": dict(
        ctx_dim=20, hidden_dim=96, dq_max=0.25,
        value_lr=3e-4, policy_lr=1e-4, soft_tau=1e-2,
        episodes=28, steps_per_ep=220, batch_size=112,
        replay_capacity=40_000, warmup_steps=600, exp_noise_std=0.05
    ),
    "14": dict(
        ctx_dim=12, hidden_dim=64, dq_max=0.25,
        value_lr=3e-4, policy_lr=1e-4, soft_tau=1e-2,
        episodes=20, steps_per_ep=180, batch_size=96,
        replay_capacity=20_000, warmup_steps=400, exp_noise_std=0.05
    ),
}

MAT_P_PATH = os.path.join("pandapower models", "aggr_p.mat")
MAT_Q_PATH = os.path.join("pandapower models", "aggr_q.mat")
MAT_PV_PATH = os.path.join("pandapower models", "PV_profile.mat")

def parse_args():
    p = argparse.ArgumentParser(description="Safe multi-agent DDPG trainer")
    p.add_argument("--case", choices=["13","14","123","both"], default="both")
    p.add_argument("--preset", choices=["none","final","fast"], default="none")
    p.add_argument("--tune", action="store_true")

    p.add_argument("--episodes", type=int, default=DEF_EPISODES)
    p.add_argument("--steps-per-ep", type=int, default=DEF_STEPS_PER_EP)
    p.add_argument("--batch-size", type=int, default=DEF_BATCH_SIZE)
    p.add_argument("--replay-capacity", type=int, default=DEF_REPLAY_CAPACITY)
    p.add_argument("--warmup-steps", type=int, default=DEF_WARMUP_STEPS)
    p.add_argument("--exp-noise-std", type=float, default=DEF_EXP_NOISE_STD)

    p.add_argument("--ctx-dim", type=int, default=DEF_CTX_DIM)
    p.add_argument("--hidden-dim", type=int, default=DEF_HIDDEN_DIM)
    p.add_argument("--dq-max", type=float, default=DEF_DQ_MAX)
    p.add_argument("--value-lr", type=float, default=DEF_VALUE_LR)
    p.add_argument("--policy-lr", type=float, default=DEF_POLICY_LR)
    p.add_argument("--soft-tau", type=float, default=DEF_SOFT_TAU)
    return p.parse_args()

def _load_mat_1d(path, prefer=("p","q","actual_PV_profile","pv")):
    d = loadmat(path)
    for k in prefer:
        if k in d: return np.array(d[k]).squeeze()
    for k, v in d.items():
        if k.startswith("__"): continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            return v.squeeze()
    raise ValueError(f"No numeric array in {path}")

def load_doe_timeseries():
    p  = _load_mat_1d(MAT_P_PATH,  prefer=("p",))
    q  = _load_mat_1d(MAT_Q_PATH,  prefer=("q",))
    pv = _load_mat_1d(MAT_PV_PATH, prefer=("actual_PV_profile","pv"))
    T = int(min(len(p), len(q), len(pv)))
    return p[:T], q[:T], pv[:T]

def infer_scale(arr, name):
    a = np.asarray(arr, float).ravel()
    mean_abs = float(np.mean(np.abs(a)))
    if 0.2 <= mean_abs <= 50.0:  scale = 1.0
    elif mean_abs >= 200.0:      scale = 0.001
    else:                        scale = 1.0
    print(f"[scale] {name}: mean={mean_abs:.3f} -> SCALE={scale}")
    return scale

def build_env(case: str):
    if case == "13":
        net = create_13bus()
        inj = [2,7,9]
        env = IEEE13bus(pp_net=net, injection_bus=inj, v0=1.0,
                        vmax=VMAX, vmin=VMIN, all_bus=False,
                        disturb_cfg=dict(enabled=False),
                        meas_noise_std=0.0, q_limits=(-2.0, 2.0))
    elif case == "123":
        inj = (np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61]) - 1).tolist()
        env = IEEE123bus(pp_net=create_123bus(), injection_bus=inj, v0=1.0,
                         vmax=VMAX, vmin=VMIN, meas_noise_std=0.0, q_limits=(-2.0, 2.0))
    elif case == "14":
        net = create_14bus()
        inj = [2,3,8,13]  # 0-based (≈ 3,4,9,14)
        env = IEEE14bus(pp_net=net, injection_bus=inj, v0=1.0,
                        vmax=VMAX, vmin=VMIN,
                        disturb_cfg=dict(enabled=False),
                        meas_noise_std=0.0, q_limits=(-2.0, 2.0))
    else:
        raise ValueError("case must be '13', '14' or '123'")
    return env, inj

def build_agents(n_agents, env, ctx_dim, hidden_dim, dq_max, value_lr, policy_lr, soft_tau):
    policies = [SafePolicyNetworkWithCoord(1, ctx_dim, hidden_dim, VMIN, VMAX, dq_max).to(DEVICE)
                for _ in range(n_agents)]
    values   = [torch.nn.Sequential(
                    torch.nn.Linear(2, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 1)
               ).to(DEVICE) for _ in range(n_agents)]
    mixers   = [AttentionMixer(in_dim=1, hid_dim=ctx_dim).to(DEVICE) for _ in range(n_agents)]

    # targets
    target_pols = [SafePolicyNetworkWithCoord(1, ctx_dim, hidden_dim, VMIN, VMAX, dq_max).to(DEVICE)
                   for _ in range(n_agents)]
    target_vals = [torch.nn.Sequential(
                    torch.nn.Linear(2, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 1)
                   ).to(DEVICE) for _ in range(n_agents)]
    for i in range(n_agents):
        target_pols[i].load_state_dict(policies[i].state_dict())
        target_vals[i].load_state_dict(values[i].state_dict())

    ddpg = DDPG(
        policy_net=policies,
        value_net=values,
        target_policy_net=target_pols,
        target_value_net=target_vals,
        mixers=mixers,
        env=env,
        value_lr=value_lr,
        policy_lr=policy_lr,
        gamma=0.99,
        soft_tau=soft_tau,
        device=DEVICE
    )
    return policies, mixers, ddpg

def _neighbor_mean(state_vec, bus_idx, inj, neighs):
    """Mean of neighbor voltages including self, based on 1-hop graph."""
    try:
        neigh = neighs[bus_idx]
        mask = np.in1d(inj, neigh)
        if mask.any():
            return float(np.mean(state_vec[mask]))
    except Exception:
        pass
    return float(state_vec[np.where(np.array(inj)==bus_idx)[0][0]])

def _step_env(env, action_vec):
    """
    Normalize env.step_Preward outputs to (next_state, reward, done, info).
    Robust to:
      • signature: step_Preward(action)  OR  step_Preward(action, p_action)
      • tuple order: (s,r,d,info), (s,r,d),  or (s,r,zeros_vec,d) used by IEEE123bus
      • done type: scalar or vector
    """
    import numpy as np
    import inspect

    def _coerce_done(d):
        try:
            return bool(np.asarray(d).any())  # any agent done
        except Exception:
            return bool(d)

    # Call with correct signature
    try:
        out = env.step_Preward(action_vec)
    except TypeError:
        # env (like IEEE123bus) requires p_action
        p_action = np.zeros_like(action_vec, dtype=float)
        out = env.step_Preward(action_vec, p_action)

    if not isinstance(out, tuple):
        raise RuntimeError("env.step_Preward must return a tuple.")

    # Handle common return shapes
    if len(out) == 5:
        ns, r, done, trunc, info = out
        return ns, r, _coerce_done(done) or _coerce_done(trunc), info

    if len(out) == 4:
        # Could be (ns, r, done, info) OR (ns, r, zeros_vec, done)
        a, b, c, d = out
        # If c looks like a vector and d is scalar bool -> treat as (ns, r, done)
        c_arr = np.asarray(c)
        if c_arr.ndim > 0 and c_arr.size > 1 and isinstance(d, (bool, np.bool_)):
            return a, b, _coerce_done(d), {}
        # Otherwise assume (ns, r, done, info)
        return a, b, _coerce_done(c), (d if isinstance(d, dict) else {})

    if len(out) == 3:
        ns, r, done = out
        return ns, r, _coerce_done(done), {}

    raise RuntimeError(f"Unexpected step return length: {len(out)}")



def train_one_case(case: str, p_series, q_series, pv_series, args):
    LOAD_P_SCALE = infer_scale(p_series, "aggr_p")
    LOAD_Q_SCALE = infer_scale(q_series, "aggr_q")
    PV_P_SCALE   = infer_scale(pv_series, "PV")

    env, inj = build_env(case)
    n_agents = len(inj)

    ckpt_dir = os.path.join("checkpoints", "single-phase", f"{case}bus", "safe-ddpg")
    os.makedirs(ckpt_dir, exist_ok=True)
    TRAIN_LOG_CSV = os.path.join(ckpt_dir, "train_log.csv")

    policies, mixers, ddpg = build_agents(
        n_agents, env,
        ctx_dim=args.ctx_dim, hidden_dim=args.hidden_dim, dq_max=args.dq_max,
        value_lr=args.value_lr, policy_lr=args.policy_lr, soft_tau=args.soft_tau
    )
    rb = ReplayBufferPI(args.replay_capacity)

    logs = []
    T = len(p_series)
    for ep in range(args.episodes):
        start = np.random.randint(0, max(1, T - args.steps_per_ep - 1))

        # Reset and set loads/PV for this episode
        state = env.reset()
        # If your env lacks step_load, you can remove this block or adapt it.
        if hasattr(env, "step_load"):
            state, _, _, _ = env.step_load(
                np.zeros(n_agents),
                load_p=float(p_series[start]) * LOAD_P_SCALE,
                load_q=float(q_series[start]) * LOAD_Q_SCALE,
                pv_p  =float(pv_series[start]) * PV_P_SCALE
            )

        last_action = np.zeros(n_agents, dtype=np.float32)
        ep_return, ep_cost = 0.0, 0.0

        for t in range(args.steps_per_ep):
            # Build per-agent context vectors (1 x ctx_dim) from neighbor-aware scalar
            actions = []
            with torch.no_grad():
                for i in range(n_agents):
                    v_i_scalar = float(state[i])
                    bus_idx = inj[i]
                    neigh_scalar = _neighbor_mean(state, bus_idx, inj, getattr(env, "neighs", [[]]*len(state)))
                    m_np = np.full((1, args.ctx_dim), neigh_scalar, dtype=np.float32)  # (1, ctx_dim)
                    v_i = torch.tensor([v_i_scalar], dtype=torch.float32, device=DEVICE)  # (1,)
                    m_i = torch.tensor(m_np, dtype=torch.float32, device=DEVICE)          # (1, ctx_dim)
                    a_i = policies[i].get_action(v_i, m_i)
                    actions.append(float(a_i))
            actions = np.asarray(actions, dtype=np.float32)
            if args.exp_noise_std > 0:
                actions += np.random.normal(0.0, args.exp_noise_std, size=actions.shape).astype(np.float32)
            actions = np.clip(actions, -args.dq_max, args.dq_max)

            # Env step (robust to return length)
            next_state, r, done, info = _step_env(env, actions)
            ep_return += float(r)
            ep_cost   += -float(r)

            # Replay buffer: (s, a, la, r, s2, d)
            rb.push(
                state.astype(np.float32),  # shape: (n_agents,)
                actions.astype(np.float32),  # shape: (n_agents,)
                last_action.astype(np.float32),  # shape: (n_agents,)
                float(r),
                next_state.astype(np.float32),  # shape: (n_agents,)
                float(done)
            )

            state = next_state
            last_action = actions.astype(np.float32)

            # Learn after warmup
            if len(rb) > max(args.batch_size, args.warmup_steps):
                ddpg.train_step(rb, args.batch_size)

        logs.append({"episode": int(ep), "return": float(ep_return), "cost": float(ep_cost)})
        print(f"[{case}] Ep {ep:3d} | return {ep_return: .3f} | cost {ep_cost: .3f}")

        # periodic checkpoints
        if (ep+1) % 3 == 0 or ep == args.episodes-1:
            for i, pi in enumerate(policies):
                torch.save(pi.state_dict(), os.path.join(ckpt_dir, f"policy_agent{i}.pth"))
            for i, mx in enumerate(mixers):
                torch.save(mx.state_dict(), os.path.join(ckpt_dir, f"mixer_agent{i}.pth"))
            with open(os.path.join(ckpt_dir, "best_hparams.json"), "w") as f:
                json.dump(dict(ctx_dim=args.ctx_dim, hidden=args.hidden_dim, dq_max=args.dq_max,
                               value_lr=args.value_lr, policy_lr=args.policy_lr, soft_tau=args.soft_tau), f, indent=2)

    pd.DataFrame(logs).to_csv(TRAIN_LOG_CSV, index=False)
    print(f"[{case}] Training complete -> {TRAIN_LOG_CSV} | ckpts -> {ckpt_dir}")

def main():
    args = parse_args()

    cases = ("13","123") if args.case == "both" else (args.case,)
    for case in cases:
        p_series, q_series, pv_series = load_doe_timeseries()

        if args.preset in ("final","fast"):
            fp = (PRESET_FINAL if args.preset=="final" else PRESET_FAST)[case]
            class _NS: pass
            ns = _NS()
            ns.case = case
            ns.ctx_dim = fp["ctx_dim"]; ns.hidden_dim = fp["hidden_dim"]; ns.dq_max = fp["dq_max"]
            ns.value_lr = fp["value_lr"]; ns.policy_lr = fp["policy_lr"]; ns.soft_tau = fp["soft_tau"]
            ns.episodes = fp["episodes"]; ns.steps_per_ep = fp["steps_per_ep"]
            ns.batch_size = fp["batch_size"]; ns.replay_capacity = fp["replay_capacity"]
            ns.warmup_steps = fp["warmup_steps"]; ns.exp_noise_std = fp["exp_noise_std"]
            train_one_case(case, p_series, q_series, pv_series, ns)
        else:
            train_one_case(case, p_series, q_series, pv_series, args)

if __name__ == "__main__":
    main()
