# test.py
# Evaluates BOTH IEEE-13 and IEEE-123 with noise:
# - measurement noise (meas_noise_std)
# - process noise (disturb_cfg) on aggregate P/Q/PV inside step_load
# Plots u(v) curves with Monte-Carlo resampling to visualize noise bands,
# runs a short rollout, and prints a simple success metric.

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from hybrid_safe_ddpg import AttentionMixer, SafePolicyNetworkWithCoord

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VMIN, VMAX = 0.95, 1.05
plt.rcParams["font.size"] = 14

# ---------------- Noise knobs (edit here) ----------------
MEAS_NOISE_STD = 0.02     # measurement noise on volt readings (both cases)
DISTURB_EVAL = dict(      # process noise on P/Q/PV (both cases)
    enabled=True,
    load_sigma=0.08,      # MW (after scaling in your pipeline)
    pv_sigma=0.06,        # MW
    load_step_prob=0.03,  # per step
    pv_step_prob=0.02,
    load_step_scale=0.20, # MW
    pv_step_scale=0.15,   # MW
)

# Monte-Carlo for u(v) band visualization
UVOLTS_SAMPLES = 21       # volt samples per curve
MC_SAMPLES     = 12       # number of noisy draws to show per bus
LINE_ALPHA     = 0.35
UVIS_NO_CLAMP  = True     # show raw (unclamped) policy outputs in u(v) plots

def case_cfg(case):
    if case == "13":
        inj = [2, 7, 9]
        build_env = lambda: IEEE13bus(
            create_13bus(), inj, v0=1.0, vmax=VMAX, vmin=VMIN,
            all_bus=False,
            disturb_cfg=DISTURB_EVAL,
            meas_noise_std=MEAS_NOISE_STD
        )
    else:
        inj = (np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61]) - 1).tolist()
        build_env = lambda: IEEE123bus(
            create_123bus(), inj, v0=1.0, vmax=VMAX, vmin=VMIN,
            meas_noise_std=MEAS_NOISE_STD,
            disturb_cfg=DISTURB_EVAL
        )
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "single-phase", f"{case}bus", "safe-ddpg")
    hp_json = os.path.join(ckpt_dir, "best_hparams.json")
    return inj, build_env, ckpt_dir, hp_json

def load_hparams(hp_json, defaults=(16, 64, 0.30)):
    ctx, hid, dq = defaults
    if os.path.exists(hp_json):
        with open(hp_json, "r") as f:
            hp = json.load(f)
        ctx = int(hp.get("ctx_dim", ctx))
        hid = int(hp.get("hidden", hid))
        dq  = float(hp.get("dq_max", dq))
    return ctx, hid, dq

def _find_ckpt(ckpt_dir, kind, i):
    fname = "policy_agent{}.pth".format(i) if kind == "policy" else "mixer_agent{}.pth".format(i)
    path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {kind} checkpoint {path}")
    return path

def load_models(n_agents, ctx_dim, hidden_dim, dq_max, ckpt_dir):
    policies = [SafePolicyNetworkWithCoord(1, ctx_dim, hidden_dim, VMIN, VMAX, dq_max).to(DEVICE)
                for _ in range(n_agents)]
    mixers   = [AttentionMixer(in_dim=1, hid_dim=ctx_dim).to(DEVICE)
                for _ in range(n_agents)]
    for i in range(n_agents):
        policies[i].load_state_dict(torch.load(_find_ckpt(ckpt_dir, "policy", i), map_location=DEVICE), strict=True)
        mixers[i].load_state_dict(torch.load(_find_ckpt(ckpt_dir, "mixer",  i), map_location=DEVICE), strict=True)
        policies[i].eval(); mixers[i].eval()
    return policies, mixers

@torch.no_grad()
def neighbors_tensor(env, V_1xN: torch.Tensor, agent_idx: int, inj):
    bus_id = int(inj[agent_idx])
    nb = env.neighs.get(bus_id, [])
    if not nb:
        return torch.zeros((1, 1, 1), dtype=torch.float32, device=V_1xN.device)
    idxs = [env.bus_to_idx[b] for b in nb if b in env.bus_to_idx]
    if len(idxs) == 0:
        return torch.zeros((1, 1, 1), dtype=torch.float32, device=V_1xN.device)
    return V_1xN[:, idxs].unsqueeze(-1)

@torch.no_grad()
def compute_actions(env, policies, mixers, state_vec: np.ndarray, dq_max: float, inj, clamp=True) -> np.ndarray:
    V = torch.tensor(state_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    outs = []
    for i in range(len(inj)):
        v_i  = V[:, i:i+1]
        v_nb = neighbors_tensor(env, V, i, inj)
        m_i  = mixers[i](v_i, v_nb)
        u_i  = policies[i](v_i, m_i)     # raw
        outs.append(u_i)
    U = torch.cat(outs, dim=1).squeeze(0)
    if clamp:
        U = torch.clamp(U, -dq_max, dq_max)
    return U.cpu().numpy()

@torch.no_grad()
def plot_action_curves(env, policies, mixers, inj, dq_max, title=""):
    # Monte-Carlo around each v_i to visualize noise effect
    v_nom = torch.tensor(env.reset(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    s_array = np.linspace(0.80, 1.20, UVOLTS_SAMPLES, dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    total_pts, sat_pts = 0, 0

    for i, bus in enumerate(inj):
        for m in range(MC_SAMPLES):
            a_vals = []
            for v in s_array:
                V = v_nom.clone()
                V[:, i] = float(v)
                # resample measurement noise for this eval
                if MEAS_NOISE_STD > 0.0:
                    V = V + torch.randn_like(V) * MEAS_NOISE_STD
                v_i  = V[:, i:i+1]
                v_nb = neighbors_tensor(env, V, i, inj)
                m_i  = mixers[i](v_i, v_nb)
                u_i  = policies[i](v_i, m_i).item()  # raw (unclamped)
                a_vals.append(u_i)
                total_pts += 1
                if abs(u_i) >= dq_max:  # would saturate
                    sat_pts += 1
            ax.plot(s_array, a_vals, alpha=LINE_ALPHA, label=f"Policy @ bus {bus}" if m == 0 else None)

    sat_pct = 100.0 * sat_pts / max(1, total_pts)
    ax.set_xlabel("Voltage [p.u.]")
    ylabel = "Reactive injection change [MVar] (raw)" if UVIS_NO_CLAMP else "Reactive injection change [MVar]"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  —  saturation if clamped: {sat_pct:.1f}% of samples")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout(); plt.show()

def rollout(env, policies, mixers, inj, dq_max, steps=60, seed=0, max_action_step=0.30):
    state = env.reset(seed)
    last_action = np.zeros((len(inj),), dtype=np.float32)
    states  = [state]
    actions = []
    for _ in range(steps):
        action = compute_actions(env, policies, mixers, state, dq_max, inj, clamp=True)
        action = np.clip(action, -max_action_step, max_action_step)
        cmd = last_action - action
        next_state, reward, reward_sep, done = env.step_Preward(cmd, (last_action - cmd))
        actions.append(cmd)
        states.append(next_state)
        last_action = cmd
        state = next_state
        if done:
            break
    return np.asarray(states), np.asarray(actions)

def plot_rollout(states, actions, inj, title=""):
    T = actions.shape[0]
    if T == 0:
        print("[warn] rollout produced 0 steps. Plotting initial state only.")
        T = 1
        actions = np.zeros((1, len(inj)), dtype=float)
    xs = np.arange(T)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for i, bus in enumerate(inj):
        axs[0].plot(xs, states[:T, i], label=f"Bus {bus}")
        axs[1].plot(xs, actions[:, i], label=f"Bus {bus}")
    axs[0].hlines([VMIN, VMAX], 0, T-1, colors="gray", linestyles="--", alpha=0.6)
    axs[0].set_xlabel("Steps"); axs[0].set_ylabel("Voltage [p.u.]")
    axs[1].set_xlabel("Steps"); axs[1].set_ylabel("Reactive Injection [MVar]")
    axs[0].legend(); axs[1].legend()
    axs[0].grid(True, alpha=0.3); axs[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout(); plt.show()

def success_rate(env, policies, mixers, inj, dq_max, trials=40, step_limit=80, max_action_step=0.30):
    succ, steps_to_success, eval_time = 0, [], 0.0
    for rep in range(trials):
        state = env.reset(rep + 1)
        last_action = np.zeros((len(inj),), dtype=np.float32)
        for t in range(step_limit):
            t0 = time.time()
            action = compute_actions(env, policies, mixers, state, dq_max, inj, clamp=True)
            action = np.clip(action, -max_action_step, max_action_step)
            eval_time += (time.time() - t0)
            cmd = last_action - action
            next_state, reward, reward_sep, done = env.step_Preward(cmd, (last_action - cmd))
            if done:
                succ += 1
                steps_to_success.append(t + 1)
                break
            last_action = cmd
            state = next_state
    print(f"Success: {succ}/{trials}")
    if steps_to_success:
        print(f"Avg steps to success: {np.mean(steps_to_success):.2f} ± {np.std(steps_to_success):.2f}")
        denom = max(1, sum(steps_to_success)) * len(inj)
        print("Avg policy eval time: {:.4f} ms / bus-step".format(1000.0 * eval_time / denom))

def main():
    for case in ("13", "123"):
        inj, build_env, ckpt_dir, hp_json = case_cfg(case)
        ctx_dim, hidden_dim, dq_max = load_hparams(hp_json)
        env = build_env()
        policies, mixers = load_models(len(inj), ctx_dim, hidden_dim, dq_max, ckpt_dir)

        # 1) u(v) Monte-Carlo curves (noise visible)
        plot_action_curves(env, policies, mixers, inj, dq_max, title=f"u(v) curves — IEEE-{case}")

        # 2) Short rollout
        states, actions = rollout(env, policies, mixers, inj, dq_max, steps=60, seed=0, max_action_step=dq_max)
        plot_rollout(states, actions, inj, title=f"Closed-loop rollout — IEEE-{case}")

        # 3) Success stats
        success_rate(env, policies, mixers, inj, dq_max, trials=40, step_limit=80, max_action_step=dq_max)

if __name__ == "__main__":
    main()
