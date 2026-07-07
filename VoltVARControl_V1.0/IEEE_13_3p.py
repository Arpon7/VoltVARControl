# IEEE_13_3p.py  — portable, fast DOE runner for IEEE-13 (single-phase) with projector
# - STRIDE=5 for fewer samples
# - projector: diag_step=0.25, max_bsearch_steps=4
# - relies on updated env_single_phase_13bus.py (BFSW + uniform split)
# - saves results/trajectory_doe_ieee13_stable.{csv,png}

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env_single_phase_13bus import IEEE13bus, create_13bus
from hybrid_safe_ddpg import AttentionMixer, SafePolicyNetworkWithCoord

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VMIN, VMAX = 0.95, 1.05
Q_LIMITS = (-5.0, 5.0)
STRIDE = 5  # <<< process every 5th sample

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- portable .mat loading --------
def find_data_file(name: str, extra=None) -> str:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    here, root = BASE, BASE.parent
    env_dir = os.environ.get("DATA_DIR")
    cands = []
    if env_dir:
        cands += [Path(env_dir) / name]
    cands += [
        here / name, root / name,
        here / "data" / name, here / "datasets" / name,
        root / "data" / name, root / "datasets" / name,
    ]
    if extra: cands = list(extra) + cands
    for c in cands:
        c = Path(c)
        if c.exists(): return str(c)
        if c.suffix.lower() == ".mat":
            alt = c.with_suffix(".MAT")
            if alt.exists(): return str(alt)
    raise FileNotFoundError(f"Could not find '{name}'. Set DATA_DIR or place in ./data or ../data.")

def load_mat_1d(filename, prefer=("p","q","actual_PV_profile","pv")) -> np.ndarray:
    path = find_data_file(filename)
    d = loadmat(path)
    for k in prefer:
        if k in d: return np.array(d[k]).squeeze()
    for k, v in d.items():
        if k.startswith("__"): continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            return np.array(v).squeeze()
    raise ValueError(f"No numeric array found in '{path}'")

def infer_scale(arr, name):
    a = np.asarray(arr, float).ravel()
    mean_abs = float(np.mean(np.abs(a)))
    if 0.2 <= mean_abs <= 50.0:  scale = 1.0
    elif mean_abs >= 200.0:      scale = 0.001
    else:                        scale = 1.0
    print(f"[scale] {name}: mean={mean_abs:.3f} -> SCALE={scale}")
    return scale

# -------- models --------
CKPT_DIR = BASE / "checkpoints" / "single-phase" / "13bus" / "safe-ddpg"
HP_JSON  = CKPT_DIR / "best_hparams.json"

def _find_ckpt(kind, i):
    fname = f"policy_agent{i}.pth" if kind == "policy" else f"mixer_agent{i}.pth"
    path = CKPT_DIR / fname
    return str(path) if path.exists() else None

def load_models(n_agents, vmin=VMIN, vmax=VMAX, defaults=(16, 64, 0.25)):
    ctx, hid, dq = defaults
    if HP_JSON.exists():
        import json
        hp = json.loads(HP_JSON.read_text())
        ctx = int(hp.get("ctx_dim", ctx))
        hid = int(hp.get("hidden", hid))
        dq  = float(hp.get("dq_max", dq))
    policies = [SafePolicyNetworkWithCoord(1, ctx, hid, vmin, vmax, dq).to(DEVICE) for _ in range(n_agents)]
    mixers   = [AttentionMixer(in_dim=1, hid_dim=ctx).to(DEVICE) for _ in range(n_agents)]
    loaded = False
    for i in range(n_agents):
        ppth = _find_ckpt("policy", i)
        mpth = _find_ckpt("mixer",  i)
        if ppth and mpth:
            policies[i].load_state_dict(torch.load(ppth, map_location=DEVICE), strict=True)
            mixers[i].load_state_dict(torch.load(mpth, map_location=DEVICE), strict=True)
            loaded = True
        policies[i].eval(); mixers[i].eval()
    return policies, mixers, dq, loaded

@torch.no_grad()
def actor_actions(state_vec, policies, mixers, dq_max, env, inj):
    V = torch.tensor(state_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    outs = []
    for i in range(len(inj)):
        v_i  = V[:, i:i+1]
        nb = env.neighs.get(int(inj[i]), [])
        if not nb:
            v_nb = torch.zeros((1,1,1), dtype=torch.float32, device=DEVICE)
        else:
            idxs = [env.bus_to_idx[b] for b in nb if b in env.bus_to_idx]
            v_nb = V[:, idxs].unsqueeze(-1) if idxs else torch.zeros((1,1,1), dtype=torch.float32, device=DEVICE)
        m_i = mixers[i](v_i, v_nb)
        u_i = policies[i](v_i, m_i)
        outs.append(u_i)
    U = torch.cat(outs, dim=1).squeeze(0).cpu().numpy()
    return np.clip(U, -dq_max, dq_max)

# -------- fast projector (diag_step=0.25, max_bsearch_steps=4) --------
def _eval_q(env, q_vec, load_p, load_q, pv_p, inj):
    v_next, reward, done, info = env.step_load(q_vec, load_p, load_q, pv_p)
    applied_q = np.array([float(env.network.sgen.at[env.sgen_idx_for_inj[int(b)], "q_mvar"]) for b in inj])
    return v_next, applied_q

def project_to_band_fast(env, q_guess, load_p, load_q, pv_p, inj,
                         tol=1e-4, diag_step=0.25, max_bsearch_steps=4):
    ql, qu = Q_LIMITS
    q0 = np.clip(np.asarray(q_guess, float).reshape(-1), ql, qu)
    v0, _ = _eval_q(env, q0, load_p, load_q, pv_p, inj); v0 = np.asarray(v0, float).reshape(-1)
    hi_idx = int(np.argmax(v0 - VMAX))
    lo_idx = int(np.argmax(VMIN - v0))
    hi_gap = float(v0[hi_idx] - VMAX)
    lo_gap = float(VMIN - v0[lo_idx])
    if hi_gap <= tol and lo_gap <= tol: return v0, q0
    over_high = hi_gap > lo_gap
    i = hi_idx if over_high else lo_idx
    v_target = VMAX if over_high else VMIN
    dq_try = -diag_step if over_high else diag_step
    if dq_try > 0: dq_try = min(dq_try, qu - q0[i])
    else:          dq_try = -min(abs(dq_try), q0[i] - ql)
    if abs(dq_try) < 1e-8: return v0, q0
    q1 = q0.copy(); q1[i] = np.clip(q0[i] + dq_try, ql, qu)
    v1, _ = _eval_q(env, q1, load_p, load_q, pv_p, inj); v1 = np.asarray(v1, float).reshape(-1)
    dv, dq = float(v1[i] - v0[i]), float(q1[i] - q0[i])
    if abs(dv) < 1e-8:
        return (v1, q1) if abs(v1[i] - v_target) < abs(v0[i] - v_target) else (v0, q0)
    dq_corr = float(np.clip((v_target - v0[i]) * (dq / dv), -abs(diag_step), abs(diag_step)))
    q2 = q0.copy(); q2[i] = np.clip(q0[i] + dq_corr, ql, qu)
    v2, _ = _eval_q(env, q2, load_p, load_q, pv_p, inj); v2 = np.asarray(v2, float).reshape(-1)
    if (VMIN - tol) <= v2[i] <= (VMAX + tol): return v2, q2
    a, b = (q2[i], qu) if v2[i] < VMIN else (ql, q2[i])
    q_best, v_best = q2.copy(), v2.copy()
    for _ in range(max_bsearch_steps):
        mid = 0.5*(a+b)
        q_try = q2.copy(); q_try[i] = mid
        v_mid, _ = _eval_q(env, q_try, load_p, load_q, pv_p, inj); v_mid = np.asarray(v_mid, float).reshape(-1)
        v_i = float(v_mid[i])
        if v_i < VMIN: a = mid
        elif v_i > VMAX: b = mid
        else:
            q_best, v_best = q_try, v_mid
            break
    return v_best, q_best

# -------- plotting --------
def plot_trajectory(df, inj, out_path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for k, b in enumerate(inj):
        axs[0].plot(df["t"], df[f"v_bus_{b}"], label=f"Bus {b}")
        axs[1].plot(df["t"], df[f"q_bus_{b}"], label=f"Bus {b}")
    axs[0].hlines([VMIN, VMAX], df["t"].iloc[0], df["t"].iloc[-1], colors="gray", linestyles="--", alpha=0.6)
    axs[0].set_ylabel("Voltage [p.u.]"); axs[1].set_ylabel("Reactive Injection [MVar]")
    axs[0].set_xlabel("Time step");      axs[1].set_xlabel("Time step")
    axs[0].legend(ncol=2); axs[1].legend(ncol=2)
    axs[0].grid(True, alpha=0.3); axs[1].grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
    print(f"Saved figure -> {out_path}")

# -------- main --------
def main():
    # datasets
    p  = load_mat_1d("aggr_p.mat",  prefer=("p",))
    q  = load_mat_1d("aggr_q.mat",  prefer=("q",))
    pv = load_mat_1d("PV.mat",      prefer=("actual_PV_profile","pv"))
    T = int(min(len(p), len(q), len(pv)))
    # scaling heuristic
    load_p_scale = infer_scale(p,  "aggr_p")
    load_q_scale = infer_scale(q,  "aggr_q")
    pv_p_scale   = infer_scale(pv, "PV")
    # env (BFSW + uniform split in updated env file)
    inj = [2, 7, 9]
    env = IEEE13bus(create_13bus(), inj, v0=1.0, vmax=VMAX, vmin=VMIN,
                    all_bus=False, disturb_cfg=dict(enabled=False),
                    meas_noise_std=0.0, q_limits=Q_LIMITS)
    # models
    policies, mixers, dq_max, loaded = load_models(len(inj), defaults=(16, 64, 0.25))
    print(f"Policies loaded? {loaded}. dq_max={dq_max}")
    # rollout with STRIDE
    rows = []
    state = env.reset(seed=0)
    for t in range(0, T, STRIDE):
        u_rl = actor_actions(state, policies, mixers, dq_max, env, inj)
        v_next, q_applied = project_to_band_fast(
            env, u_rl,
            load_p=float(p[t])  * load_p_scale,
            load_q=float(q[t])  * load_q_scale,
            pv_p  =float(pv[t]) * pv_p_scale,
            inj=inj,
            diag_step=0.25,            # <<< requested
            max_bsearch_steps=4        # <<< requested
        )
        rows.append({
            "t": t,
            **{f"v_bus_{b}": v_next[i] for i, b in enumerate(inj)},
            **{f"q_bus_{b}": q_applied[i] for i, b in enumerate(inj)},
        })
        state = v_next
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "trajectory_doe_ieee13_stable.csv"
    fig_path = OUT_DIR / "trajectory_doe_ieee13_stable.png"
    df.to_csv(csv_path, index=False); print(f"Saved CSV -> {csv_path}")
    plot_trajectory(df, inj, str(fig_path))

if __name__ == "__main__":
    main()

