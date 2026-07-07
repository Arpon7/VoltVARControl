# test_real_data.py
# DOE playback for 13- and 123-bus with neighbor-aware actor and a FAST hard-band projector.
import os, json, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from hybrid_safe_ddpg import AttentionMixer, SafePolicyNetworkWithCoord

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VMIN, VMAX = 0.95, 1.05
EPS_V = 1e-4
Q_LIMITS = (-5.0, 5.0)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_P, DATA_Q, DATA_PV = [os.path.join(BASE_DIR, f) for f in ("aggr_p.mat","aggr_q.mat","PV.mat")]
OUT_DIR = os.path.join(BASE_DIR, "results"); os.makedirs(OUT_DIR, exist_ok=True)

# ---------- utilities ----------
def _load_mat_1d(path, prefer=("p","q","actual_PV_profile","pv")):
    d = loadmat(path)
    for k in prefer:
        if k in d: return np.array(d[k]).squeeze()
    for k,v in d.items():
        if k.startswith("__"): continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            return np.array(v).squeeze()
    raise ValueError(f"No numeric array found in {path}")

def load_doe():
    for pth in (DATA_P, DATA_Q, DATA_PV):
        if not os.path.exists(pth):
            raise FileNotFoundError(pth)
    p  = _load_mat_1d(DATA_P,  ("p",))
    q  = _load_mat_1d(DATA_Q,  ("q",))
    pv = _load_mat_1d(DATA_PV, ("actual_PV_profile","pv"))
    T  = int(min(map(len, (p,q,pv))))
    return p[:T], q[:T], pv[:T]

def infer_scale(arr, name):
    a = np.asarray(arr, float).ravel()
    mean_abs = float(np.mean(np.abs(a)))
    if 0.2 <= mean_abs <= 50.0:  return 1.0
    if mean_abs >= 200.0:        return 0.001   # kW → MW style
    return 1.0

def case_cfg(case):
    if case == "13":
        inj = [2,7,9]
        env = IEEE13bus(create_13bus(), inj, v0=1.0, vmax=VMAX, vmin=VMIN,
                        meas_noise_std=0.0, q_limits=Q_LIMITS)
    else:
        inj = (np.array([10,11,16,20,33,36,48,59,66,75,83,92,104,61]) - 1).tolist()
        env = IEEE123bus(create_123bus(), inj, v0=1.0, vmax=VMAX, vmin=VMIN,
                         meas_noise_std=0.0, q_limits=Q_LIMITS)
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints", "single-phase", f"{case}bus", "safe-ddpg")
    return env, inj, ckpt_dir

def _ckpt(ckpt_dir, kind, i):
    f = "policy_agent{}.pth".format(i) if kind=="policy" else "mixer_agent{}.pth".format(i)
    p = os.path.join(ckpt_dir, f)
    if not os.path.exists(p): raise FileNotFoundError(p)
    return p

def load_models(n, ckpt_dir, vmin=VMIN, vmax=VMAX, ctx=16, hid=64, dq_max=0.30):
    pols  = [SafePolicyNetworkWithCoord(1, ctx, hid, vmin, vmax, dq_max).to(DEVICE) for _ in range(n)]
    mixrs = [AttentionMixer(in_dim=1, hid_dim=ctx).to(DEVICE) for _ in range(n)]
    for i in range(n):
        pols[i].load_state_dict(torch.load(_ckpt(ckpt_dir,"policy",i), map_location=DEVICE), strict=True)
        mixrs[i].load_state_dict(torch.load(_ckpt(ckpt_dir,"mixer", i), map_location=DEVICE), strict=True)
        pols[i].eval(); mixrs[i].eval()
    return pols, mixrs

@torch.no_grad()
def neighbor_tensor(env, V_1xN: torch.Tensor, idx: int, inj):
    nb = env.neighs.get(int(inj[idx]), [])
    if not nb: return torch.zeros((1,1,1), dtype=torch.float32, device=V_1xN.device)
    cols = [env.bus_to_idx[b] for b in nb if b in env.bus_to_idx]
    if len(cols) == 0: return torch.zeros((1,1,1), dtype=torch.float32, device=V_1xN.device)
    return V_1xN[:, cols].unsqueeze(-1)

@torch.no_grad()
def actor_delta_q(state_vec, env, inj, pols, mixrs, dq_max):
    V = torch.tensor(state_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    outs = []
    for i in range(len(inj)):
        v_i  = V[:, i:i+1]
        v_nb = neighbor_tensor(env, V, i, inj)
        m_i  = mixrs[i](v_i, v_nb)
        u_i  = pols[i](v_i, m_i)  # raw ΔQ
        outs.append(u_i)
    U = torch.cat(outs, dim=1).squeeze(0)
    U = torch.clamp(U, -dq_max, dq_max)  # rate limit
    return U.cpu().numpy()

# ---------------- FAST projector ----------------
def _eval_q(env, q_abs, load_p, load_q, pv_p, inj):
    v_meas, _, _, _ = env.step_load(q_abs, load_p, load_q, pv_p)
    return v_meas, np.array([env.network.sgen.at[env.sgen_idx_for_inj[int(b)], "q_mvar"] for b in inj])

def _current_q(env, inj):
    return np.array([env.network.sgen.at[env.sgen_idx_for_inj[int(b)], "q_mvar"] for b in inj], dtype=float)

def project_to_band_fast(env, u_delta, load_p, load_q, pv_p, inj,
                         diag_step=0.25, max_bsearch_steps=4, tol=1e-4):
    # 1) start from current |q|
    q0 = _current_q(env, inj)
    q1 = np.clip(q0 + np.asarray(u_delta, float).reshape(-1), Q_LIMITS[0], Q_LIMITS[1])
    v1, q1 = _eval_q(env, q1, load_p, load_q, pv_p, inj)
    v1 = np.asarray(v1, float).reshape(-1)

    # if all inside band, done
    if (VMIN - tol) <= v1.min() and v1.max() <= (VMAX + tol):
        return v1, q1

    # 2) one-shot diagonal correction
    q2 = q1.copy()
    for i in range(len(inj)):
        if v1[i] < VMIN - tol:
            q2[i] = min(Q_LIMITS[1], q1[i] + diag_step)
        elif v1[i] > VMAX + tol:
            q2[i] = max(Q_LIMITS[0], q1[i] - diag_step)
    v2, q2 = _eval_q(env, q2, load_p, load_q, pv_p, inj)
    v2 = np.asarray(v2, float).reshape(-1)
    if (VMIN - tol) <= v2.min() and v2.max() <= (VMAX + tol):
        return v2, q2

    # 3) at most one 1D bisection on the *worst* offending bus
    i = int(np.argmax(np.maximum(v2 - VMAX, 0.0) + np.maximum(VMIN - v2, 0.0)))
    ql, qu = Q_LIMITS
    # move in right direction and then bisect
    dq = +diag_step if v2[i] < VMIN else -diag_step
    q_try = q2.copy(); q_try[i] = np.clip(q2[i] + dq, ql, qu)
    v_try, q_try = _eval_q(env, q_try, load_p, load_q, pv_p, inj)
    v_try = np.asarray(v_try, float).reshape(-1)
    if (VMIN - tol) <= v_try[i] <= (VMAX + tol):
        v2, q2 = v_try, q_try
    else:
        a, b = (q2[i], qu) if v2[i] < VMIN else (ql, q2[i])
        q_best, v_best = q2.copy(), v2.copy()
        for _ in range(max_bsearch_steps):
            mid = 0.5 * (a + b)
            q_mid = q2.copy(); q_mid[i] = mid
            v_mid, q_mid = _eval_q(env, q_mid, load_p, load_q, pv_p, inj)
            v_i = float(np.asarray(v_mid, float).reshape(-1)[i])
            if v_i < VMIN: a = mid
            elif v_i > VMAX: b = mid
            else: q_best, v_best = q_mid, v_mid; break
        v2, q2 = v_best, q_best
    return v2, q2

# ---------------- main playback ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["13","123"], default="both")
    parser.add_argument("--save-prefix", default="trajectory_doe")
    args = parser.parse_args()

    P, Q, PV = load_doe()
    lp_s, lq_s, pv_s = infer_scale(P,"P"), infer_scale(Q,"Q"), infer_scale(PV,"PV")

    cases = ["13","123"] if args.case == "both" else [args.case]
    for c in cases:
        env, inj, ckpt_dir = case_cfg(c)

        # load hparams if present
        hp_json = os.path.join(ckpt_dir, "best_hparams.json")
        ctx, hid, dq = 16, 64, 0.30
        if os.path.exists(hp_json):
            with open(hp_json,"r") as f:
                hp = json.load(f)
            ctx = int(hp.get("ctx_dim", ctx)); hid = int(hp.get("hidden", hid)); dq = float(hp.get("dq_max", dq))

        pols, mixrs = load_models(len(inj), ckpt_dir, ctx=ctx, hid=hid, dq_max=dq)

        # roll through DOE with the projector
        rows, state = [], env.reset(seed=0)
        for t in range(len(P)):
            u = actor_delta_q(state, env, inj, pols, mixrs, dq_max=dq)            # ΔQ from policy
            v_next, q_applied = project_to_band_fast(
                env, u, load_p=float(P[t])*lp_s, load_q=float(Q[t])*lq_s, pv_p=float(PV[t])*pv_s, inj=inj,
                diag_step=0.25, max_bsearch_steps=4
            )
            rows.append({
                "t": t,
                **{f"v_bus_{b}": float(v_next[i]) for i,b in enumerate(inj)},
                **{f"q_bus_{b}": float(q_applied[i]) for i,b in enumerate(inj)},
            })
            state = v_next   # next step sees new voltages

        df = pd.DataFrame(rows)
        csv_path = os.path.join(OUT_DIR, f"{args.save_prefix}_{c}.csv")
        fig_path = os.path.join(OUT_DIR, f"{args.save_prefix}_{c}.png")
        df.to_csv(csv_path, index=False)

        # quick plot
        fig, axs = plt.subplots(1, 2, figsize=(14,5))
        for k,b in enumerate(inj):
            axs[0].plot(df["t"], df[f"v_bus_{b}"], label=f"Bus {b}")
            axs[1].plot(df["t"], df[f"q_bus_{b}"], label=f"Bus {b}")
        axs[0].hlines([VMIN, VMAX], df["t"].iloc[0], df["t"].iloc[-1], colors="gray", linestyles="--", alpha=0.6)
        axs[0].set_ylabel("Voltage [p.u.]"); axs[1].set_ylabel("Reactive Injection [MVar]")
        axs[0].set_xlabel("Time step");      axs[1].set_xlabel("Time step")
        axs[0].legend(ncol=2); axs[1].legend(ncol=2)
        axs[0].grid(True, alpha=0.3); axs[1].grid(True, alpha=0.3)
        plt.tight_layout(); fig.savefig(fig_path, dpi=160); plt.close(fig)
        print(f"[{c}] saved {csv_path} and {fig_path}")

if __name__ == "__main__":
    main()
