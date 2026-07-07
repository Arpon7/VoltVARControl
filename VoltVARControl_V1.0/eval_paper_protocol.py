# eval_paper_protocol.py
# Paper-style evaluation using YOUR trained checkpoints + neighbor context,
# TASRL projected gradient flow, and paper-weighted ΔF. Supports:
#   --ctrl           : explicit controller buses
#   --paper_strict   : single, non-escalating disturbance per trial (match paper)
#   --out            : write JSON to file (still prints to stdout)
from __future__ import annotations
import argparse, json, os, random, re, glob
from typing import Sequence, Tuple

import numpy as np
import torch

from graph_safe_controller import GraphSafeController, GraphSafeControllerConfig

# Envs
from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from env_single_phase_14bus import IEEE14bus, create_14bus

# Policy (your actor + mixer)
from hybrid_safe_ddpg import AttentionMixer, SafePolicyNetworkWithCoord

# TASRL safe gradient flow
from tasrl_flow import TASRLWrapper

# Sensitivity estimator (we'll compute F inline using X_inv)
from paper_metrics import estimate_X_from_env


# ----------------- utils -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def infer_ctrl_buses_from_net(pp_net):
    # Prefer sgen buses
    if hasattr(pp_net, "sgen") and len(pp_net.sgen):
        return sorted(pp_net.sgen["bus"].astype(int).unique().tolist())
    # Fallback: all buses (not ideal, but safe)
    return sorted(pp_net.bus.index.astype(int).tolist())


def build_env(case: int, ctrl_str: str = ""):
    """Build env and set controller buses (from --ctrl if provided, else from net)."""
    def parse_ctrl(default_list):
        if ctrl_str.strip():
            return [int(x) for x in ctrl_str.split(",") if x.strip()]
        return default_list

    if case == 13:
        pp_net = create_13bus()
        inj_default = infer_ctrl_buses_from_net(pp_net) or [2, 7, 9]
        inj = parse_ctrl(inj_default)
        env = IEEE13bus(pp_net, injection_bus=inj)
    elif case == 123:
        pp_net = create_123bus()
        inj_default = infer_ctrl_buses_from_net(pp_net) or [9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]
        inj = parse_ctrl(inj_default)
        env = IEEE123bus(pp_net, injection_bus=inj)
    elif case == 14:
        pp_net = create_14bus()
        inj_default = [2, 3, 8, 13]
        inj = parse_ctrl(inj_default)
        env = IEEE14bus(pp_net, injection_bus=inj)
    else:
        raise ValueError("case must be 13, 14 or 123")

    env.ctrl_buses = list(env.injection_bus)
    # warm-up run (no action) just to populate caches if env supports it
    try:
        _ = env.step_Preward(np.zeros(len(env.ctrl_buses), dtype=float))
    except Exception:
        pass
    return env


# ---- helper to read controller-bus voltages for Case 1/Case 2 envs ----
def _get_ctrl_voltages(env) -> np.ndarray:
    """
    Unified voltage reader:
    - If env has get_ctrl_bus_voltages(), use it.
    - Else read from env.network/pp_net/net.res_bus at env.ctrl_buses/injection_bus.
    """
    if hasattr(env, "get_ctrl_bus_voltages"):
        return np.asarray(env.get_ctrl_bus_voltages(), dtype=float).reshape(-1)

    net = getattr(env, "network", getattr(env, "pp_net", getattr(env, "net", None)))
    buses = getattr(env, "ctrl_buses", getattr(env, "injection_bus", None))
    if net is None or buses is None:
        raise RuntimeError("Cannot find net/buses to read voltages.")
    return net.res_bus.iloc[buses].vm_pu.to_numpy().astype(float).reshape(-1)


# ----------------- trained-weights actor (like test.py) -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _ckpt_dir_for_case(case: int) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    sub = "13bus" if case == 13 else ("123bus" if case == 123 else "14bus")
    return os.path.join(here, "checkpoints", "single-phase", sub, "safe-ddpg")

def _load_hparams(hp_json_path: str, defaults=(16, 64, 0.30)):
    ctx, hid, dq = defaults
    if os.path.exists(hp_json_path):
        with open(hp_json_path, "r") as f:
            hp = json.load(f)
        ctx = int(hp.get("ctx_dim", ctx))
        hid = int(hp.get("hidden", hid))
        dq  = float(hp.get("dq_max", dq))
    return ctx, hid, dq

def _neighbors_tensor(env, V_1xN: torch.Tensor, agent_idx: int, inj):
    bus_id = int(inj[agent_idx])
    nb = env.neighs.get(bus_id, []) if hasattr(env, "neighs") else []
    if not nb:
        return torch.zeros((1, 1, 1), dtype=torch.float32, device=V_1xN.device)
    idxs = [env.bus_to_idx[b] for b in nb if b in env.bus_to_idx]
    if len(idxs) == 0:
        return torch.zeros((1, 1, 1), dtype=torch.float32, device=V_1xN.device)
    return V_1xN[:, idxs].unsqueeze(-1)

def _find_ckpt(ckpt_dir, kind, i):
    fname = f"policy_agent{i}.pth" if kind == "policy" else f"mixer_agent{i}.pth"
    path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {kind} checkpoint {path}")
    return path

def _list_checkpoint_indices(ckpt_dir: str, kind: str) -> list[int]:
    pat = os.path.join(ckpt_dir, f"{'policy' if kind=='policy' else 'mixer'}_agent*.pth")
    files = glob.glob(pat)
    idxs = []
    for f in files:
        m = re.search(r"agent(\d+)\.pth$", f.replace("\\","/"))
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)

def build_actor(n_ctrl: int, dq_max: float, device: str, env=None):
    """Use YOUR trained per-agent policies + mixers with neighbor context. Auto-trims to available checkpoints."""
    assert env is not None, "build_actor needs env to compute neighbors"
    case = 13 if len(getattr(env, "ctrl_buses", [])) <= 20 else (123 if len(getattr(env, "ctrl_buses", [])) > 20 else 14)
    ckpt_dir = _ckpt_dir_for_case(case)
    hp_json = os.path.join(ckpt_dir, "best_hparams.json")
    ctx_dim, hidden_dim, dq_loaded = _load_hparams(hp_json, defaults=(16, 64, dq_max))
    dq_use = float(dq_loaded if np.isfinite(dq_loaded) else dq_max)

    policy_idxs = _list_checkpoint_indices(ckpt_dir, "policy")
    mixer_idxs  = _list_checkpoint_indices(ckpt_dir, "mixer")
    available = sorted(set(policy_idxs) & set(mixer_idxs))
    if not available:
        raise FileNotFoundError(f"No matching policy/mixer checkpoints in {ckpt_dir}")
    n_avail = 1 + max(available)  # assuming contiguous 0..N-1
    n_use = min(n_ctrl, n_avail)

    inj_all = list(getattr(env, "injection_bus", getattr(env, "ctrl_buses", range(n_ctrl))))
    inj = inj_all[:n_use]
    n_ctrl = n_use

    env.ctrl_buses = inj
    if not hasattr(env, "bus_to_idx"):
        env.bus_to_idx = {int(b): i for i, b in enumerate(inj)}
    if not hasattr(env, "neighs"):
        env.neighs = {int(b): [] for b in inj}

    policies = [SafePolicyNetworkWithCoord(1, ctx_dim, hidden_dim, 0.90, 1.10, dq_use).to(DEVICE)
                for _ in range(n_ctrl)]
    mixers   = [AttentionMixer(in_dim=1, hid_dim=ctx_dim).to(DEVICE)
                for _ in range(n_ctrl)]
    for i in range(n_ctrl):
        policies[i].load_state_dict(torch.load(_find_ckpt(ckpt_dir, "policy", i), map_location=DEVICE), strict=True)
        mixers[i].load_state_dict(torch.load(_find_ckpt(ckpt_dir, "mixer",  i), map_location=DEVICE), strict=True)
        policies[i].eval(); mixers[i].eval()

    @torch.no_grad()
    def pi_fn(v_vec_np: np.ndarray) -> np.ndarray:
        v_arr = np.asarray(v_vec_np, dtype=np.float32).reshape(1, -1)
        V = torch.tensor(v_arr, dtype=torch.float32, device=DEVICE)
        outs = []
        for i in range(n_ctrl):
            v_i  = V[:, i:i+1]
            v_nb = _neighbors_tensor(env, V, i, inj)
            m_i  = mixers[i](v_i, v_nb)
            u_i  = policies[i](v_i, m_i)         # raw
            outs.append(u_i)
        U = torch.cat(outs, dim=1).squeeze(0)
        U = torch.clamp(U, -dq_use, dq_use)
        return U.detach().cpu().numpy().reshape(-1)

    return pi_fn


# ---- paper-style steady-state objective using X^{-1} (computed inline) ----
def paper_F(q: np.ndarray, v: np.ndarray, eta_over_sbar: np.ndarray, X_inv: np.ndarray) -> float:
    """
    F(q, v) = 0.5 * (v-1)^T X^{-1} (v-1) + 0.5 * sum_i (eta_i/sbar_i) * q_i^2
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    e = (v - 1.0).reshape(-1, 1)
    quad = (e.T @ X_inv @ e)                        # shape (1,1)
    term_v = 0.5 * quad.item()
    term_q = 0.5 * float(np.sum(eta_over_sbar.reshape(-1) * (q ** 2)))
    return term_v + term_q


# ---- STRICT disturbance helpers (non-escalating, single kick) ----
def _snapshot_PQPV(env):
    """Return copies of p_mw/q_mvar for load and sgen so we can restore after a trial."""
    net = getattr(env, "network", getattr(env, "pp_net", getattr(env, "net", None)))
    snap = {}
    if hasattr(net, "load") and len(net.load):
        snap["load_p"] = net.load["p_mw"].to_numpy().copy()
        if "q_mvar" in net.load: snap["load_q"] = net.load["q_mvar"].to_numpy().copy()
    if hasattr(net, "sgen") and len(net.sgen):
        snap["sgen_p"] = net.sgen["p_mw"].to_numpy().copy()
        if "q_mvar" in net.sgen: snap["sgen_q"] = net.sgen["q_mvar"].to_numpy().copy()
    return snap

def _restore_PQPV(env, snap):
    net = getattr(env, "network", getattr(env, "pp_net", getattr(env, "net", None)))
    if "load_p" in snap: net.load.loc[:, "p_mw"] = snap["load_p"]
    if "load_q" in snap: net.load.loc[:, "q_mvar"] = snap["load_q"]
    if "sgen_p" in snap: net.sgen.loc[:, "p_mw"] = snap["sgen_p"]
    if "sgen_q" in snap: net.sgen.loc[:, "q_mvar"] = snap["sgen_q"]
    if hasattr(env, "_runpp"): env._runpp(init_dc=False)

def _apply_single_kick(env, rng: np.random.Generator,
                       v_band=(0.95, 1.05),
                       load_scale_range=(0.04, 0.07),
                       pv_scale_range=(0.00, 0.05),
                       pv_sign=-1.0):
    """
    One-shot moderate perturbation:
      - Scale all loads by (1 + u),     u ~ U[+4%, +7%]
      - Scale all PV  by (1 + v*sign),  v ~ U[ 0%, +5%], sign=-1 => reduce PV
    No escalation / no repeated kicks.
    """
    net = getattr(env, "network", getattr(env, "pp_net", getattr(env, "net", None)))
    if hasattr(net, "load") and len(net.load):
        u = rng.uniform(load_scale_range[0], load_scale_range[1])
        net.load.loc[:, "p_mw"] = net.load["p_mw"] * (1.0 + u)
        if "q_mvar" in net.load:
            net.load.loc[:, "q_mvar"] = net.load["q_mvar"] * (1.0 + u)
    if hasattr(net, "sgen") and len(net.sgen):
        v = rng.uniform(pv_scale_range[0], pv_scale_range[1])
        net.sgen.loc[:, "p_mw"] = net.sgen["p_mw"] * (1.0 + pv_sign * v)
    if hasattr(env, "_runpp"): env._runpp(init_dc=False)
    # Ensure at least one violation; nudge lightly if needed
    v_ctrl = _get_ctrl_voltages(env)
    if np.all((v_ctrl >= v_band[0]) & (v_ctrl <= v_band[1])):
        if hasattr(net, "load") and len(net.load):
            u2 = 0.01  # +1% nudge
            net.load.loc[:, "p_mw"] = net.load["p_mw"] * (1.0 + u2)
            if "q_mvar" in net.load:
                net.load.loc[:, "q_mvar"] = net.load["q_mvar"] * (1.0 + u2)
        if hasattr(env, "_runpp"): env._runpp(init_dc=False)


# ---- Lazy import to avoid circulars ----
def _get_paper_scenarios():
    try:
        from disturbance_gen import paper_scenarios as _ps  # lazy import
        return _ps
    except Exception:
        # Fallback: simple generator that just yields seeds (no env changes)
        def _fallback(env, n_scenarios: int, start_seed: int = 0):
            for i in range(n_scenarios):
                yield None, None, start_seed + i
        return _fallback


# ----------------- core eval -----------------
def run_trial(env,
              controller,
              horizon_s: float,
              dt_s: float,
              v_band: Tuple[float, float],
              X_inv: np.ndarray,
              device: str = "cpu") -> Tuple[float, float, float]:
    """
    Returns (recovery_time_seconds, transient_reward_sum, steady_state_deltaF_true).

    controller:
      • TASRLWrapper -> returns q_setpoints via .act(v_ctrl)
      • GraphSafeController (or compatible) -> returns delta_q via .act(v_ctrl)

    Notes:
      - env.step_Preward expects ΔQ (incremental) actions.
      - We track q internally to compute ΔF with the paper-style objective.
    """
    n = len(env.ctrl_buses)
    t = 0.0
    recovery = None
    reward_sum = 0.0

    # controller/state tracking
    q_prev = np.zeros(n, dtype=float)
    try:
        q_prev = np.asarray(env.get_ctrl_q(), dtype=float).reshape(-1)
    except Exception:
        pass

    last_v = None

    steps = int(np.ceil(horizon_s / dt_s))
    for _ in range(steps):
        v_ctrl = _get_ctrl_voltages(env)  # measured/true; env already includes noise if configured
        last_v = v_ctrl.copy()

        # --- Controller output to ΔQ ---
        if hasattr(controller, "eta_over_sbar") and hasattr(controller, "act"):
            # TASRLWrapper: produces q_set
            q_set = np.asarray(controller.act(v_ctrl), dtype=float).reshape(-1)
            delta_q = q_set - q_prev
            q_prev = q_set
        else:
            # GraphSafeController: produces delta_q directly
            delta_q = np.asarray(controller.act(v_ctrl), dtype=float).reshape(-1)
            q_prev = q_prev + delta_q

        # --- step the env (ΔQ semantics) ---
        obs, r, done, info = env.step_Preward(delta_q)
        reward_sum += float(r)
        t += dt_s

        # recovery time uses the latest control-bus voltages (prefer true if present)
        v_for_recovery = np.asarray(info.get("v_true", obs), dtype=float).reshape(-1)
        if recovery is None and np.all((v_for_recovery >= v_band[0]) & (v_for_recovery <= v_band[1])):
            recovery = t

        if done:
            break

    if recovery is None:
        recovery = t

    # --- F* at the controlled steady state (paper-weighted) ---
    F_star = paper_F(q=q_prev, v=last_v, eta_over_sbar=np.full(n, getattr(controller, "eta_over_sbar_scalar", 0.01)),
                     X_inv=X_inv)

    # --- Baseline F0: same loads/PV, q = 0 (re-solve), paper-weighted ---
    F0 = None
    saved_q = []
    try:
        if hasattr(env, "network") and hasattr(env.network, "sgen") and len(env.network.sgen):
            for b in env.injection_bus:
                sidx = env.sgen_idx_for_inj[int(b)]
                q_old = float(env.network.sgen.at[sidx, "q_mvar"])
                saved_q.append((sidx, q_old))
                env.network.sgen.at[sidx, "q_mvar"] = 0.0
            if hasattr(env, "_runpp"): env._runpp(init_dc=False)
            v0 = _get_ctrl_voltages(env)
            F0 = paper_F(q=np.zeros_like(q_prev), v=v0,
                         eta_over_sbar=np.full(n, getattr(controller, "eta_over_sbar_scalar", 0.01)),
                         X_inv=X_inv)
    finally:
        # Restore q and PF for consistency
        if saved_q:
            for sidx, q_old in saved_q:
                env.network.sgen.at[sidx, "q_mvar"] = q_old
            if hasattr(env, "_runpp"): env._runpp(init_dc=False)

    if F0 is None:
        F0 = 0.0

    deltaF_true = float(F_star - F0)
    return float(recovery), float(reward_sum), float(deltaF_true)


def evaluate(case: int,
             trials: int,
             dt: float,
             horizon: float,
             band: Sequence[float],
             dq_max: float,
             alpha: float,
             h: float,
             eta_over_sbar_scalar: float,
             device: str = "cpu",
             seed: int = 2025,
             n_scenarios: int = 500,
             ctrl_str: str = "",
             paper_strict: bool = False,
             controller_name: str = "graph"):
    set_seed(seed)
    env = build_env(case, ctrl_str=ctrl_str)
    n = len(env.ctrl_buses)

    # clamp & order the band
    vmin, vmax = float(min(band)), float(max(band))

    # q bounds and per-bus weight c_i = η_i / s̄_i
    q_lo = -dq_max * np.ones(n)
    q_hi =  dq_max * np.ones(n)
    eta_over_sbar = eta_over_sbar_scalar * np.ones(n)

    # Controller selection:
    #   • graph: neighbor-aware convex-potential controller + conservative stability scaling
    #   • tasrl: TASRL projected flow using YOUR learned actor (legacy path)
    controller_name = str(controller_name).strip().lower()
    controller = None

    if controller_name == "graph":
        cfg = GraphSafeControllerConfig(
            dt_s=float(dt),
            vmin=float(vmin),
            vmax=float(vmax),
            beta=2.0,
            init_local_gain=4.0,
            init_edge_weight=1.0,
            stability_margin=1e-6,
        )
        controller = GraphSafeController.from_env(env, cfg=cfg, device=device)
        # carry scalar for paper_F weighting (kept compatible with TASRL path)
        controller.eta_over_sbar_scalar = float(eta_over_sbar_scalar)
    elif controller_name in ("tasrl", "legacy"):
        # actor and TASRL safe-flow
        pi_fn = build_actor(n, dq_max, device=device, env=env)
        tasrl = TASRLWrapper(actor_fn=pi_fn,
                             q_lo=q_lo, q_hi=q_hi,
                             eta_over_sbar=eta_over_sbar,
                             alpha=alpha, h=h, v_nom=1.0)
        tasrl.eta_over_sbar_scalar = float(eta_over_sbar_scalar)
        controller = tasrl
    else:
        raise ValueError("controller_name must be one of: graph, tasrl")

    # ---- Estimate X and build a robust, normalized inverse (paper-style weighting) ----
    X = estimate_X_from_env(env, getattr(env, "injection_bus", None))
    eps = 1e-8
    X_inv = np.linalg.pinv(X + eps * np.eye(X.shape[0]))
    tr = np.trace(X_inv)
    if tr > 0:
        X_inv = X_inv / tr

    rec_all, trR_all, dF_all = [], [], []

    rng = np.random.default_rng(seed)

    if paper_strict:
        # STRICT: one non-escalating kick per trial
        for _ in range(trials):
            snap = _snapshot_PQPV(env)
            try:
                _apply_single_kick(env, rng, v_band=(vmin, vmax))
                if hasattr(controller, 'reset'): controller.reset(q0=np.zeros(n))
                rtime, treward, dF = run_trial(env, controller,
                                               horizon_s=horizon, dt_s=dt,
                                               v_band=(vmin, vmax),
                                               X_inv=X_inv, device=device)
                rec_all.append(rtime); trR_all.append(treward); dF_all.append(dF)
            finally:
                _restore_PQPV(env, snap)
    else:
        # DEFAULT: lazy-import to avoid circular import
        paper_scenarios = _get_paper_scenarios()
        for i, (_, _, seed_i) in enumerate(paper_scenarios(env, n_scenarios=n_scenarios, start_seed=seed)):
            if hasattr(controller, 'reset'): controller.reset(q0=np.zeros(n))
            rtime, treward, dF = run_trial(env, controller,
                                           horizon_s=horizon, dt_s=dt,
                                           v_band=(vmin, vmax),
                                           X_inv=X_inv, device=device)
            rec_all.append(rtime); trR_all.append(treward); dF_all.append(dF)
            if len(rec_all) >= trials:
                break

    out = {
        "case": case,
        "trials": trials,
        "dt_s": dt,
        "horizon_s": horizon,
        "band": [vmin, vmax],
        "dq_max": dq_max,
        "alpha": alpha,
        "h": h,
        "eta_over_sbar": eta_over_sbar_scalar,
        # Paper-style signs: transient = sum of rewards (often negative)
        "mean_recovery_time_s": float(np.mean(rec_all)) if rec_all else None,
        "mean_transient_reward": float(np.mean(trR_all)) if trR_all else None,
        # True ΔF with paper weighting (more negative = better)
        "mean_deltaF": float(np.mean(dF_all)) if dF_all else None,
    }
    print(json.dumps(out, indent=2))
    return out


# ----------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, choices=[13, 14, 123], default=13, required=True)
    ap.add_argument("--trials", type=int, default=500, help="how many scenarios to score")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--band", type=float, nargs=2, default=[0.95, 1.05])
    ap.add_argument("--dq_max", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--eta_over_sbar", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--ctrl", type=str, default="",
                    help="Comma-separated list of controller bus numbers (e.g. '2,7,9'). "
                         "If empty, infer from the network; will still trim to available checkpoints.")
    ap.add_argument("--controller", type=str, default="graph", choices=["graph","tasrl"],
                    help="Controller to evaluate: graph (neighbor-aware + stability scaling) or tasrl (legacy).")
    ap.add_argument("--paper_strict", action="store_true",
                    help="Use a single, non-escalating disturbance per trial (closer to base paper).")
    ap.add_argument("--out", type=str, default="",
                    help="If set, write the metrics JSON to this path.")
    args = ap.parse_args()

    out = evaluate(case=args.case, trials=args.trials, dt=args.dt, horizon=args.horizon,
                   band=args.band, dq_max=args.dq_max, alpha=args.alpha, h=args.h,
                   eta_over_sbar_scalar=args.eta_over_sbar, device=args.device, seed=args.seed,
                   ctrl_str=args.ctrl, paper_strict=args.paper_strict, controller_name=args.controller)

    # Save to file if requested (still printed to stdout above by evaluate())
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

