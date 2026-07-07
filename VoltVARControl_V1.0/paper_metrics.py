# paper_metrics.py  — env-agnostic helpers for paper-style metrics
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandapower as pp

# --------- env introspection helpers ---------
def _get_pp_net(env):
    """
    Return the underlying pandapower net from the env.
    Supports Case 1 and Case 2 naming: env.pp_net / env.net / env.network
    """
    if hasattr(env, "pp_net"):   return env.pp_net
    if hasattr(env, "net"):      return env.net
    if hasattr(env, "network"):  return env.network
    raise RuntimeError("No net/pp_net/network in env.")

def _get_ctrl_buses(env, ctrl_buses=None) -> Sequence[int]:
    """
    Return the list of controlled bus indices (0-based).
    Supports Case 1 and Case 2 naming: env.ctrl_buses / env.injection_bus
    """
    if ctrl_buses is not None:
        return list(ctrl_buses)
    if hasattr(env, "ctrl_buses"):    return list(env.ctrl_buses)
    if hasattr(env, "injection_bus"): return list(env.injection_bus)
    raise RuntimeError("No ctrl_buses/injection_bus in env.")

# --------- voltage reader used by estimators ---------
def read_v_ctrl(env, ctrl_buses=None) -> np.ndarray:
    """
    Read per-unit voltages at the controller buses from env/net.
    """
    net = _get_pp_net(env)
    buses = _get_ctrl_buses(env, ctrl_buses)
    # Ensure a power flow has been solved prior to reading results
    if not hasattr(net, "res_bus") or "vm_pu" not in getattr(net, "res_bus", {}):
        try:
            if hasattr(env, "_runpp"):
                env._runpp(init_dc=False)
            else:
                pp.runpp(net, algorithm="bfsw", init="dc", enforce_q_lims=True,
                         calculate_voltage_angles=False, numba=False)
        except Exception:
            pass
    return net.res_bus.iloc[buses].vm_pu.to_numpy().astype(float)

# --------- finite-difference sensitivity (paper) ---------
def estimate_X_from_env(env, ctrl_buses=None, eps: float = 1e-3) -> np.ndarray:
    """
    Estimate the sensitivity matrix X = dV/dQ at the control buses via finite differences.

    Steps:
      1) Ensure each control bus has an sgen row (create if missing) whose q_mvar we can perturb.
      2) Compute baseline voltages v0 at control buses.
      3) For each control bus j, add +eps to its sgen.q_mvar, solve PF, read v1, and set X[:, j] = (v1 - v0)/eps.
      4) Restore original q_mvar and PF.

    Notes:
      * Uses env._runpp(...) when available (Case 1); otherwise calls pandapower.runpp(...) directly.
      * ctrl_buses may be provided; if None, we infer from env.ctrl_buses or env.injection_bus.
    """
    net   = _get_pp_net(env)
    buses = _get_ctrl_buses(env, ctrl_buses)
    k = len(buses)
    if k == 0:
        raise ValueError("No control buses provided or found in env.")

    # 0) solve PF to get a consistent baseline
    if hasattr(env, "_runpp"):
        env._runpp(init_dc=False)
    else:
        pp.runpp(net, algorithm="bfsw", init="dc", enforce_q_lims=True,
                 calculate_voltage_angles=False, numba=False)

    # 1) ensure sgen rows exist at each control bus
    bus_to_sgen = {}
    if hasattr(net, "sgen") and len(net.sgen):
        for i, b in enumerate(net.sgen.bus.values):
            b = int(b)
            if b not in bus_to_sgen:
                bus_to_sgen[b] = int(i)
    for b in buses:
        b = int(b)
        if b not in bus_to_sgen:
            idx = pp.create_sgen(net, b, p_mw=0.0, q_mvar=0.0)
            bus_to_sgen[b] = int(idx)

    # 2) baseline voltages
    v0 = read_v_ctrl(env, buses)

    # 3) per-bus perturbation
    X = np.zeros((k, k), dtype=float)
    for j, b in enumerate(buses):
        b   = int(b)
        idx = bus_to_sgen[b]
        q_old = float(net.sgen.at[idx, "q_mvar"])
        net.sgen.at[idx, "q_mvar"] = q_old + eps

        # solve PF
        if hasattr(env, "_runpp"):
            env._runpp(init_dc=False)
        else:
            pp.runpp(net, algorithm="bfsw", init="results", enforce_q_lims=True,
                     calculate_voltage_angles=False, numba=False)

        v1 = read_v_ctrl(env, buses)
        X[:, j] = (v1 - v0) / eps

        # restore q
        net.sgen.at[idx, "q_mvar"] = q_old

    # 4) restore PF
    if hasattr(env, "_runpp"):
        env._runpp(init_dc=False)
    else:
        pp.runpp(net, algorithm="bfsw", init="results", enforce_q_lims=True,
                 calculate_voltage_angles=False, numba=False)

    return X
