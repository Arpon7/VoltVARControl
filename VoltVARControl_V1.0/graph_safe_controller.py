# graph_safe_controller.py
# End-to-end neighbor-aware controller with stability scaling.
#
# Components:
#   • GraphPotentialPolicy: produces Δq proposals from control-bus voltages (neighbor-aware)
#   • StabilityScaler: computes a conservative global scaling to satisfy the paper-style Jacobian bound
#
# NOTE: This controller does not require RL training to be useful. You can optionally tune its parameters.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from paper_metrics import estimate_X_from_env
from graph_potential_policy import GraphPotentialPolicy, GraphPolicySpec
from stability_layer import StabilityScaler, StabilitySpec


def build_ctrl_graph_edges(env) -> List[Tuple[int, int]]:
    """
    Build an undirected edge list over control buses using the env's 1-hop physical adjacency.
    Only edges between controllable buses are included.

    Supports env.neighbors {bus_id: [neighbors]} or env.neighbor_map / env.bus_neighbors.
    """
    ctrl_buses = list(getattr(env, "ctrl_buses", getattr(env, "injection_bus", [])))
    bus_to_local = {int(b): i for i, b in enumerate(ctrl_buses)}
    edges = set()

    neigh_dict = None
    for cand in ["neighbors", "neighbor_map", "bus_neighbors", "adj"]:
        if hasattr(env, cand):
            neigh_dict = getattr(env, cand)
            break
    if neigh_dict is None:
        # fall back: no edges
        return []

    # neigh_dict may be keyed by bus id or local index; assume bus id
    for b in ctrl_buses:
        nb = neigh_dict.get(int(b), []) if isinstance(neigh_dict, dict) else []
        for n in nb:
            if int(n) in bus_to_local:
                i = bus_to_local[int(b)]
                j = bus_to_local[int(n)]
                if i != j:
                    a, c = (i, j) if i < j else (j, i)
                    edges.add((a, c))

    return sorted(edges)


@dataclass
class GraphSafeControllerConfig:
    dt_s: float = 1.0
    vmin: float = 0.95
    vmax: float = 1.05
    beta: float = 2.0          # neighbor coupling strength (in u, before dt)
    init_local_gain: float = 4.0
    init_edge_weight: float = 1.0
    stability_margin: float = 1e-6


class GraphSafeController:
    """
    Compute Δq actions from measured control-bus voltages with a conservative stability scaling.

    Call pattern:
      ctrl = GraphSafeController.from_env(env, cfg)
      ctrl.reset(q0=env.get_ctrl_q())
      delta_q = ctrl.act(v_ctrl)   # Δq proposal for env.step_Preward / env.step_profile
    """
    def __init__(self,
                 policy: GraphPotentialPolicy,
                 scaler: StabilityScaler,
                 cfg: GraphSafeControllerConfig,
                 device: str = "cpu"):
        self.policy = policy.to(device)
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self._q = None  # optional internal tracker (not required)

        # Pre-compute a conservative *global* scaling using worst-case mask (all outside band),
        # evaluated at an arbitrary out-of-band point. This avoids per-step eigen computations.
        self.s_global = 1.0
        self._update_global_scale()

    @classmethod
    def from_env(cls, env, cfg: GraphSafeControllerConfig = GraphSafeControllerConfig(), device: str = "cpu"):
        edges = build_ctrl_graph_edges(env)
        n = len(getattr(env, "ctrl_buses", getattr(env, "injection_bus", [])))
        spec = GraphPolicySpec(vmin=cfg.vmin, vmax=cfg.vmax, dt_s=cfg.dt_s, beta=cfg.beta,
                               init_local_gain=cfg.init_local_gain, init_edge_weight=cfg.init_edge_weight)
        policy = GraphPotentialPolicy(n_ctrl=n, edges_undirected=edges, spec=spec)

        X = estimate_X_from_env(env, getattr(env, "ctrl_buses", None))
        scaler = StabilityScaler(X_ctrl=X, spec=StabilitySpec(dt_s=cfg.dt_s, margin=cfg.stability_margin))
        return cls(policy=policy, scaler=scaler, cfg=cfg, device=device)

    def _update_global_scale(self):
        # Worst-case J occurs when all nodes are out of band (local slopes active)
        n = self.policy.n
        v_worst = np.full((n,), self.cfg.vmax + 0.1, dtype=float)
        J = self.policy.jacobian_analytic(v_worst)
        s, _ = self.scaler.scale_for_J(J)
        self.s_global = float(s)

    def reset(self, q0: Optional[np.ndarray] = None):
        self._q = None if q0 is None else np.asarray(q0, dtype=float).reshape(-1)

    def act(self, v_ctrl: np.ndarray) -> np.ndarray:
        v = np.asarray(v_ctrl, dtype=float).reshape(1, -1)
        with torch.no_grad():
            delta_q_t, diag = self.policy(torch.tensor(v, dtype=torch.float32, device=self.device))
            delta_q = delta_q_t.detach().cpu().numpy().reshape(-1)

        # Global stability scaling
        delta_q = self.s_global * delta_q

        return delta_q

    def diagnostics(self, v_ctrl: np.ndarray) -> Dict[str, float]:
        """Quick numeric diagnostics for logging/debugging."""
        v = np.asarray(v_ctrl, dtype=float).reshape(-1)
        J = self.policy.jacobian_analytic(v)
        s, eigs = self.scaler.scale_for_J(J)
        return {
            "s_step": float(s),
            "s_global": float(self.s_global),
            "min_eig_M": float(np.min(eigs)),
            "max_eig_M": float(np.max(eigs)),
        }
