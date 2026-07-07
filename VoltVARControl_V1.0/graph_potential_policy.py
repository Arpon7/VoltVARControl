# graph_potential_policy.py
# Neighbor-aware, stability-oriented voltage controller.
#
# Design:
#   u(v) = -∇_v Φ(v), where Φ is a convex graph-structured potential over control-bus voltages.
# This yields a symmetric negative-semidefinite Jacobian J = ∂u/∂v = -∇^2 Φ.
#
# We implement:
#   Φ(v) = Σ_i φ_i(v_i) + (β/2) Σ_(i,j) w_ij (v_i - v_j)^2
# where:
#   • φ_i provides a DEAD-BAND: u_i = 0 when v_i ∈ [vmin, vmax]
#   • w_ij ≥ 0 are learnable edge weights (softplus)
#   • β ≥ 0 is a (global) coupling gain
#
# The resulting controller:
#   u_i = u_local_i + u_nb_i
#   u_local_i = -k_i * (v_i - vmax) for v_i > vmax
#               -k_i * (v_i - vmin) for v_i < vmin
#               0 otherwise
#   u_nb_i    = -β Σ_{j∈N(i)} w_ij (v_i - v_j)
#
# Output convention: the module returns Δq = dt * u  (increment in reactive power).
#
# NOTE: This is a control policy module; training/tuning of parameters is optional.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GraphPolicySpec:
    vmin: float = 0.95
    vmax: float = 1.05
    dt_s: float = 1.0
    beta: float = 1.0
    init_local_gain: float = 3.0   # k_i (pre-softplus)
    init_edge_weight: float = 1.0  # w_ij (pre-softplus)


def _softplus_inv(y: float) -> float:
    """Inverse of softplus for scalar y>0 (approx)."""
    # softplus(x) = log(1+exp(x)) ~ y => x ~ log(exp(y)-1)
    y = float(y)
    if y <= 1e-12:
        return -30.0
    return float(np.log(np.expm1(y)))


class GraphPotentialPolicy(nn.Module):
    """
    Neighbor-aware controller on the control-bus graph.

    Inputs:
      v_ctrl: Tensor [B, N] control bus voltages (pu)

    Outputs:
      delta_q: Tensor [B, N] reactive power increments (MVar) for ΔQ semantics
      diag: dict with useful diagnostics
    """
    def __init__(self,
                 n_ctrl: int,
                 edges_undirected: Sequence[Tuple[int, int]],
                 spec: GraphPolicySpec = GraphPolicySpec()):
        super().__init__()
        self.n = int(n_ctrl)
        self.edges = [(int(i), int(j)) for i, j in edges_undirected if int(i) != int(j)]
        self.spec = spec

        # Local gains k_i >= 0 (softplus)
        init_k = _softplus_inv(max(1e-6, spec.init_local_gain))
        self._raw_k = nn.Parameter(torch.full((self.n,), init_k, dtype=torch.float32))

        # Edge weights w_e >= 0 (softplus), one per undirected edge
        init_w = _softplus_inv(max(1e-6, spec.init_edge_weight))
        self._raw_w = nn.Parameter(torch.full((len(self.edges),), init_w, dtype=torch.float32))

    @property
    def k(self) -> torch.Tensor:
        return F.softplus(self._raw_k)  # [N]

    @property
    def w(self) -> torch.Tensor:
        return F.softplus(self._raw_w)  # [E]

    def forward(self, v_ctrl: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v = v_ctrl
        if v.ndim != 2 or v.shape[1] != self.n:
            raise ValueError(f"v_ctrl must have shape [B,{self.n}], got {tuple(v.shape)}")

        vmin = float(self.spec.vmin)
        vmax = float(self.spec.vmax)
        dt = float(self.spec.dt_s)
        beta = float(self.spec.beta)

        # --- local term with exact deadband ---
        k = self.k  # [N]
        v_hi = torch.clamp(v - vmax, min=0.0)         # >0 if over-voltage
        v_lo = torch.clamp(vmin - v, min=0.0)         # >0 if under-voltage
        # local "outside" deviation with sign:
        dev = v_hi - v_lo                              # positive if over-voltage, negative if under-voltage
        u_local = -(dev * k.unsqueeze(0))               # [B,N]; 0 inside band

        # --- neighbor coupling term ---
        u_nb = torch.zeros_like(v)
        if len(self.edges) > 0 and beta > 0.0:
            w = self.w  # [E]
            for e, (i, j) in enumerate(self.edges):
                wij = w[e]
                diff = v[:, i] - v[:, j]              # [B]
                # u_i += -beta*wij*(v_i - v_j), u_j += -beta*wij*(v_j - v_i)
                u_nb[:, i] += -beta * wij * diff
                u_nb[:, j] += +beta * wij * diff

        # --- optional "do no harm" gating ---
        # If ALL control buses are inside the band, take no action (exact invariance).
        in_band_all = ((v >= vmin) & (v <= vmax)).all(dim=1, keepdim=True)  # [B,1]
        u = torch.where(in_band_all, torch.zeros_like(v), u_local + u_nb)

        delta_q = dt * u

        diag = {
            "u_local": u_local.detach(),
            "u_nb": u_nb.detach(),
            "u": u.detach(),
            "delta_q": delta_q.detach(),
            "k": k.detach(),
            "w": self.w.detach(),
        }
        return delta_q, diag

    def jacobian_analytic(self, v_ctrl: np.ndarray) -> np.ndarray:
        """
        Analytic Jacobian J = ∂u/∂v for a single state v_ctrl (shape [N]).
        We use the piecewise-linear local term (deadband) and quadratic neighbor term.

        Returns:
          J: [N,N] numpy array (for u, not delta_q). For delta_q Jacobian, multiply by dt.
        """
        v = np.asarray(v_ctrl, dtype=float).reshape(-1)
        if v.size != self.n:
            raise ValueError(f"v_ctrl must have size {self.n}, got {v.size}")

        vmin = float(self.spec.vmin)
        vmax = float(self.spec.vmax)
        beta = float(self.spec.beta)

        # local slope mask: outside band => du_i/dv_i = -k_i; inside band => 0
        k = self.k.detach().cpu().numpy().reshape(-1)
        mask_out = (v < vmin) | (v > vmax)
        J = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            if mask_out[i]:
                J[i, i] += -k[i]

        # neighbor term => -beta * Laplacian(W)
        if len(self.edges) > 0 and beta > 0.0:
            w = self.w.detach().cpu().numpy().reshape(-1)
            for e, (i, j) in enumerate(self.edges):
                wij = float(w[e]) * beta
                # contributions: u_i = -wij*(v_i - v_j) => du_i/dv_i -= wij, du_i/dv_j += wij
                J[i, i] += -wij
                J[j, j] += -wij
                J[i, j] += +wij
                J[j, i] += +wij

        # If all in band, u is forced to 0 => Jacobian treated as 0 for invariance.
        if mask_out.sum() == 0:
            return np.zeros_like(J)

        return J
