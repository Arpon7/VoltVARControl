# stability_layer.py
# Enforces the discrete-time stability inequality (sufficient condition) used in the base paper:
#   -(2/ΔT) X^{-1}  ≺  J  ≺  0,  where J = ∂u/∂v.
#
# For neighbor-aware policies, J is not diagonal. This layer provides a conservative global scaling
# u_scaled = s * u such that the matrix inequality holds (with respect to an estimated X).
#
# We enforce the left inequality in the "congruence transformed" form:
#   X^{1/2} J X^{1/2}  ≻  -(2/ΔT) I
# which is equivalent for SPD X.
#
# If J is symmetric negative semidefinite (as for negative-gradient-of-convex-potential policies),
# scaling by s ∈ (0,1] preserves symmetry and the right inequality J ≼ 0.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class StabilitySpec:
    dt_s: float = 1.0
    margin: float = 1e-6         # numerical safety margin
    jitter: float = 1e-6         # for SPD fix of X
    min_scale: float = 1e-3      # do not collapse control completely


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _spd_fix(X: np.ndarray, jitter: float) -> np.ndarray:
    Xs = _sym(np.asarray(X, dtype=float))
    # Add jitter until Cholesky works
    lam_min = np.min(np.linalg.eigvalsh(Xs))
    if lam_min <= 0:
        Xs = Xs + (abs(lam_min) + jitter) * np.eye(Xs.shape[0])
    else:
        Xs = Xs + jitter * np.eye(Xs.shape[0])
    return Xs


def _sqrtm_spd(X: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(X)
    vals = np.clip(vals, 1e-12, np.inf)
    return (vecs * np.sqrt(vals)) @ vecs.T


class StabilityScaler:
    """
    Compute a conservative global scaling factor s for a given Jacobian J.

    Usage:
      scaler = StabilityScaler(X_ctrl, StabilitySpec(dt_s=1.0))
      s, M_eigs = scaler.scale_for_J(J)
    """
    def __init__(self, X_ctrl: np.ndarray, spec: StabilitySpec = StabilitySpec()):
        self.spec = spec
        X = _spd_fix(X_ctrl, spec.jitter)
        self.X = X
        self.X_sqrt = _sqrtm_spd(X)

    def scale_for_J(self, J: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Return (s, eigs_M) where:
          M = X^{1/2} J X^{1/2}.
        Condition: min_eig(M) > -(2/dt) + margin.
        """
        dt = float(self.spec.dt_s)
        margin = float(self.spec.margin)
        min_scale = float(self.spec.min_scale)

        J = _sym(np.asarray(J, dtype=float))
        M = self.X_sqrt @ J @ self.X_sqrt
        eigs = np.linalg.eigvalsh(_sym(M))

        lower = -(2.0 / dt) + margin
        min_e = float(np.min(eigs))

        if min_e >= lower:
            return 1.0, eigs

        # Need to scale up towards 0: M_scaled = s*M => min_e_scaled = s*min_e
        # Require s*min_e >= lower => s <= lower/min_e (both negative).
        if min_e >= 0:
            # Should not happen for stable controllers; fall back to minimal scaling
            return min_scale, eigs

        s = lower / min_e  # both negative => positive
        s = float(np.clip(s, min_scale, 1.0))
        return s, eigs
