# tasrl_flow.py
# Analytic ∇F(q) and the safe-gradient-flow step used in TASRL.
# References: Eq. (4) (∇F) and Alg. 1 / Prop. 1 (projected flow).
# ∇F(q) = Cq q + v - v_nom   where Cq = diag(eta_i / sbar_i)

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Callable

Array = np.ndarray

def grad_F(q: Array, v: Array, eta_over_sbar: Array, v_nom: float = 1.0) -> Array:
    """
    Compute ∇F(q) = Cq*q + v - v_nom, elementwise.
    q, v: shape (n,)
    eta_over_sbar: shape (n,), entries η_i / s̄_i
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    c = np.asarray(eta_over_sbar, dtype=float).reshape(-1)
    assert q.shape == v.shape == c.shape
    return c * q + (v - v_nom)

def safe_flow_increment(pi_minus_grad: Array, q: Array, q_lo: Array, q_hi: Array, alpha: float) -> Array:
    """
    Project each component of (π(v) - ∇F(q)) onto the interval
    [ -α (q_i - q_lo_i),  α (q_hi_i - q_i) ]  (Alg. 1 lines 413-418).
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    lo = np.asarray(q_lo, dtype=float).reshape(-1)
    hi = np.asarray(q_hi, dtype=float).reshape(-1)
    x  = np.asarray(pi_minus_grad, dtype=float).reshape(-1)
    assert q.shape == lo.shape == hi.shape == x.shape
    lower = -alpha * (q - lo)
    upper =  alpha * (hi - q)
    return np.minimum(np.maximum(x, lower), upper)

def tasrl_step(q: Array,
               v: Array,
               pi_v: Array,
               eta_over_sbar: Array,
               q_lo: Array,
               q_hi: Array,
               alpha: float = 5.0,
               h: float = 1.0,
               v_nom: float = 1.0) -> Array:
    """
    One discrete safe-gradient-flow step:
      q_{k+1} = q_k + h * Proj_{[-α(q-q_lo), α(q_hi-q)]} ( π(v) - ∇F(q) ).
    """
    g = grad_F(q, v, eta_over_sbar, v_nom=v_nom)
    inc = safe_flow_increment(pi_v - g, q, q_lo, q_hi, alpha)
    return np.clip(q + h * inc, q_lo, q_hi)

class TASRLWrapper:
    """
    Wrap your learned actor to apply TASRL safe gradient flow at execution time.
    Expects a callable actor π(v) -> q_suggested with shape (n,).
    """
    def __init__(self,
                 actor_fn: Callable[[Array], Array],
                 q_lo: Array,
                 q_hi: Array,
                 eta_over_sbar: Array,
                 alpha: float = 5.0,
                 h: float = 1.0,
                 v_nom: float = 1.0):
        self.actor_fn = actor_fn
        self.q_lo = np.asarray(q_lo, dtype=float).reshape(-1)
        self.q_hi = np.asarray(q_hi, dtype=float).reshape(-1)
        self.eta_over_sbar = np.asarray(eta_over_sbar, dtype=float).reshape(-1)
        self.alpha = float(alpha)
        self.h = float(h)
        self.v_nom = float(v_nom)
        self._q = np.zeros_like(self.q_lo)

    def reset(self, q0: Optional[Array] = None):
        self._q = np.zeros_like(self._q) if q0 is None else np.asarray(q0, dtype=float).reshape(-1)

    def act(self, v: Array) -> Array:
        v = np.asarray(v, dtype=float).reshape(-1)
        pi_v = np.asarray(self.actor_fn(v), dtype=float).reshape(-1)
        self._q = tasrl_step(self._q, v, pi_v, self.eta_over_sbar,
                             self.q_lo, self.q_hi, alpha=self.alpha, h=self.h, v_nom=self.v_nom)
        return self._q.copy()
