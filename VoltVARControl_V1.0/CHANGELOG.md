# Changelog (Your Paper vs. Base Paper Alignment)

## 2025-12-23 — Neighbor-aware control-bus protection + stability scaling (new method)

### Added
- **GraphPotentialPolicy (`graph_potential_policy.py`)**
  - New *neighbor-aware* controller over the control-bus subgraph.
  - Action is **ΔQ** (incremental reactive power) computed as a negative gradient of a convex, graph-structured potential:
    - local deadband term enforces **no action** inside `[vmin, vmax]`
    - quadratic neighbor coupling discourages *harming other control buses* by penalizing voltage disagreement between control buses

- **StabilityScaler (`stability_layer.py`)**
  - Implements the paper-style *sufficient* discrete-time stability bound:
    - `-(2/ΔT) X^{-1} ≺ J ≺ 0`
  - Enforced via a conservative **global scaling** computed using the congruence form:
    - `X^{1/2} J X^{1/2} ≻ -(2/ΔT) I`

- **GraphSafeController (`graph_safe_controller.py`)**
  - Wraps the above into a single controller that outputs **ΔQ** actions for your environments.
  - Builds the control-bus neighbor graph from the feeder adjacency (1-hop physical neighbors, filtered to control buses).
  - Estimates `X = dV/dQ` at the control buses via finite differences (`paper_metrics.estimate_X_from_env`).
  - Computes a conservative stability scaling once (worst-case “all out of band”).

### Changed
- **Environment API consistency (`env_single_phase_13bus.py`, `env_single_phase_123bus.py`)**
  - `step_Preward` now returns: `(obs, reward, done, info)` and expects **ΔQ**.
  - Added `step_profile(delta_q, load_p, load_q, pv_p)` for DOE/evaluation with profiles (ΔQ + load/PV).
  - Added convenience helpers: `ctrl_buses`, `get_ctrl_q`, `set_ctrl_q`, `get_ctrl_voltages`, `get_all_voltages`.
  - `IEEE123bus` now supports `dq_max` and applies both rate limit and absolute `q_limits` consistently.

- **IEEE14 wrapper (`env_single_phase_14bus.py`)**
  - Updated to pass `dq_max` to the base environment.

- **Paper evaluation (`eval_paper_protocol.py`)**
  - Added `--controller {graph,tasrl}`; default is `graph`.
  - Updated trial stepping to respect **ΔQ semantics**.

- **Projector scripts (`IEEE_13_3p.py`)**
  - Updated `step_load` unpacking due to the unified `(obs, reward, done, info)` return.

### Rationale
- Your new requirement (“control bus i must not harm control bus j”) is enforced structurally by:
  - explicitly coupling control buses through a graph potential (penalize disagreement)
  - imposing a stability scaling derived from the base paper’s Jacobian condition

### Notes
- The stability bound is only as good as the `X` estimate and the assumption that the policy Jacobian is symmetric NSD.
  - The provided policy is constructed to be symmetric NSD (negative Hessian of convex potential).
