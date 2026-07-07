import os
import numpy as np
import gymnasium as gym
import pandapower as pp
import pandas as pd

class IEEE123bus(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1.0, vmax=1.05, vmin=0.95,
                 meas_noise_std=0.0, q_limits=(-2.0, 2.0), dq_max=0.25, disturb_cfg=None):
        self.network = pp_net
        self.obs_dim = 1
        self.action_dim = 1
        self.injection_bus = list(injection_bus)   # 0-based pp bus indices
        self.agentnum = len(self.injection_bus)
        self.v0 = v0
        self.vmax = vmax
        self.vmin = vmin
        self.q_limits = tuple(q_limits)
        self.dq_max = float(dq_max)
        self.meas_noise_std = float(meas_noise_std)

        # process noise config (same keys as 13-bus)
        self.disturb = disturb_cfg or dict(
            enabled=False,
            load_sigma=0.0, pv_sigma=0.0,
            load_step_prob=0.0, pv_step_prob=0.0,
            load_step_scale=0.0, pv_step_scale=0.0,
        )
        self._step_offsets = dict(load=0.0, pv=0.0)

        # snapshots to reset
        self.load0_p = np.copy(self.network.load['p_mw'])   if 'load' in self.network else np.array([])
        self.load0_q = np.copy(self.network.load['q_mvar']) if 'load' in self.network else np.array([])
        self.gen0_p  = np.copy(self.network.sgen['p_mw'])   if 'sgen' in self.network else np.array([])
        self.gen0_q  = np.copy(self.network.sgen['q_mvar']) if 'sgen' in self.network else np.array([])

        self.state = np.ones(self.agentnum, dtype=float)

        self._sanitize_net_dtypes()
        self._build_neighbors()
        self._build_sgen_map()

        self._runpp(init_dc=True)
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()

    # ---------- helpers ----------
    def _sanitize_net_dtypes(self):
        def _as_int(df, cols):
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int64')
        def _as_bool(df, col='in_service'):
            if col in df.columns:
                df[col] = df[col].astype(bool)

        net = self.network
        for name in ['load', 'sgen', 'gen', 'shunt', 'ext_grid', 'trafo', 'switch']:
            if name in net and len(net[name]):
                if 'bus' in net[name].columns:
                    _as_int(net[name], ['bus'])
                _as_bool(net[name])
        if 'line' in net and len(net.line):
            _as_int(net.line, ['from_bus', 'to_bus'])
            _as_bool(net.line)

    def _build_neighbors(self):
        self.bus_to_idx = {int(b): i for i, b in enumerate(self.injection_bus)}
        adj = {int(b): set() for b in self.injection_bus}
        if 'line' in self.network and len(self.network.line):
            for fb, tb in zip(self.network.line['from_bus'].to_numpy(),
                              self.network.line['to_bus'].to_numpy()):
                fb = int(fb); tb = int(tb)
                if fb in adj and tb in adj:
                    adj[fb].add(tb); adj[tb].add(fb)
        self.neighs = {b: sorted(list(adj[b])) for b in self.injection_bus}

    def _build_sgen_map(self):
        """Ensure each injection bus has an sgen row; remember its row index."""
        self.sgen_idx_for_inj = {}
        if 'sgen' not in self.network:
            self.network.sgen = pd.DataFrame(columns=['bus','p_mw','q_mvar','in_service'])
        sgen_bus = self.network.sgen['bus'].astype(int).to_numpy() if len(self.network.sgen) else np.array([], dtype=int)
        for b in self.injection_bus:
            b = int(b)
            matches = np.where(sgen_bus == b)[0]
            if matches.size == 0:
                pp.create_sgen(self.network, b, p_mw=0.0, q_mvar=0.0)
                idx = int(self.network.sgen.index[-1])
                sgen_bus = self.network.sgen['bus'].astype(int).to_numpy()
            else:
                idx = int(self.network.sgen.index[matches[0]])
            self.sgen_idx_for_inj[b] = idx

    def _runpp(self, init_dc=False):
        try:
            if init_dc:
                pp.runpp(self.network, algorithm='bfsw', init='dc', numba=False,
                         enforce_q_lims=True, calculate_voltage_angles=False,
                         tolerance_mva=1e-6, max_iteration=15)
            else:
                pp.runpp(self.network, algorithm='bfsw', init='results', numba=False,
                         recycle=True, enforce_q_lims=True, calculate_voltage_angles=False,
                         tolerance_mva=1e-6, max_iteration=15)
        except Exception:
            pp.runpp(self.network, algorithm='bfsw', init='dc', numba=False,
                     enforce_q_lims=True, calculate_voltage_angles=False,
                     tolerance_mva=1e-6, max_iteration=30)

    # ---------- cost ----------
    @staticmethod
    def cost(volts, action, vmin, vmax, lam_q=5.0, lam_v=100.0):
        v = np.asarray(volts, dtype=float).reshape(-1)
        a = np.asarray(action, dtype=float).reshape(-1)
        over  = np.clip(v - vmax, 0, np.inf)
        under = np.clip(vmin - v, 0, np.inf)
        return lam_q * np.linalg.norm(a, 1) + lam_v * (np.linalg.norm(over, 2)**2 + np.linalg.norm(under, 2)**2)


    def _apply_q_limits_and_rate(self, delta_q):
        """Apply per-step ΔQ limit and absolute Q limits, returning the applied ΔQ."""
        delta_q = np.asarray(delta_q, float).reshape(-1)
        if delta_q.size != self.agentnum:
            raise ValueError(f"delta_q must have size {self.agentnum}, got {delta_q.size}")
        delta_q = np.clip(delta_q, -self.dq_max, self.dq_max)

        qmin, qmax = self.q_limits
        prev_q = np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], "q_mvar"] for b in self.injection_bus], dtype=float)
        q_new = np.clip(prev_q + delta_q, qmin, qmax)
        return q_new - prev_q

    # ---------- training-style reward ----------
    def step_Preward(self, delta_q, p_action=None):
        """Incremental action step used during training (ΔQ semantics)."""
        u = self._apply_q_limits_and_rate(delta_q)
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, 'q_mvar'] += float(u[i])

        self._runpp(init_dc=False)

        v_true = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        obs = v_true if self.meas_noise_std <= 0 else v_true + np.random.normal(0.0, self.meas_noise_std, size=v_true.shape)

        reward = -self.cost(v_true, u, self.vmin, self.vmax)
        done = (np.min(v_true) >= self.vmin) and (np.max(v_true) <= self.vmax)

        q_ctrl = np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], 'q_mvar'] for b in self.injection_bus], dtype=float)
        info = {
            'v_true': v_true,
            'state_all': self.network.res_bus.vm_pu.to_numpy(),
            'delta_q_applied': u,
            'q_ctrl': q_ctrl,
        }
        self.state = obs
        return obs, float(reward), bool(done), info

    # ---------- process noise helpers ----------
    def _apply_disturbances(self, load_p, load_q, pv_p):
        if not self.disturb.get("enabled", False):
            return float(load_p), float(load_q), float(pv_p)
        lp = float(load_p) + np.random.normal(0, self.disturb.get("load_sigma", 0.0))
        lq = float(load_q) + np.random.normal(0, self.disturb.get("load_sigma", 0.0))
        pv = float(pv_p)   + np.random.normal(0, self.disturb.get("pv_sigma",   0.0))
        if np.random.rand() < self.disturb.get("load_step_prob", 0.0):
            self._step_offsets["load"] += np.random.uniform(-1, 1) * self.disturb.get("load_step_scale", 0.0)
        if np.random.rand() < self.disturb.get("pv_step_prob", 0.0):
            self._step_offsets["pv"]   += np.random.uniform(-1, 1) * self.disturb.get("pv_step_scale", 0.0)
        lp += self._step_offsets["load"]; lq += self._step_offsets["load"]; pv += self._step_offsets["pv"]
        return lp, lq, pv


    def step_profile(self, delta_q, load_p, load_q, pv_p):
        """Unified profile step (ΔQ + load/PV) for evaluation/DOE."""
        load_p, load_q, pv_p = self._apply_disturbances(load_p, load_q, pv_p)

        u = self._apply_q_limits_and_rate(delta_q)
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, 'q_mvar'] += float(u[i])

        # distribute aggregate load evenly across all loads
        if 'load' in self.network and len(self.network.load):
            nL = len(self.network.load)
            if nL > 0:
                shares = np.ones(nL, dtype=float) / float(nL)
                for idx, w in zip(self.network.load.index[:nL], shares):
                    self.network.load.at[idx, 'p_mw']   = float(load_p) * w
                    self.network.load.at[idx, 'q_mvar'] = float(load_q) * w

        # distribute PV profile across sgens (active power only)
        if 'sgen' in self.network and len(self.network.sgen):
            nS = len(self.network.sgen)
            if nS > 0:
                pv_sh = np.ones(nS, dtype=float) / float(nS)
                for idx, w in zip(self.network.sgen.index[:nS], pv_sh):
                    self.network.sgen.at[idx, 'p_mw'] = float(pv_p) * w

        self._runpp(init_dc=False)

        v_true = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        obs = v_true if self.meas_noise_std <= 0 else v_true + np.random.normal(0.0, self.meas_noise_std, size=v_true.shape)

        reward = -self.cost(v_true, u, self.vmin, self.vmax)
        done = (np.min(v_true) >= self.vmin) and (np.max(v_true) <= self.vmax)

        q_ctrl = np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], 'q_mvar'] for b in self.injection_bus], dtype=float)
        info = {
            'v_true': v_true,
            'state_all': self.network.res_bus.vm_pu.to_numpy(),
            'delta_q_applied': u,
            'q_ctrl': q_ctrl,
        }
        self.state = obs
        return obs, float(reward), bool(done), info


    # ---------- DOE-driven dynamics ----------
    def step_load(self, action, load_p, load_q, pv_p):
        load_p, load_q, pv_p = self._apply_disturbances(load_p, load_q, pv_p)

        a = np.clip(np.asarray(action, dtype=float), self.q_limits[0], self.q_limits[1])
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, 'q_mvar'] = float(a[i])

        if 'load' in self.network and len(self.network.load):
            nL = len(self.network.load)
            if nL > 0:
                shares = np.ones(nL, dtype=float) / float(nL)
                for idx, w in zip(self.network.load.index[:nL], shares):
                    self.network.load.at[idx, 'p_mw']   = float(load_p) * w
                    self.network.load.at[idx, 'q_mvar'] = float(load_q) * w

        if 'sgen' in self.network and len(self.network.sgen):
            nS = len(self.network.sgen)
            if nS > 0:
                pv_sh = np.ones(nS, dtype=float) / float(nS)
                for idx, w in zip(self.network.sgen.index[:nS], pv_sh):
                    self.network.sgen.at[idx, 'p_mw'] = float(pv_p) * w

        self._runpp(init_dc=False)

        v_true = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        v_meas = v_true if self.meas_noise_std <= 0 else \
                 v_true + np.random.normal(0.0, self.meas_noise_std, size=v_true.shape)

        reward = -self.cost(v_true, a, self.vmin, self.vmax)
        done = (np.min(v_true) >= self.vmin) and (np.max(v_true) <= self.vmax)
        self.state = v_meas
        info = {
            'v_true': v_true,
            'state_all': self.network.res_bus.vm_pu.to_numpy(),
            'q_ctrl': np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], 'q_mvar'] for b in self.injection_bus], dtype=float),
            'q_set': a,
        }
        self.state = v_meas
        return v_meas, float(reward), bool(done), info

    def reset(self, seed=1):
        self._step_offsets = dict(load=0.0, pv=0.0)
        if 'load' in self.network:
            self.network.load['p_mw']   = 0 * self.load0_p
            self.network.load['q_mvar'] = 0 * self.load0_q
        if 'sgen' in self.network:
            self.network.sgen['p_mw']   = 0 * self.gen0_p
            self.network.sgen['q_mvar'] = 0 * self.gen0_q
        self._runpp(init_dc=True)
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

# -----------------------------
# Robust builder for IEEE-123
# -----------------------------
def create_123bus(mat_path: str | None = None):
    """
    Build the IEEE-123 pandapower net from a MATPOWER .mat file.

    Path resolution priority:
      1) Explicit function arg: create_123bus(mat_path="C:/path/to/case_123.mat")
      2) Environment variable:  CASE123_MAT=C:\\path\\to\\case_123.mat
      3) Common project locations (both 'pandapower models' and 'pandapower_models')
      4) CWD / script dir fallbacks
    """
    from pathlib import Path
    import os

    # 1) explicit override
    if mat_path:
        p = Path(mat_path)
        if not p.exists():
            raise FileNotFoundError(f"Explicit mat_path does not exist: {p}")
        chosen = p
    else:
        # 2) env var
        env_path = os.getenv("CASE123_MAT")
        if env_path and Path(env_path).exists():
            chosen = Path(env_path)
        else:
            # 3) robust search of common folders (spaces/underscores, cwd/this file/parent)
            here = Path(__file__).resolve().parent
            root = here.parent
            cwd  = Path.cwd()

            candidates = [
                cwd / "pandapower models" / "case_123.mat",
                cwd / "pandapower_models" / "case_123.mat",
                here / "pandapower models" / "case_123.mat",
                here / "pandapower_models" / "case_123.mat",
                root / "pandapower models" / "case_123.mat",
                root / "pandapower_models" / "case_123.mat",
                # extra fallbacks
                cwd / "case_123.mat",
                here / "case_123.mat",
                root / "case_123.mat",
                # nested mistakes
                here / "pandapower models" / "pandapower models" / "case_123.mat",
                here / "pandapower_models" / "pandapower_models" / "case_123.mat",
            ]

            chosen = next((c for c in candidates if c.exists()), None)
            if chosen is None:
                tried = "\n  ".join(str(c) for c in candidates)
                raise FileNotFoundError(
                    "Could not find case_123.mat. Set CASE123_MAT, pass mat_path, "
                    "or place the file in one of these locations:\n  " + tried
                )

    print(f"[create_123bus] Using MATPOWER file: {chosen}")
    pp_net = pp.converter.from_mpc(str(chosen), casename_mpc_file='case_mpc')
    if 'sgen' in pp_net and len(pp_net.sgen):
        pp_net.sgen['p_mw'] = 0.0
        pp_net.sgen['q_mvar'] = 0.0

    # controllable buses (create if missing)
    for b in [9, 10, 15, 19, 32, 35, 47, 58, 65, 74, 82, 91, 103, 60]:
        if 'sgen' not in pp_net or b not in set(pp_net.sgen.bus.values):
            pp.create_sgen(pp_net, b, p_mw=0.0, q_mvar=0.0)
    # extras (reset logic safety)
    for b in [13, 14, 18]:
        if 'sgen' not in pp_net or b not in set(pp_net.sgen.bus.values):
            pp.create_sgen(pp_net, b, p_mw=0.0, q_mvar=0.0)
    return pp_net
    @property
    def ctrl_buses(self):
        return list(self.injection_bus)

    def get_ctrl_voltages(self):
        return self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy().astype(float)

    def get_all_voltages(self):
        return self.network.res_bus.vm_pu.to_numpy().astype(float)

    def get_ctrl_q(self):
        return np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], 'q_mvar'] for b in self.injection_bus], dtype=float)

    def set_ctrl_q(self, q_set):
        q_set = np.asarray(q_set, dtype=float).reshape(-1)
        assert q_set.size == self.agentnum
        qmin, qmax = self.q_limits
        q_set = np.clip(q_set, qmin, qmax)
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, 'q_mvar'] = float(q_set[i])
        self._runpp(init_dc=False)
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state


