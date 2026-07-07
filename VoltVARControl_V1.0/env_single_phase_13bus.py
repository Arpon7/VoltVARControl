# env_single_phase_13bus.py
# Single-phase IEEE-13-like environment (pandapower)
# - ΔQ semantics for training via step_Preward()
# - |Q| semantics for DOE/playback via step_load()
# - Per-step rate limit and absolute Q limits
# - Neighbor map + bus_to_idx for attention mixer
# - Robust BFSW with NR fallback to avoid non-convergence

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pandapower as pp


@dataclass
class EnvConfig:
    dq_max: float = 0.30                          # MVar per-step rate limit for ΔQ
    vmin: float = 0.95
    vmax: float = 1.05
    controllable_buses: Iterable[int] = (2, 7, 9)
    q_limits_global: Tuple[float, float] = (-5.0, 5.0)


class IEEE13bus:
    """
    Minimal single-phase IEEE-13-like env (pandapower) with the same public API as your 123-bus env:
      - step_Preward(Δq)         -> training (incremental control)
      - step_load(|q|, P, Q, PV) -> DOE/playback (absolute setpoints)
      - injection_bus, neighs, bus_to_idx
    """

    def __init__(
        self,
        pp_net=None,
        injection_bus: Iterable[int] = (2, 7, 9),
        v0: float = 1.0,
        vmax: float = 1.05,
        vmin: float = 0.95,
        all_bus: bool = False,
        disturb_cfg=None,
        meas_noise_std: float = 0.0,
        q_limits: Tuple[float, float] = (-5.0, 5.0),
    ):
        cfg = EnvConfig(
            controllable_buses=tuple(int(b) for b in injection_bus),
            vmin=vmin,
            vmax=vmax,
            q_limits_global=q_limits,
        )
        self.vmin, self.vmax = cfg.vmin, cfg.vmax
        self.injection_bus: List[int] = list(cfg.controllable_buses)
        self.agentnum = len(self.injection_bus)
        self.meas_noise_std = float(meas_noise_std)
        self.q_limits = tuple(q_limits)           # (min, max) absolute Q limits per device
        self.dq_max = float(cfg.dq_max)           # per-step ΔQ limit

        # --- build / accept the network ---
        if pp_net is None:
            self.network = self._build_net_base()
        else:
            self.network = pp_net

        # Ensure controllable sgens exist on each injection bus
        self._build_sgen_map()

        # Neighbor map (1-hop) among controllable buses + bus_to_idx
        self._build_neighbors()

        # Snapshots for reset
        self.load0_p = np.copy(self.network.load['p_mw']) if 'load' in self.network else np.array([])
        self.load0_q = np.copy(self.network.load['q_mvar']) if 'load' in self.network else np.array([])
        self.gen0_p = np.copy(self.network.sgen['p_mw']) if 'sgen' in self.network else np.array([])
        self.gen0_q = np.copy(self.network.sgen['q_mvar']) if 'sgen' in self.network else np.array([])

        # Initial PF
        self._runpp(init_dc=True)
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()

    # -------------------- network builder --------------------
    def _build_net_base(self) -> pp.pandapowerNet:
        """
        Build a small, well-conditioned single-phase radial feeder (14 buses: 0..13).
        Bus 0 is the slack. Lines form a chain 0-1-2-…-13. Loads are modest.
        Includes a few PV-like sgens (negative P) and a tiny tail shunt.
        """
        vn_kv = 12.66  # typical for IEEE 13-bus
        net = pp.create_empty_network(sn_mva=1.0)

        # Buses
        buses = [pp.create_bus(net, vn_kv=vn_kv, name=f"bus_{i}") for i in range(14)]

        # Slack
        pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0, name="slack")

        # Lines (radial chain)
        r_ohm_per_km = 0.4
        x_ohm_per_km = 0.25
        c_nf_per_km = 0.0
        max_i_ka = 0.4
        length_km = 0.3

        for i in range(1, len(buses)):
            pp.create_line_from_parameters(
                net,
                from_bus=buses[i - 1],
                to_bus=buses[i],
                length_km=length_km,
                r_ohm_per_km=r_ohm_per_km,
                x_ohm_per_km=x_ohm_per_km,
                c_nf_per_km=c_nf_per_km,
                max_i_ka=max_i_ka,
                name=f"line_{i-1}_{i}",
            )

        # Loads (spread out, modest)
        load_spec = {
            3: (0.08, 0.03),
            4: (0.05, 0.02),
            6: (0.07, 0.025),
            8: (0.06, 0.02),
            10: (0.05, 0.018),
            12: (0.07, 0.025),
            13: (0.06, 0.02),
        }
        for b, (p_mw, q_mvar) in load_spec.items():
            pp.create_load(net, bus=buses[b], p_mw=p_mw, q_mvar=q_mvar, name=f"load_{b}")

        # PV-like sgens (negative P, zero Q initially)
        for b in [5, 7, 10]:
            pp.create_sgen(net, bus=buses[b], p_mw=-0.05, q_mvar=0.0, name=f"pv_{b}")

        # Tiny shunt at the tail helps NR if DOE pushes hard
        pp.create_shunt(net, bus=buses[13], q_mvar=0.01, p_mw=0.0, name="tail_shunt")

        # Sanitize dtypes (important on Windows)
        self._sanitize_net_dtypes(net)

        # Initial solve with robust fallback
        try:
            pp.runpp(
                net,
                algorithm="bfsw",
                init="dc",
                numba=False,
                enforce_q_lims=True,
                calculate_voltage_angles=False,
                tolerance_mva=1e-6,
                max_iteration=20,
            )
        except Exception:
            pp.runpp(
                net,
                algorithm="nr",
                init="flat",
                numba=False,
                enforce_q_lims=True,
                calculate_voltage_angles=False,
                tolerance_mva=1e-7,
                max_iteration=50,
            )

        return net

    # -------------------- helpers --------------------
    @staticmethod
    def _sanitize_net_dtypes(net):
        def _as_int(df, cols):
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

        def _as_bool(df, col="in_service"):
            if col in df.columns:
                df[col] = df[col].astype(bool)

        for name in ["load", "sgen", "gen", "shunt", "ext_grid", "trafo", "switch"]:
            if name in net and len(net[name]):
                if "bus" in net[name].columns:
                    _as_int(net[name], ["bus"])
                _as_bool(net[name])
        if "line" in net and len(net.line):
            _as_int(net.line, ["from_bus", "to_bus"])
            _as_bool(net.line)

    def _build_sgen_map(self):
        """Ensure a controllable sgen exists at each controllable bus (for Q control)."""
        if "sgen" not in self.network:
            self.network.sgen = pd.DataFrame(columns=["bus", "p_mw", "q_mvar", "in_service", "name"])
        sgen_bus = (
            self.network.sgen["bus"].astype(int).to_numpy() if len(self.network.sgen) else np.array([], dtype=int)
        )
        self.sgen_idx_for_inj: Dict[int, int] = {}
        for b in self.injection_bus:
            b = int(b)
            matches = np.where(sgen_bus == b)[0]
            if matches.size == 0:
                pp.create_sgen(self.network, b, p_mw=0.0, q_mvar=0.0, name=f"ctrl_{b}")
                sgen_bus = self.network.sgen["bus"].astype(int).to_numpy()
                idx = int(self.network.sgen.index[-1])
            else:
                idx = int(self.network.sgen.index[matches[0]])
            self.sgen_idx_for_inj[b] = idx

    def _build_neighbors(self):
        """Build a 1-hop neighbor map among controllable buses and bus_to_idx."""
        self.bus_to_idx: Dict[int, int] = {int(b): i for i, b in enumerate(self.injection_bus)}
        adj: Dict[int, set] = {int(b): set() for b in self.injection_bus}
        if "line" in self.network and len(self.network.line):
            for fb, tb in zip(self.network.line["from_bus"].to_numpy(), self.network.line["to_bus"].to_numpy()):
                fb, tb = int(fb), int(tb)
                if fb in adj and tb in adj:
                    adj[fb].add(tb)
                    adj[tb].add(fb)
        self.neighs: Dict[int, List[int]] = {b: sorted(list(adj[b])) for b in self.injection_bus}

    def _runpp(self, init_dc: bool = False):
        """Run power flow with BFSW; fallback to NR if needed."""
        try:
            pp.runpp(
                self.network,
                algorithm="bfsw",
                init="dc" if init_dc else "results",
                numba=False,
                recycle=not init_dc,
                enforce_q_lims=True,
                calculate_voltage_angles=False,
                tolerance_mva=1e-6,
                max_iteration=20,
            )
        except Exception:
            pp.runpp(
                self.network,
                algorithm="nr",
                init="flat",
                numba=False,
                enforce_q_lims=True,
                calculate_voltage_angles=False,
                tolerance_mva=1e-7,
                max_iteration=50,
            )

    # -------------------- cost --------------------
    @staticmethod
    def cost(volts, action, vmin, vmax, lam_q=5.0, lam_v=100.0):
        v = np.asarray(volts, float).reshape(-1)
        a = np.asarray(action, float).reshape(-1)
        over = np.clip(v - vmax, 0.0, np.inf)
        under = np.clip(vmin - v, 0.0, np.inf)
        return lam_q * np.linalg.norm(a, 1) + lam_v * (np.linalg.norm(over, 2) ** 2 + np.linalg.norm(under, 2) ** 2)

    # -------------------- ΔQ clamp for training --------------------
    def _apply_q_limits_and_rate(self, delta_q: np.ndarray) -> np.ndarray:
        """Apply per-step rate limit to ΔQ and absolute Q limits to the resulting |Q|."""
        delta_q = np.asarray(delta_q, float).reshape(-1)
        delta_q = np.clip(delta_q, -self.dq_max, self.dq_max)  # per-step ΔQ limit

        qmin, qmax = self.q_limits
        prev_q = np.array(
            [self.network.sgen.at[self.sgen_idx_for_inj[int(b)], "q_mvar"] for b in self.injection_bus], dtype=float
        )
        q_new = np.clip(prev_q + delta_q, qmin, qmax)
        return q_new - prev_q

    # -------------------- training step (ΔQ semantics) --------------------
    def step_Preward(self, delta_q, p_action=None):
        """Incremental action step used during training (ΔQ semantics)."""
        u = self._apply_q_limits_and_rate(delta_q)
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, "q_mvar"] += float(u[i])

        self._runpp(init_dc=False)

        v_true = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        if self.meas_noise_std > 0.0:
            obs = v_true + np.random.normal(0.0, self.meas_noise_std, size=v_true.shape)
        else:
            obs = v_true

        reward = -self.cost(v_true, u, self.vmin, self.vmax)
        done = (np.min(v_true) >= self.vmin) and (np.max(v_true) <= self.vmax)

        q_ctrl = np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], "q_mvar"] for b in self.injection_bus], dtype=float)
        info = {
            "v_true": v_true,
            "state_all": self.network.res_bus.vm_pu.to_numpy(),
            "delta_q_applied": u,
            "q_ctrl": q_ctrl,
        }
        self.state = obs
        return obs, float(reward), bool(done), info

    # -------------------- DOE playback step (|Q| semantics) --------------------

    # -------------------- unified profile step (ΔQ + loads/PV) --------------------
    def step_profile(self, delta_q, load_p, load_q, pv_p):
        """
        Unified dynamics step for evaluation/DOE: applies incremental ΔQ plus feeder load/PV profiles.
        Returns (obs, reward, done, info) where obs is control-bus measured voltages.
        """
        # 1) apply incremental Q (ΔQ semantics)
        u = self._apply_q_limits_and_rate(delta_q)
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, "q_mvar"] += float(u[i])

        # 2) apply aggregate load evenly across all loads
        if "load" in self.network and len(self.network.load):
            nL = len(self.network.load)
            if nL > 0:
                w = np.ones(nL, dtype=float) / float(nL)
                for idx, wi in zip(self.network.load.index[:nL], w):
                    self.network.load.at[idx, "p_mw"] = float(load_p) * wi
                    self.network.load.at[idx, "q_mvar"] = float(load_q) * wi

        # 3) apply aggregate PV evenly across all sgens (active power only; Q is controlled)
        if "sgen" in self.network and len(self.network.sgen):
            nS = len(self.network.sgen)
            if nS > 0:
                pv_sh = np.ones(nS, dtype=float) / float(nS)
                for idx, wi in zip(self.network.sgen.index[:nS], pv_sh):
                    # keep q_mvar as set by controller; override p_mw profile
                    self.network.sgen.at[idx, "p_mw"] = float(pv_p) * wi

        # 4) solve PF
        self._runpp(init_dc=False)

        # 5) observe voltages
        v_true = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        if self.meas_noise_std > 0.0:
            obs = v_true + np.random.normal(0.0, self.meas_noise_std, size=v_true.shape)
        else:
            obs = v_true

        # 6) reward / done
        reward = -self.cost(v_true, u, self.vmin, self.vmax)
        done = (np.min(v_true) >= self.vmin) and (np.max(v_true) <= self.vmax)

        # 7) info
        q_ctrl = np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], "q_mvar"] for b in self.injection_bus], dtype=float)
        info = {
            "v_true": v_true,
            "state_all": self.network.res_bus.vm_pu.to_numpy(),
            "delta_q_applied": u,
            "q_ctrl": q_ctrl,
        }
        self.state = obs
        return obs, float(reward), bool(done), info

    # -------------------- small convenience helpers --------------------
    @property
    def ctrl_buses(self):
        return list(self.injection_bus)

    def get_ctrl_voltages(self):
        return self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy().astype(float)

    def get_all_voltages(self):
        return self.network.res_bus.vm_pu.to_numpy().astype(float)

    def get_ctrl_q(self):
        return np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], "q_mvar"] for b in self.injection_bus], dtype=float)

    def set_ctrl_q(self, q_set):
        q_set = np.asarray(q_set, dtype=float).reshape(-1)
        assert q_set.size == self.agentnum
        qmin, qmax = self.q_limits
        q_set = np.clip(q_set, qmin, qmax)
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, "q_mvar"] = float(q_set[i])
        self._runpp(init_dc=False)
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state
    def step_load(self, q_set, load_p, load_q, pv_p):
        """
        Absolute |Q| step used during DOE playback / projector. Also applies feeder P/Q/PV.
        """
        # Clamp |Q|
        a = np.clip(np.asarray(q_set, float).reshape(-1), self.q_limits[0], self.q_limits[1])
        for i, b in enumerate(self.injection_bus):
            sidx = self.sgen_idx_for_inj[int(b)]
            self.network.sgen.at[sidx, "q_mvar"] = float(a[i])

        # Distribute aggregate load evenly across all loads (simple & robust)
        if "load" in self.network and len(self.network.load):
            nL = len(self.network.load)
            if nL > 0:
                w = np.ones(nL, dtype=float) / float(nL)
                for idx, wi in zip(self.network.load.index[:nL], w):
                    self.network.load.at[idx, "p_mw"] = float(load_p) * wi
                    self.network.load.at[idx, "q_mvar"] = float(load_q) * wi

        # Distribute PV across all sgens (note: includes controllable ones—OK for simple playback)
        if "sgen" in self.network and len(self.network.sgen):
            nS = len(self.network.sgen)
            if nS > 0:
                w = np.ones(nS, dtype=float) / float(nS)
                for idx, wi in zip(self.network.sgen.index[:nS], w):
                    self.network.sgen.at[idx, "p_mw"] = float(pv_p) * wi

        self._runpp(init_dc=False)

        v_true = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        v_meas = (
            v_true + np.random.normal(0.0, self.meas_noise_std, size=v_true.shape)
            if self.meas_noise_std > 0.0
            else v_true
        )
        self.state = v_meas
        reward = -self.cost(v_true, a, self.vmin, self.vmax)
        done = (np.min(v_true) >= self.vmin) and (np.max(v_true) <= self.vmax)
        info = {
            "v_true": v_true,
            "state_all": self.network.res_bus.vm_pu.to_numpy(),
            "q_ctrl": np.array([self.network.sgen.at[self.sgen_idx_for_inj[int(b)], "q_mvar"] for b in self.injection_bus], dtype=float),
            "q_set": a,
        }
        self.state = v_meas
        return v_meas, float(reward), bool(done), info

    def reset(self, seed: int = 1):
        np.random.seed(int(seed))
        if "load" in self.network:
            self.network.load["p_mw"] = 0.0 * self.load0_p
            self.network.load["q_mvar"] = 0.0 * self.load0_q
        if "sgen" in self.network:
            self.network.sgen["p_mw"] = 0.0 * self.gen0_p
            self.network.sgen["q_mvar"] = 0.0 * self.gen0_q
        self._runpp(init_dc=True)
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state


# Keep the factory so existing imports continue to work (used by IEEE_13_3p.py, etc.)
def create_13bus():
    """
    Returns a pandapowerNet for the single-phase 13-bus-like feeder.
    The IEEE13bus class will add controllable sgens for the specified buses.
    """
    env_tmp = IEEE13bus(pp_net=None, injection_bus=(2, 7, 9))
    return env_tmp.network


