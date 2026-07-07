# env_single_phase_14bus.py
from __future__ import annotations
from typing import Sequence
import pandapower as pp
import pandapower.networks as pn

# Reuse the single-phase 123-bus env implementation
from env_single_phase_123bus import IEEE123bus as _BaseEnv

# Default control buses (0-based): corresponds to buses 3,4,9,14 in 1-based indexing
DEFAULT_CTRL_BUSES_14 = [2, 3, 8, 13]

def create_14bus() -> pp.pandapowerNet:
    """
    Build an IEEE-14 network using pandapower's canned test system.
    Ensure there are sgens on our control buses so we can command q_mvar.
    """
    net = pn.case14()
    # Make sure sgen table exists
    if not hasattr(net, "sgen"):
        pp.create_empty_tables(net)
    existing = set(net.sgen.bus.values) if hasattr(net, "sgen") and len(net.sgen) else set()
    for b in DEFAULT_CTRL_BUSES_14:
        if b not in existing:
            pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0, name=f"vc_sgen_{b}")
    return net

class IEEE14bus(_BaseEnv):
    """Same behavior/API as IEEE123bus; we just use a 14-bus net and different defaults."""
    def __init__(self,
                 pp_net: pp.pandapowerNet,
                 injection_bus: Sequence[int] = DEFAULT_CTRL_BUSES_14,
                 v0: float = 1.0,
                 vmax: float = 1.05,
                 vmin: float = 0.95,
                 meas_noise_std: float = 0.0,
                 q_limits = (-2.0, 2.0),
                 dq_max: float = 0.25,
                 disturb_cfg = None):
        super().__init__(pp_net, injection_bus, v0, vmax, vmin,
                         meas_noise_std=meas_noise_std, q_limits=q_limits, dq_max=dq_max, disturb_cfg=disturb_cfg)

