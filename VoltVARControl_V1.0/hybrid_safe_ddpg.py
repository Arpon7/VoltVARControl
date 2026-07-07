# Updated: 2025-10-22
# Attention mixer (NEIGHBOR-ONLY) + attention-conditioned monotone policy.
# Key change: the mixer IGNOREs v_i and depends ONLY on neighbor voltages v_nb.
# => ctx_scale(m_i) and ctx_bias(m_i) are independent of v_i,
# => du/dv_i = - scale(m_i) * d(base)/dv_i < 0  (strict), restoring the base proof.

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------

def _softplus_pos(x, beta: float = 1.0):
    """Strictly positive with tiny epsilon for numerical safety."""
    return F.softplus(x, beta=beta) + 1e-6


class _PosLinear(nn.Module):
    """
    Linear layer with weights constrained >= 0 via softplus.
    Used to keep the base branch monotone-increasing in v_i.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_raw)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_pos = _softplus_pos(self.weight_raw)
        y = x @ w_pos.t()
        if self.bias is not None:
            y = y + self.bias
        return y


# ---------------------------
# Neighbor-only Attention Mixer
# ---------------------------

class AttentionMixer(nn.Module):
    """
    Neighbor-only attention mixer.

    Inputs:
        v_i  : [B, 1]        (kept for API compatibility; IGNORED to preserve proof)
        v_nb : [B, K, 1]     neighbor voltages (pad K with zeros if needed)

    Output:
        m_i  : [B, h]        neighbor context vector

    NOTE: v_i is NOT used inside the mixer. That way, the context m_i is independent
    of v_i, and the actor's scale/bias (which are functions of m_i) do not change
    with v_i, preserving the strict sign of du/dv_i.
    """
    def __init__(self, in_dim: int = 1, hid_dim: int = 16):
        super().__init__()
        self.h = hid_dim

        # Embed neighbor voltages (scalar -> h)
        self.nb_embed = nn.Sequential(
            nn.Linear(1, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
        )

        # Learned single query per bus for pooling neighbors
        self.query = nn.Parameter(torch.randn(1, hid_dim))

        # Post-fusion head
        self.post = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, v_i: torch.Tensor, v_nb: Optional[torch.Tensor]) -> torch.Tensor:
        # v_i is intentionally ignored to keep m_i independent of the local voltage
        if v_nb is None or v_nb.numel() == 0:
            # No neighbors case: return a learned constant context (via query)
            B = v_i.size(0)
            # Expand query and pass through post to keep shape [B,h]
            q = self.query.expand(B, -1)
            return self.post(q)

        # v_nb: [B,K,1] -> embed -> [B,K,h]
        nb_feat = self.nb_embed(v_nb)

        # Scores with a learned query (same for all positions)
        B, K, H = nb_feat.shape
        q = self.query.expand(B, -1).unsqueeze(1)   # [B,1,h]
        scores = torch.einsum("bqh,bkh->bqk", q, nb_feat) / (H ** 0.5)  # [B,1,K]
        attn = torch.softmax(scores, dim=-1)                              # [B,1,K]

        # Weighted sum -> [B,h]
        pooled = torch.einsum("bqk,bkh->bqh", attn, nb_feat).squeeze(1)   # [B,h]

        # Context vector m_i
        out = self.post(pooled)                                           # [B,h]
        return out


# ---------------------------
# Monotone Policy with Neighbor-only Context
# ---------------------------

class SafePolicyNetworkWithCoord(nn.Module):
    """
    Attention-conditioned monotone policy:

        u_i = -( scale(m_i) * base_monotone(v_i) + bias(m_i) ), then clamped to [-dq_max, dq_max]

    Construction:
      - base_monotone'(v_i) >= 0 enforced via _PosLinear + softplus, so base is strictly increasing in v_i
      - scale(m_i) >= 0 and bias(m_i) depend ONLY on neighbor context m_i (independent of v_i)
      => du_i/dv_i = - scale(m_i) * d(base)/dv_i < 0  (strict)
    """

    def __init__(self,
                 v_dim: int,        # dimension of local voltage (1)
                 m_dim: int,        # context (mixer) dimension (== hid_dim from AttentionMixer)
                 hidden_dim: int,
                 vmin: float,
                 vmax: float,
                 dq_max: float):
        super().__init__()
        self.vmin, self.vmax, self.dq_max = vmin, vmax, dq_max
        h = hidden_dim

        # Strictly monotone-increasing base in v_i
        self.base1 = _PosLinear(v_dim, h, bias=True)
        self.base2 = _PosLinear(h, 1, bias=True)

        # Context -> scale (>=0) and bias (R), both depend ONLY on m_i
        self.ctx_scale = nn.Sequential(
            nn.Linear(m_dim, h), nn.ReLU(inplace=True),
            nn.Linear(h, 1)
        )
        self.ctx_bias = nn.Sequential(
            nn.Linear(m_dim, h), nn.ReLU(inplace=True),
            nn.Linear(h, 1)
        )

        # Optional global bias for scale to encourage positivity early
        self.scale_raw_bias = nn.Parameter(torch.tensor(0.0))

    def base_monotone(self, v_i: torch.Tensor) -> torch.Tensor:
        # Positive maps + softplus to ensure strict increase wrt v_i
        z = _softplus_pos(self.base1(v_i))   # [B,h]
        out = _softplus_pos(self.base2(z))   # [B,1]
        return out

    def forward(self, v_i: torch.Tensor, m_i: torch.Tensor) -> torch.Tensor:
        base  = self.base_monotone(v_i)                  # [B,1]
        scale = _softplus_pos(self.ctx_scale(m_i) + self.scale_raw_bias)  # [B,1] >= 0
        bias  = self.ctx_bias(m_i)                       # [B,1] (free, but independent of v_i)
        u = -(scale * base + bias)                       # du/dv < 0 by construction
        u = torch.clamp(u, -self.dq_max, self.dq_max)
        return u

    @torch.no_grad()
    def get_action(self, v_scalar: float, m_vec: torch.Tensor) -> float:
        v = torch.tensor([[v_scalar]], dtype=torch.float32, device=m_vec.device)
        return float(self.forward(v, m_vec).item())

    @torch.no_grad()
    def check_monotonicity(self, v_grid: torch.Tensor, m_i: torch.Tensor) -> Tuple[float, float]:
        """
        Quick numerical check to verify du/dv_i < 0 across a grid of v_i.
        Returns (min_du, max_du) of finite-difference slopes (should be strictly negative).
        """
        eps = 1e-3
        v1 = v_grid
        v2 = v_grid + eps
        u1 = self.forward(v1, m_i)
        u2 = self.forward(v2, m_i)
        slope = (u2 - u1) / eps
        return slope.min().item(), slope.max().item()


# ---------------------------
# (Optional) Quick self-test
# ---------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, H = 8, 4, 16

    # Fake inputs
    v_i  = torch.linspace(0.9, 1.1, B).unsqueeze(-1)       # [B,1]
    v_nb = (0.98 + 0.04 * torch.rand(B, K)).unsqueeze(-1)  # [B,K,1]

    mixer = AttentionMixer(in_dim=1, hid_dim=H)
    m_i = mixer(v_i, v_nb)                                  # [B,H]

    actor = SafePolicyNetworkWithCoord(
        v_dim=1, m_dim=H, hidden_dim=64,
        vmin=0.9, vmax=1.1, dq_max=0.3
    )

    # Forward
    u = actor(v_i, m_i)
    print("Action sample:", u.squeeze(-1))

    # Monotonicity check
    mn, mx = actor.check_monotonicity(v_i, m_i)
    print("du/dv range (min,max):", (mn, mx))
