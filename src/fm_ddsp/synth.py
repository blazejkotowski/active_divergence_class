"""
DDX7-style FM operator bank with fixed frequency ratios and DX7 routing.

Based on Caspe et al. 2022: "DDX7: Differentiable FM Synthesis of Musical
Instrument Sounds."

Key design decisions from the paper:
  - Frequency ratios are FIXED (DX7 patch constraint); only amplitudes/mod-indices
    are learned.  This avoids the gradient ambiguity of jointly optimising both
    frequency and amplitude in MSS loss.
  - Modulators are scaled by I_max, carriers by 1 — both from sigmoid envelopes.
  - Routing is specified as a list of (modulator, carrier) index pairs.
  - Operators are synthesised in topological order (no feedback loops).

Violin "STRINGS 1" patch (DDX7 paper, Figure 3):
    Ratios : [14, 3, 1, 1, 1, 1]
    Routing: Op0(14) → Op1(3) → Op2(1)[carrier]
             Op3(1)  → Op4(1)[carrier]
             Op5(1)  [carrier]
    I_max  = 2 (best for violin per §5.2.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Violin DX7 patch ─────────────────────────────────────────────────────────

VIOLIN_RATIOS   = [14.0, 3.0, 1.0, 1.0, 1.0, 1.0]
VIOLIN_ROUTING  = [(0, 1), (1, 2), (3, 4)]   # (modulator_idx, carrier_idx)
VIOLIN_CARRIERS = [2, 4, 5]
VIOLIN_I_MAX    = 2.0


class FMOperatorBank(nn.Module):
    """
    N-operator FM synthesizer bank with fixed frequency ratios.

    Operators are computed in topological order.  The modulator output is added
    as a phase term to the carrier's instantaneous phase — standard linear FM.

    Args:
        ratios:      list of N_ops frequency ratios (relative to f0)
        routing:     list of (modulator_idx, carrier_idx) pairs — who modulates whom
        carriers:    indices of operators whose output is summed to the audio output
        I_max:       maximum modulation index (modulators scaled by this, carriers by 1)
        sample_rate: audio sample rate in Hz
        hop_length:  envelope frame → audio sample stride
    """

    def __init__(
        self,
        ratios: list[float] = VIOLIN_RATIOS,
        routing: list[tuple[int, int]] = VIOLIN_ROUTING,
        carriers: list[int] = VIOLIN_CARRIERS,
        I_max: float = VIOLIN_I_MAX,
        sample_rate: int = 16000,
        hop_length: int = 64,
    ):
        super().__init__()
        self.register_buffer("fixed_ratios", torch.tensor(ratios, dtype=torch.float32))
        self.routing  = routing
        self.carriers = carriers
        self.I_max    = I_max
        self.sr       = sample_rate
        self.hop      = hop_length
        self.N_ops    = len(ratios)

        # Build maps for synthesis
        self._is_modulator = {m for m, _ in routing}
        self._incoming: dict[int, list[int]] = {k: [] for k in range(self.N_ops)}
        for m, c in routing:
            self._incoming[c].append(m)

        # Topological order (Kahn's algorithm on DAG)
        self._topo_order = self._toposort()

    def _toposort(self) -> list[int]:
        """Return operator indices in topological order (dependencies first)."""
        in_deg = {k: len(self._incoming[k]) for k in range(self.N_ops)}
        queue  = [k for k in range(self.N_ops) if in_deg[k] == 0]
        order  = []
        while queue:
            k = queue.pop(0)
            order.append(k)
            for m, c in self.routing:
                if m == k:
                    in_deg[c] -= 1
                    if in_deg[c] == 0:
                        queue.append(c)
        return order

    def forward(
        self,
        f0: torch.Tensor,            # (B, N_frames) Hz
        envelopes: torch.Tensor,     # (B, N_ops, N_frames) in [0, 1]
        ratios: torch.Tensor | None = None,  # override fixed_ratios (for Flex model)
    ) -> torch.Tensor:               # (B, T) audio
        B, N_frames = f0.shape
        T = N_frames * self.hop
        device = f0.device

        r = ratios if ratios is not None else self.fixed_ratios.to(device)  # (N_ops,)

        # Upsample f0 and envelopes to audio rate
        f0_up  = F.interpolate(f0.unsqueeze(1), T, mode="linear", align_corners=True
                               ).squeeze(1)                      # (B, T)
        env_up = F.interpolate(envelopes, T, mode="linear", align_corners=True
                               )                                 # (B, N_ops, T)

        # Scale: modulators → [0, I_max], carriers → [0, 1]
        scale = torch.ones(self.N_ops, device=device)
        for m in self._is_modulator:
            scale[m] = self.I_max
        env_up = env_up * scale.view(1, self.N_ops, 1)

        # Compute base phases: φ_k(t) = 2π × ratio_k × ∫f0(t)/sr dt  — (B, N_ops, T)
        base_phases = 2.0 * torch.pi * torch.cumsum(
            r.view(1, self.N_ops, 1) * f0_up.unsqueeze(1) / self.sr, dim=-1
        )

        # Synthesise operators in topological order, accumulating phase modulation
        op_out: dict[int, torch.Tensor] = {}
        for k in self._topo_order:
            pm = sum(op_out[m] for m in self._incoming[k]) if self._incoming[k] else 0.0
            phase = base_phases[:, k, :] + (pm if isinstance(pm, torch.Tensor) else 0.0)
            op_out[k] = env_up[:, k, :] * torch.sin(phase)   # (B, T)

        # Sum carrier outputs
        audio = sum(op_out[c] for c in self.carriers)
        return audio                                           # (B, T)
