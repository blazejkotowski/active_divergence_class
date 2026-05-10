# `src/fm_ddsp` — Differentiable FM Synthesis

A DDX7-style differentiable FM synthesizer for instrument resynthesis, based on Caspe et al. (2022). The key constraint is that frequency ratios are fixed to a known DX7 patch — only operator amplitudes and modulation indices are learned. This avoids the gradient ambiguity that arises when jointly optimizing frequency and amplitude in a multi-scale spectral loss and is what makes the model trainable in practice.

---

## Module overview

| File | Purpose |
|---|---|
| `synth.py` | FM operator bank with fixed ratios and topological synthesis order |
| `decoder.py` | GRU+MLP decoder: (f0, loudness) → per-operator envelopes + noise magnitudes |
| `model.py` | Full FM model; fixed-ratio and learnable-ratio variants |
| `loss.py` | Placeholder; FM training uses `MultiScaleSpectralLoss` from `src/ddsp` |

---

## `synth.py`

### Violin "STRINGS 1" patch constants

```python
VIOLIN_RATIOS   = [14.0, 3.0, 1.0, 1.0, 1.0, 1.0]
VIOLIN_ROUTING  = [(0, 1), (1, 2), (3, 4)]  # (modulator_idx, carrier_idx)
VIOLIN_CARRIERS = [2, 4, 5]
VIOLIN_I_MAX    = 2.0
```

The patch matches DDX7 paper Figure 3. Op0 (ratio 14) modulates Op1 (ratio 3) which modulates Op2 (ratio 1, carrier); Op3 (ratio 1) modulates Op4 (carrier); Op5 is an independent carrier. `I_max = 2` was found optimal for violin in the DDX7 paper (§5.2.1).

### `FMOperatorBank(ratios, routing, carriers, I_max, sample_rate, hop_length)`
N-operator FM bank with fixed frequency ratios. Operators are synthesized in topological order (Kahn's algorithm on the routing DAG) so modulators are always computed before the carriers they affect.

**`forward(f0, envelopes, ratios=None) → (B, T)`**

- `f0`: `(B, N_frames)` Hz
- `envelopes`: `(B, N_ops, N_frames)` — sigmoid envelope per operator, range `[0, 1]`
- `ratios`: optional override of `fixed_ratios` buffer (used by `DDSPFMModelFlex`)

Processing:
1. Upsample f0 and envelopes to audio rate via linear interpolation.
2. Scale envelopes: modulators → `[0, I_max]`, carriers → `[0, 1]`.
3. Compute base phase for each operator: `φ_k(t) = 2π × ratio_k × ∫f0(t)/sr dt`.
4. In topological order: accumulate phase modulation from incoming modulators, then compute `envelope_k × sin(phase_k)`.
5. Sum carrier outputs.

---

## `decoder.py`

### `FMDecoder(n_ops, n_noise_bands, hidden_size, n_mlp_layers)`
Maps per-frame `(f0_hz, loudness)` to per-operator FM envelopes and noise band magnitudes. Mirrors the DDSP decoder architecture: input MLP → GRU → output MLP → two output heads.

F0 is log-normalized using the violin pitch range constants (log(32.7)–log(2093) mapped to [0, 1]).

**`forward(f0_hz, loudness) → (envelopes, noise_mags)`**

- `envelopes`: `(B, N_ops, N_frames)` — sigmoid, ready for `FMOperatorBank`
- `noise_mags`: `(B, N_frames, N_bands)` — sigmoid, ready for `FilteredNoise`

---

## `model.py`

### `DDSPFMModel`
FM resynthesis model with fixed frequency ratios.

**Components:** `FMDecoder` → `FMOperatorBank` (fixed ratios) + `FilteredNoise` (65 bands) → `Reverb` (learnable IR).

**`forward(f0_hz, loudness) → dict`**

Returns: `audio` (FM + noise + reverb), `dry_audio`, `operator_envelopes`, `fm_audio` (FM + reverb only, no noise). The `fm_audio` key is used when fine-tuning FM-specific behavior without the noise branch compensating.

**`synthesise(f0_hz, loudness) → (B, T)`** — no-grad inference.

### `DDSPFMModelFlex`
Inherits `DDSPFMModel` but promotes `fixed_ratios` from a buffer to an `nn.Parameter`. Initialized so ratios start at the violin STRINGS 1 values, using the softplus-inverse parameterization `r = softplus(raw) + 0.5` which keeps all ratios ≥ 0.5.

During fine-tuning on divergent targets, the ratios drift from integer values (e.g. 3.0 → 3.47), placing FM sidebands at non-harmonic positions and introducing genuine inharmonicity. The `forward` method passes the decoded ratios through to `FMOperatorBank` and also returns them as `ratios` for logging.
