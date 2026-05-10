# `src/plaud` — PLAUD Latent Trajectory Generators

Utilities for generating latent input trajectories for PLAUD generative audio models. PLAUD models are exported as TorchScript (`.ts`) files; this module provides the latent inputs they consume.

Used alongside `src/network_bending` in `notebooks/04_network_bending_plaud.ipynb`.

---

## PLAUD architecture context

PLAUD's decoder expects a latent trajectory of shape `(1, T, 4)`:
- `T` = number of frames, at 1500 frames/second (SR=48000, HOP=32)
- `4` = latent dimensionality

The decoder path is:

```
z (1, T, 4)
  → input_bottleneck   (MLP)
  → gru                (2-layer, 512 hidden)
  → inter_mlp          (MLP)
  → output_params      (linear)
  → _scaled_sigmoid    (2 · sigmoid(x)^ln(10) + ε)
  → permute(0, 2, 1)
  → _synthesize        (DDSP-style synthesis)
```

The GRU is stateful (`streaming=True`), so `_hidden_state` must be zeroed before each synthesis call. See `src/network_bending/activations.py` for the manual decode path.

---

## `latents.py`

**Constants:**
```python
LATENT_DIM     = 4
HOP            = 32
SR             = 48000
FRAMES_PER_SEC = 1500.0
```

### `lfo_latent(duration_s, freqs, global_freq_mult, sr, hop) → (1, T, 4)`
Multi-channel LFO bank. Each of the 4 latent channels is an independent sine wave. Default frequencies: `[1.0, 2.0, 0.5, 0.3]` Hz. `global_freq_mult` scales all frequencies uniformly, providing a single slider to speed up or slow down the entire LFO bank.

Produces continuously moving latent trajectories that explore the model's learned space in a structured, periodic way. The different per-channel frequencies create slowly evolving polyrhythmic patterns.

### `constant_latent(duration_s, values, sr, hop) → (1, T, 4)`
Constant-vector latent trajectory. Each channel is fixed at its given value for the entire duration. Default: all zeros.

Useful as a baseline for isolating the effect of bending operations from the effect of latent dynamics — a constant latent should produce a time-invariant sound (modulo the synthesis oscillator's natural evolution), so any change in the output under bending is attributable to the bending alone.

### `random_walk_latent(duration_s, step_size, sr, hop, seed) → (1, T, 4)`
Random-walk latent trajectory. Each of the 4 channels performs an independent cumulative random walk with step standard deviation `step_size`, then the result is normalized to `[-1, 1]` by the global maximum absolute value. The `seed` parameter makes the trajectory reproducible.

Provides wide-ranging, unpredictable variation that stresses bending effects across a broad portion of the latent space, without the periodicity artifacts of LFO inputs.
