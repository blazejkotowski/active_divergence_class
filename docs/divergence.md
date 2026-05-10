# `src/divergence` — Inference-Time Active Divergence

Utilities for applying active divergence to a frozen `DDSPAutoencoder` at inference time. All manipulation happens outside the model weights: either by supplying out-of-distribution input trajectories or by intercepting the synthesis chain between the decoder and the audio output.

The module is designed to be used interactively in `notebooks/02_inference_divergence.ipynb`.

---

## `inference.py`

### Model loading

#### `load_model(checkpoint_path, device) → DDSPAutoencoder`
Loads a trained checkpoint, restores model weights, freezes all parameters, and returns the model in eval mode.

---

### Input trajectory generators

All generators return `(1, N)` tensors in the shape expected by `model(f0, loudness)`.

#### `n_frames(duration_s, sr, hop) → int`
Converts a duration in seconds to a frame count given sample rate and hop length.

#### `constant(f0_hz, loudness_db, nf) → (f0, loudness)`
Flat pitch and loudness — the simplest test input.

#### `pitch_glide(f0_start, f0_end, nf, log=True) → f0`
Smooth pitch glide between two frequencies. `log=True` gives perceptually linear movement (equal semitone steps per frame).

#### `microtonal_glide(f0_center, semitone_range, nf, cycles) → f0`
Sinusoidal pitch oscillation around a center frequency, covering `±semitone_range/2` over `cycles` full cycles. Produces vibrato-like or slow pitch LFO effects at moderate semitone ranges, and microtonal beating effects at small ranges.

#### `fast_am(nf, rate_hz, depth_db, base_db, sr, hop) → loudness`
Sinusoidal amplitude modulation of the loudness control. At rates above ~8 Hz produces tremolo; at audio-rate-equivalent frame rates produces ring-modulation-like timbres.

#### `square_gate(nf, rate_hz, on_db, off_db, sr, hop) → loudness`
Binary on/off loudness gating with hard cuts. Produces staccato envelopes and rhythmic chopping effects.

#### `random_walk(nf, start, step_std, lo, hi) → trajectory`
Brownian motion trajectory clamped to `[lo, hi]`. Useful for unpredictable slowly-varying pitch or loudness.

#### `periodic_pitch(f0_center, nf, rate_hz, semitone_range, sr, hop) → f0`
Structured periodic pitch oscillation (wide vibrato or slow LFO sweep). Semitone ranges above 12 produce inter-octave sweeps.

---

### Decoder interception

#### `get_controls(model, f0_hz, loudness) → dict`
Runs the decoder only and returns the raw synth-parameter dictionary (`global_amp`, `harmonic_dist`, `noise_mag`) without synthesizing audio. Used for inspecting what the decoder produces before applying synthesis-level bending.

---

### Bent harmonic oscillator

The bent oscillator is a drop-in replacement for `HarmonicOscillator` with three additional bending axes.

#### `harmonic_oscillator_bent(f0_hz, amplitudes, global_amp, *, waveform_fn, fm_depth, fm_ratio, inharmonicity, h_n, sample_rate, hop_length) → (B, T)`

**Waveform bending** (`waveform_fn`): replaces `sin(phase)` with any differentiable function of phase. The factory `make_waveform_fn` provides named variants.

**FM synthesis** (`fm_depth`, `fm_ratio`): adds a sinusoidal phase modulator to each harmonic. The modulator frequency per harmonic is `f0 × fm_ratio`. Modulation depth is in radians — small values (0.5–2) add sidebands; large values (>5) produce chaotic spectra.

**Inharmonicity** (`inharmonicity`, `h_n`): stretches harmonic frequencies using the piano inharmonicity formula `f_k = k·f0·√(1 + B·k²)`. At small B (0.001–0.01) this adds subtle piano-like stiffness; at large B it produces bell-like or metallic spectra. The `h_n` parameter limits the stretch to the first `h_n` harmonics, leaving the upper partials harmonic.

#### `make_waveform_fn(name, amount) → callable`
Parametric waveform factory. The `amount` parameter controls distortion strength.

| Name | Effect | `amount` controls |
|---|---|---|
| `sin (original)` | No change | — |
| `tanh` | Soft saturation | pre-gain |
| `abs (rectify)` | Full-wave rectification | — |
| `hard clip` | Hard saturation | inverse threshold |
| `square` | Sign function | — |
| `triangle` | Triangle wave | — |
| `sawtooth` | Phase-modulo sawtooth | — |
| `wavefold` | Wavefolding | fold gain |

---

### Harmonic amplitude manipulation

#### `mask_harmonics(harmonic_dist, mode, k, rand_keep, n_harmonics) → Tensor`
Zeros out subsets of harmonics in the decoder's output distribution before synthesis. If `n_harmonics` is set, the mask applies only within the first `n_harmonics`; higher harmonics are unchanged.

| Mode | Effect |
|---|---|
| `all` | Passthrough |
| `odd` | Keep only odd harmonics (1, 3, 5, …) — square-wave-like |
| `even` | Keep only even harmonics — hollow, reedy tone |
| `first_k` | Keep only the lowest `k` harmonics — dark, low-harmonic timbre |
| `above_k` | Keep only harmonics above `k` — removes fundamental, hollow |
| `random` | Randomly keep each harmonic with probability `rand_keep` |

---

### Noise filter manipulation

#### `bend_noise_mag(noise_mag, mode, lo, hi, smooth_k) → Tensor`
Transforms noise-band magnitudes before `FilteredNoise` synthesis.

| Mode | Effect |
|---|---|
| `original` | Passthrough |
| `invert` | Spectral inversion (flip magnitude profile) |
| `randomise` | White noise (flat random magnitudes) |
| `low_pass` | Zero bands above `hi` |
| `high_pass` | Zero bands below `lo` |
| `notch` | Zero bands in `[lo, hi]` |
| `swap_frames` | Randomly permute frames (destroys temporal coherence) |
| `smooth` | Box-filter blur of the spectral envelope |
| `sharpen` | High-boost: `original + 3 × (original − smooth)` |

---

### Reverb IR manipulation

#### `get_ir(model) → Tensor`
Extracts the learned reverb IR from the model as a detached tensor of shape `(ir_length,)`.

#### `bend_ir(ir, mode, lo_hz, hi_hz, sr) → Tensor`
Manipulates a reverb impulse response before convolution.

| Mode | Effect |
|---|---|
| `original` | Passthrough |
| `invert` | Phase inversion |
| `reverse` | Time-reverse (reverse reverb effect) |
| `randomise_phase` | Keeps magnitude spectrum, randomises phase (different room) |
| `low_pass` | Keep only energy below `lo_hz` |
| `high_pass` | Keep only energy above `hi_hz` |
| `notch` | Remove energy in `[lo_hz, hi_hz]` |

#### `apply_ir(audio, ir) → Tensor`
Convolves `(B, T)` audio with a 1-D IR via frequency-domain multiplication.

---

### Full bent synthesis pipeline

#### `synth_bent(model, f0_hz, loudness, *, wavefold, fm_depth, fm_ratio, h_n, h_inharmonicity, h_mask, h_mask_k, harmonic_gain, noise_gain, ir_reverse) → dict`
End-to-end synthesis with all interception points active simultaneously. The decoder runs once (weights frozen); all divergence is structural.

Parameters:
- `wavefold`: wavefolding amount; 0 = pure sinusoidal
- `fm_depth` / `fm_ratio`: FM modulation index and carrier-to-modulator ratio
- `h_n`: harmonic scope for inharmonicity and masking (None = all 100)
- `h_inharmonicity`: piano inharmonicity B coefficient
- `h_mask` / `h_mask_k`: harmonic masking mode and parameter
- `harmonic_gain` / `noise_gain`: per-component gain before mixing
- `ir_reverse`: time-reverse the learned reverb IR

Returns: `audio`, `dry`, `harmonic_audio`, `noise_audio`, `harmonic_dist`, `noise_mag`, `global_amp`.

---

### Visualization helpers

#### `specshow(audio, sr, title, ax, fmax)`
Log-frequency spectrogram using librosa. Plots inline with `plt.show()`.

#### `play(audio, sr, normalize)`
Peak-normalizes and displays an inline `IPython.display.Audio` widget.

#### `heatmap(harmonic_dist, title)`
Per-frame harmonic amplitude distribution as a 2-D heatmap (frames × harmonics).
