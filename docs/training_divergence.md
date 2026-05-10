# `src/training_divergence` — Training-Time Active Divergence

Utilities for two training-time active divergence strategies: inspiring-set fine-tuning and loss hacking. Both strategies modify the model's behavior by intervening during training rather than at inference, allowing the model's learned representation to shift rather than just its outputs.

Used in `notebooks/03_training_divergence.ipynb`.

---

## Module overview

| File | Purpose |
|---|---|
| `inspiring_set.py` | Modular synthesizer for animal vocalization approximation + per-clip optimisation |
| `finetune_utils.py` | Data loading, gradient steps, probe trajectories, and checkpoint saving |

---

## `inspiring_set.py`

Implements the inspiring-set technique from Hagiwara et al. (2022): a constrained modular synthesizer is optimized per target clip, producing an approximation that cannot faithfully reproduce the target. The gap between the model's current behavior and this deliberately imperfect approximation is the creative force.

### `VoiceSynth(sample_rate, duration)`
Pure-PyTorch re-implementation of the TorchSynth Voice architecture (~74 parameters). All parameters are stored as raw `nn.Parameter` tensors in unconstrained space; the synthesizer's forward pass applies appropriate sigmoid/tanh mappings to keep them in physically meaningful ranges.

**Architecture:**
- **Keyboard**: MIDI pitch (0–127) + note-on duration as fractions of clip length.
- **ADSR 1/2**: main envelopes (attack [0, 2s], decay [0, 2s], sustain [0, 1], release [0, 4s]).
- **LFO 1/2**: each with independent rate and amplitude ADSRs. Rate is modulated by its rate ADSR, ranging up to 40 Hz.
- **Modulation matrix**: 4 signal sources (ADSR 1/2, LFO 1/2) × 5 destinations (VCO1 pitch/amp, VCO2 pitch/amp, noise amp). Sigmoid-normalized to `[0, 1]`.
- **VCO 1 (Sine)**: sinusoidal oscillator with ±12-semitone detune and initial phase offset.
- **VCO 2 (Saw/Sine blend)**: shape parameter blends between sine (0) and sawtooth via truncated Fourier series (1). Also detuned.
- **Noise**: white noise with gain and spectral tilt.
- **Mixer**: softmax-weighted sum of VCO1, VCO2, noise.

**`forward() → (T,)`** — synthesizes one audio clip.

**`get_param_dict() → dict`** — returns human-readable decoded parameters: MIDI Hz, note-on duration, detunes, LFO rates, mixer weights.

### Internal helpers

#### `_adsr_envelope(attack, decay, sustain, release, note_on, T, sr) → (T,)`
Differentiable piecewise-linear ADSR. All segments are computed in parallel using masked linear interpolations, making the envelope fully differentiable with respect to the ADSR parameters.

#### `_saw_wave(phase, n_harmonics=8) → Tensor`
Differentiable sawtooth via truncated Fourier series: `Σ (-1)^(k+1) sin(k·φ) / k`.

#### `_lfo(rate_hz, T, sr) → (T,)`
Sinusoidal LFO at a given (possibly time-varying) rate.

---

### Per-clip optimisation

#### `fit_synth_to_target(target_audio, n_iter, lr, sample_rate, verbose) → (VoiceSynth, loss_history)`
Optimises `VoiceSynth` parameters to match `target_audio` using Adam and multi-scale STFT loss. Replicates Hagiwara et al. (2022) exactly: 200 iterations at lr=0.001.

Each call seeds the RNG from a hash of the first 200 samples of `target_audio`, giving each clip an independent, reproducible initialisation while avoiding identical local minima across clips.

Uses multi-scale STFT loss over FFT sizes `(2048, 1024, 512, 256)` with 25% hop.

#### `_spectral_loss(pred, target, fft_sizes) → scalar`
Multi-scale spectral distance (linear + log magnitude). Same formulation as `MultiScaleSpectralLoss` in `src/ddsp/loss.py`.

---

### ESC-50 loading

#### `load_esc50_animals(esc50_dir, categories, max_per_category, duration, sample_rate) → list[dict]`
Loads animal vocalization clips from the ESC-50 dataset. Filters by category, resamples to `sample_rate`, trims or zero-pads to `duration` seconds, and peak-normalizes. Returns a list of `{'audio': (T,), 'category': str, 'filename': str}` dicts.

Default categories: `{'cat', 'frog', 'chirping_birds', 'crickets'}`.

---

## `finetune_utils.py`

### Data loading

#### `load_audio_chunks(audio_dir, chunk_duration, sample_rate, max_chunks, fixed_f0_hz, hop_length) → list[dict]`
Loads WAV files from a directory, slices them into non-overlapping fixed-length chunks, and extracts F0 and A-weighted loudness for each. Returns a list of `{'audio': (T,), 'f0_hz': (N,), 'loudness': (N,)}` dicts.

`fixed_f0_hz` bypasses CREPE and substitutes a constant F0 value, which is necessary for unpitched material (e.g. EMF noise recordings) where pitch estimation produces meaningless results.

`max_chunks` limits total data for CPU-feasible training.

#### `extract_features_from_audio(audio, sample_rate, hop_length, fixed_f0_hz) → (f0_hz, loudness)`
Thin wrapper around `extract_loudness` and `extract_f0` for one-off feature extraction from a single clip.

---

### Probe trajectory

#### `make_probe_trajectory(f0_hz, loudness_db, duration, sample_rate, hop_length) → (f0, loudness)`
Returns a constant-pitch, constant-loudness probe trajectory as `(1, N_frames)` tensors. The same probe is rendered at each checkpoint to make model drift audible.

#### `make_probe_from_clip(processed_dir, clip_idx, max_frames) → (f0_probe, loud_probe, ref_audio)`
Loads a real pre-processed violin clip and uses its actual F0 and loudness trajectory as the probe. Unvoiced frames (F0 < 50 Hz) are filled with the mean voiced F0. The corresponding original audio is returned as a visual reference.

---

### Training step

#### `finetune_step(model, optimizer, loss_fn, batch, device, output_key) → float`
Single gradient step. Clips gradient norm to 1.0. The `output_key` parameter selects which key in the model's output dict is compared against the target audio — setting it to `'fm_audio'` trains only the FM branch, preventing the noise branch from absorbing the loss and masking FM-level changes.

---

### Checkpoint saving

#### `save_divergent_checkpoint(model, step, source_tag, target_tag, models_dir, config) → str`
Saves a checkpoint as `models/ddsp_divergent_{source_tag}_{target_tag}_step{N:04d}.pt`. The `source_tag` and `target_tag` encode the training history (e.g. `violin` and `emf`). Never overwrites the baseline checkpoint.

#### `save_inharmonic_checkpoint(model, step, weight, models_dir, config) → str`
Saves an inharmonicity-loss checkpoint as `models/ddsp_inharmonic_w{W}_step{N:04d}.pt`, where `W` is the inharmonicity loss weight formatted as `5p000` for `5.000`.
