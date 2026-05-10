# `src/ddsp` — DDSP Baseline Model

A supervised DDSP autoencoder trained on URMP violin stems. The hand-crafted encoder extracts A-weighted loudness and CREPE F0; the learned decoder maps these to synthesizer controls; three synthesis modules (harmonic oscillator, filtered noise, learned reverb) reconstruct audio. The whole pipeline follows Engel et al. (2020) with a minimal, no-latent-space architecture.

---

## Module overview

| File | Purpose |
|---|---|
| `features.py` | A-weighted loudness extraction + CREPE F0 estimation |
| `synth.py` | Harmonic oscillator, filtered noise, and learned reverb |
| `decoder.py` | MLP+GRU decoder: (f0, loudness) → synth controls |
| `model.py` | Full autoencoder: wires encoder features → decoder → synths |
| `loss.py` | Multi-scale spectral loss + inharmonicity loss |
| `dataset.py` | PyTorch Dataset over pre-processed URMP clips |
| `preprocess.py` | Slice URMP stems into fixed-length clips, extract features |
| `train.py` | Training script: Adam + cosine LR schedule + TensorBoard |

---

## `features.py`

**Constants:** `SAMPLE_RATE = 16000`, `HOP_LENGTH = 64` (250 Hz frame rate), `A_WEIGHT_DB_REF = -20.0`.

### `_a_weighting_weights(fft_size, sr) → Tensor`
Computes the A-weighting frequency response for the given FFT size. Returns magnitude weights for each RFFT bin, normalized so the response at 1 kHz equals 1. Uses the IEC 61672 formula.

### `extract_loudness(audio, sr, hop_length, n_fft) → Tensor`
Computes per-frame A-weighted log loudness from a mono waveform `(T,)`. Returns `(N_frames,)` in dB, scaled to approximately `[-1, 1]` by subtracting and dividing by `A_WEIGHT_DB_REF`. Uses `torch.stft` with a Hann window.

### `extract_f0(audio, sr, hop_length, fmin, fmax, device, batch_size) → (f0_hz, voiced)`
Estimates per-frame fundamental frequency using torchcrepe. Returns `(f0_hz, voiced)` each of shape `(N_frames,)`. Unvoiced frames (periodicity ≤ 0.21) have their F0 zeroed. F0 range defaults to C1–C7 (32.7–2093 Hz).

---

## `synth.py`

### `HarmonicOscillator(n_harmonics, sample_rate, hop_length)`
Additive sinusoidal synthesizer (§3.2 of Engel et al. 2020). Takes frame-rate controls and interpolates them to audio rate before synthesis.

**`forward(f0_hz, amplitudes, global_amp) → (B, T)`**

Inputs:
- `f0_hz`: `(B, N_frames)` — fundamental frequency in Hz
- `amplitudes`: `(B, N_frames, N_harmonics)` — per-harmonic amplitude distribution
- `global_amp`: `(B, N_frames)` — overall amplitude envelope

Processing:
1. Upsample all frame-rate signals to audio rate via linear interpolation.
2. Zero harmonics above Nyquist (anti-aliasing).
3. Integrate instantaneous frequency to cumulative phase via Euler integration.
4. Sum `global_amp × amplitude_k × sin(phase_k)` over all harmonics.

### `FilteredNoise(n_bands, hop_length)`
Time-varying FIR noise filter (§3.4–3.5). Each frame has its own magnitude spectrum; an overlap-add convolution applies these per-frame IRs to white noise.

**`forward(noise_magnitudes) → (B, T)`**

Input: `(B, N_frames, N_bands)` — frequency-domain magnitudes (output of the decoder noise head).

Processing:
1. Generate white noise of length `T = N × hop_length`.
2. Per frame: inverse-FFT the magnitude spectrum to get a zero-phase IR, apply a Hann window, shift to causal.
3. Frequency-domain convolution of each noise frame with its IR.
4. Overlap-add to produce the full waveform.

### `Reverb(ir_length, sample_rate)`
Fixed-IR reverb via frequency-domain convolution. The IR is a single `nn.Parameter` shared across the batch (appropriate because all training examples share the same recording environment). Initialized as a decaying exponential to approximate room acoustics from step 0.

**`forward(audio) → (B, T)`** — Convolves `(B, T)` audio with the learned IR.

---

## `decoder.py`

### `modified_sigmoid(x)`
Non-negative output activation: `2 · sigmoid(x)^2.3 + 1e-7`. Ensures strictly positive synth parameters while allowing near-zero values.

### `DDSPDecoder(n_harmonics, n_noise_bands, hidden_size, n_mlp_layers, gru_layers)`
Maps per-frame `(f0_hz, loudness)` pairs to synthesizer control parameters.

**Architecture:** input MLP → single-layer GRU → output MLP → three parallel output heads.

- Input: log-normalized F0 and loudness concatenated to `(B, N, 2)`.
- Input MLP: 3 × (Linear → LayerNorm → ReLU), projecting 2 → hidden_size.
- GRU: provides temporal context across frames.
- Output MLP: mirrors input MLP.
- `head_amp`: Linear → `modified_sigmoid` → global amplitude `(B, N)`.
- `head_harmonics`: Linear → softmax → harmonic distribution `(B, N, K)` summing to 1.
- `head_noise`: Linear → `modified_sigmoid` → noise magnitudes `(B, N, M)`.

**`forward(f0_hz, loudness) → dict`** returns `global_amp`, `harmonic_dist`, `noise_mag`.

---

## `model.py`

### `DDSPAutoencoder(n_harmonics, n_noise_bands, hidden_size, n_mlp_layers, sample_rate, hop_length, reverb_ir_length)`
Full supervised DDSP autoencoder. No Z encoder; loudness and F0 are extracted externally and passed in directly.

**`forward(f0_hz, loudness) → dict`**

Returns: `audio` (reverberant), `dry_audio`, `harmonic_audio`, `noise_audio`, plus all decoder controls (`global_amp`, `harmonic_dist`, `noise_mag`).

**`synthesise(f0_hz, loudness) → (B, T)`** — no-grad inference convenience method.

Default configuration: `n_harmonics=100`, `n_noise_bands=65`, `hidden_size=512`, `reverb_ir_length=64000`.

---

## `loss.py`

### `MultiScaleSpectralLoss(fft_sizes, alpha, overlap)`
Multi-scale spectral loss (Engel et al. §4.2.1):

```
L = Σ_i  ||S_i − Ŝ_i||₁  +  α · ||log S_i − log Ŝ_i||₁
```

summed over six FFT sizes `(2048, 1024, 512, 256, 128, 64)` with 75% overlap. The log term penalizes relative spectral error and is critical for matching quiet partials.

**`forward(audio, target) → scalar`**

### `InharmonicityLoss(eps)`
Penalizes concentrated harmonic distributions to push the model toward inharmonic or spectrally spread timbres. Operates on the decoder's softmax harmonic distribution directly — no FFT required.

**Definition:** Shannon entropy of the harmonic distribution averaged over batch and frames. Negated so minimizing the loss maximizes entropy (spreads energy across harmonics).

```
L_inh = −mean_{B,N} [ Σ_k  p_k · log(p_k + ε) ]
```

Use as: `total_loss = spectral_loss + λ × inh_loss`.

**`forward(harmonic_dist) → scalar`** — input shape `(B, N_frames, K)`.

---

## `dataset.py`

### `URMPViolinDataset(data_dir)`
PyTorch Dataset over pre-processed URMP clips stored as `.pt` files. Each file contains `audio (T,)`, `f0_hz (N_frames,)`, `loudness (N_frames,)`, and `filename`.

**`collate_fn(batch)`** — static method; stacks audio, f0, and loudness into batch tensors for use with `DataLoader`.

---

## `preprocess.py`

Slices four URMP violin stems (movements 09, 17, 26, 44) into non-overlapping 4-second clips at 16 kHz, extracts F0 and loudness for each, and saves them as `.pt` files in `samples/processed/`. Clips with fewer than 25% voiced frames are discarded.

**`main()`** — entry point. Run as `python -m src.ddsp.preprocess`. Expects URMP stems at `/mnt/mariadata/datasets/URMP/Dataset`.

---

## `train.py`

Training script for the DDSP baseline.

**`main()`** — Adam optimizer with cosine LR annealing (`T_max=steps`, `eta_min=1e-5`). Validates every 500 steps; saves the best checkpoint to `models/ddsp_baseline_violin.pt`. Logs scalars and audio reconstructions to TensorBoard.

```
python -m src.ddsp.train [--steps 5000] [--batch-size 8] [--lr 3e-4]
```

Trained configuration: 5000 steps, batch size 8, lr=3e-4. Best val_loss ≈ 7.65 on 149 clips (135 train / 14 val).
