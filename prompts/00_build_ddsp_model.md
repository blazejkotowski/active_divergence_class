You are building teaching materials for a 2.5h class on Active Divergence in generative audio models. This is task 1 of 4: train a baseline DDSP (Differentiable Digital Signal Processing) model that will be reused as the foundation for subsequent notebooks on inference-time active divergence techniques.

## Goal
Create a minimal, conventional DDSP implementation (harmonic oscillator + filtered noise) with encoder/decoder, trained on a small subset of a single instrument from the URMP dataset. The model must generate audio from only two control signals: frame-wise loudness and pitch (f0).

## Project structure (strict)
- `src/` — all Python modules go here. You may create subpackages (e.g. `src/ddsp/`).
- `papers/` — READ-ONLY. Contains reference PDFs including the original DDSP paper by Engel et al. Read it before implementing.
- `models/` — write trained checkpoints here.
- `samples/` — write a small curated audio subset here for notebook demos (CPU-friendly).
- `training/` — write training logs (TensorBoard or plain text) here.
- `notebooks/` — write the Jupyter notebook here as `01_ddsp_baseline.ipynb`.
- DO NOT touch anything outside `src/` and `notebooks/` except for writing checkpoints to `models/`, samples to `samples/`, and logs to `training/`. Do not modify `papers/`.

## Dataset
- Location: `/mnt/mariadata/datasets/URMP/Dataset`
- Documentation: `/mnt/mariadata/datasets/URMP/README_for_Dataset.tar.pdf`
- First, read the README PDF to understand the structure (multi-instrument recordings with separated stems, per-note F0 annotations, etc.).
- Pick ONE instrument with sufficient solo-stem material (violin is a safe default if you have no strong reason otherwise). Document your choice and why.
- Curate a SMALL subset (a few minutes total of audio is enough for didactic purposes). Copy the curated raw audio into `samples/` so notebooks have local access without touching the original dataset.

## Environment
- Use a dedicated conda environment named `active-divergence`. This SAME environment will be reused across all 4 tasks, so install a superset of dependencies you anticipate needing (PyTorch with CUDA, torchaudio, librosa, CREPE or torchcrepe for f0, numpy, scipy, matplotlib, ipywidgets, jupyter, tensorboard, soundfile, tqdm). Pin nothing aggressively; just make it work.
- If the env already exists, reuse it and add missing packages.
- Provide an `environment.yml` or shell snippet inside the notebook's first cell showing how to recreate it.
- Training uses GPU. Inference in notebooks must work on CPU — load checkpoints with `map_location='cpu'` in notebook code paths.

## Model spec (keep it minimal and conventional)
- Encoder: extract per-frame loudness (A-weighted, log-scale) and f0 (use torchcrepe or precomputed CREPE). No learned encoder of timbre is required for this baseline — DDSP autoencoder style with hand-crafted features is the canonical minimal version. If you do include a small Z encoder, keep it optional and off by default.
- Decoder: small MLP/GRU stack mapping (loudness, f0) → harmonic amplitudes (per-harmonic), overall amplitude, and noise-band magnitudes. Follow the original DDSP paper's modified-sigmoid output activation for non-negative controls.
- Synthesis:
  - Harmonic oscillator: additive sinusoidal synthesis with anti-aliasing (zero out harmonics above Nyquist).
  - Filtered noise: time-varying FIR filter applied to white noise via frequency-domain magnitude response (windowed IR), as in the DDSP paper.
  - Sum harmonic + noise. No reverb module needed for the baseline (keeps it minimal; out-of-scope additions are fine if trivial).
- Loss: multi-scale spectral (magnitude STFT at several FFT sizes, L1 + log-L1).
- Sample rate: 16 kHz is sufficient and keeps things fast. Frame rate: 250 Hz (hop 64 at 16 kHz) is standard.

## Training
- Train on GPU. Keep it short — a few thousand steps is enough for didactic results on one instrument. Save the best checkpoint to `models/ddsp_baseline_<instrument>.pt`.
- Log losses and a few audio reconstructions to `training/`.

## The notebook (`notebooks/01_ddsp_baseline.ipynb`)
This is a TEACHING notebook. It should be interactive and explanatory.
- Markdown intro: what DDSP is, why it's a great substrate for active divergence (interpretable controls + differentiable synthesis chain).
- Show the dataset choice and play a few samples (use `IPython.display.Audio`).
- Show feature extraction: plot loudness and f0 contours.
- Walk through the model architecture with code cells importing from `src/`. The notebook should not redefine the model — only import, instantiate, and demonstrate.
- A cell that loads the trained checkpoint (CPU) and reconstructs a held-out sample. Plot original vs. reconstructed spectrograms side by side and play both.
- An interactive cell using `ipywidgets`: sliders for a constant pitch (MIDI) and a constant loudness, plus a duration; render and play the result. This previews the parameter-control surface that notebook 2 will exploit.
- Final cell: a brief "what's next" pointing to inference-time active divergence.

## Rules
- Be fully autonomous. Do not ask questions. Make sensible decisions and document them in the notebook.
- Run until done. If a step fails, debug and continue.
- Code in `src/` should be minimal but fully functional and importable.
- Verify the notebook runs end-to-end on CPU after training (restart kernel, run all). Fix anything that breaks.
- Read the DDSP paper in `papers/` before implementing — match its conventions.
