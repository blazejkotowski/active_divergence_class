You are building teaching materials for a 2.5h class on Active Divergence in neural audio
synthesis. This is STEP 1 of 4: training a baseline DDSP model that subsequent notebooks
will hack and modify.

PROJECT STRUCTURE (you may ONLY touch `src/` and `notebooks/`):
- src/         → Python modules (DDSP model code goes here)
- papers/      → Reference PDFs (READ-ONLY). Relevant for this step:
                 - Engel et al., "DDSP: Differentiable Digital Signal Processing" (ICLR 2020)
- models/      → Save trained checkpoints HERE (write-only target)
- samples/     → Audio samples for demos (write-only target, you may save curated subsets)
- training/    → Training logs go HERE (write-only target)
- notebooks/   → Place the notebook for this step here as `00_baseline_ddsp.ipynb`

DATASET:
- Location: /mnt/mariadata/datasets/URMP/Dataset
- Documentation: /mnt/mariadata/datasets/URMP/README_for_Dataset.tar.pdf
- Task: Read the documentation, then select a SMALL subset (~50-200 samples) of a SINGLE
  instrument with clear pitched content (suggest violin, flute, or trumpet — pick whichever
  has the cleanest harmonic structure in the subset).

GOAL:
Implement a minimal, conventional DDSP autoencoder following Engel et al. (2020) Section 4.1:
- Encoder: extracts time-varying loudness (A-weighted, in dB) and pitch (f0, via CREPE or
  similar) from input audio. These are the ONLY two control parameters — keep it minimal,
  no learned z embedding for this baseline.
- Decoder: an MLP+GRU+MLP stack mapping (loudness, f0) → synthesizer parameters:
    * harmonic amplitudes (per-harmonic, normalized by a global amplitude)
    * filtered-noise FIR filter magnitudes
- Synthesizer: harmonic oscillator bank + filtered noise, summed. Must be fully
  differentiable and implemented in PyTorch.
- Loss: multi-scale spectral loss (L1 on magnitudes across multiple FFT sizes,
  e.g., [2048, 1024, 512, 256, 128, 64]).

RULES:
1. Use Python + Jupyter only. The notebook should be interactive — use `ipywidgets` for
   parameter controls (e.g., a slider to scrub through training samples and listen).
2. Training MUST use GPU (assume CUDA available). Save model in CPU-loadable form so
   downstream notebooks can run inference on CPU only.
3. Use `torch`, `torchaudio`, `librosa`, `crepe` (or `torchcrepe`), `numpy`, `matplotlib`,
   `ipywidgets`. Avoid heavy frameworks like the original `ddsp` TensorFlow library —
   reimplement minimally in PyTorch.
4. Code should be MINIMAL but FULLY FUNCTIONAL. Didactic clarity > cleverness.
   Every non-obvious line should have a short comment.
5. Put reusable code in `src/` as proper modules:
     - `src/ddsp/synth.py`     (harmonic + filtered noise synths)
     - `src/ddsp/model.py`     (encoder, decoder)
     - `src/ddsp/loss.py`      (multi-scale spectral loss)
     - `src/ddsp/features.py`  (loudness + pitch extraction)
     - `src/ddsp/dataset.py`   (dataset class for the chosen subset)
     - `src/ddsp/train.py`     (training loop)
   The notebook imports from these and orchestrates.

NOTEBOOK STRUCTURE (`notebooks/00_baseline_ddsp.ipynb`):
1. Brief markdown intro: what DDSP is, why we're building it (foreshadow active divergence).
2. Load + inspect the chosen instrument subset. Plot a sample's waveform, spectrogram,
   loudness curve, and f0 curve. Audio playback widget.
3. Show the model architecture (markdown diagram + summary).
4. Train (GPU). Display loss curve live or after.
5. Save checkpoint to `models/ddsp_baseline_<instrument>.pt`. Save 5-10 representative
   audio samples (input + reconstruction) to `samples/baseline/`.
6. Final cell: side-by-side audio comparison widget (original vs. reconstruction) for
   several test samples — this is what students hear in class.

DELIVERABLES:
- Modules in `src/ddsp/`
- `notebooks/00_baseline_ddsp.ipynb`, fully executed with outputs
- Trained checkpoint in `models/`
- Demo samples in `samples/baseline/`
- Training log in `training/`

Keep the model small enough to train in <30 min on a single modern GPU. The point is
didactic clarity, not SOTA quality.
