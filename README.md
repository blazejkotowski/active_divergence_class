# Active Divergence

A research toolkit for applying active divergence techniques to neural audio synthesis. The project spans the full pipeline from training a DDSP instrument model from raw audio, through inference-time and training-time manipulation of that model, to activation-level and weight-level bending of pre-trained black-box TorchScript synthesizers.

The work accompanies the paper:

> Kotowski, B. & Font, F. (2026). *Network Bending as Circuit Bending Inspired Live Neural Synthesis Hacking*. NIME.

---

## What is active divergence?

Active divergence is the deliberate introduction of controlled instability or misalignment into a generative model's synthesis process — at inference time, during fine-tuning, or by direct manipulation of weights and activations. The goal is not reconstruction fidelity but sonic exploration: sounds that are recognisably descended from a trained model yet drift into territories the model was never designed to reach.

This project treats active divergence as a spectrum of four interception strategies:

| Strategy | Where | Module |
|---|---|---|
| OOD inputs & synthesis interception | Frozen model, changed inputs/synth | `src/divergence` |
| Inspiring-set fine-tuning | Training dynamics, foreign objective | `src/training_divergence` |
| Loss hacking | Training dynamics, modified loss | `src/ddsp`, `src/training_divergence` |
| Network bending & blending | Model internals directly | `src/network_bending`, `src/plaud` |

---

## Repository layout

```
src/
  ddsp/                    Additive+noise DDSP model — baseline instrument synthesis
  fm_ddsp/                 DDX7-style differentiable FM synthesis variant
  divergence/              Inference-time interception utilities for the DDSP model
  training_divergence/     Fine-tuning and loss-hacking tools for training-time divergence
  plaud/                   Latent trajectory generators for PLAUD TorchScript models
  network_bending/         Activation and weight manipulation for PLAUD

notebooks/
  01_ddsp_baseline.ipynb          Baseline DDSP violin model
  02_inference_divergence.ipynb   Inference-time active divergence
  03_training_divergence.ipynb    Training-time divergence
  04_network_bending_plaud.ipynb  Network bending and model blending on PLAUD

models/
  ddsp_baseline_violin.pt         Trained DDSP checkpoint (best val_loss)
  ddsp_divergent_violin_*.pt      Divergent fine-tuning checkpoints
  ddsp_inharmonic_*.pt            Inharmonicity-loss checkpoints
  ts/                             PLAUD TorchScript models

samples/
  processed/                      Pre-extracted URMP violin clips (.pt)
  *.wav                           Raw 16 kHz copies

papers/                           Reference papers and bending log
training/                         TensorBoard logs
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate active-divergence
pip install pandas          # required by src/network_bending/compatibility.py
```

Requirements: Python 3.11, PyTorch 2.5.1, CUDA 12.1. All notebooks run on CPU; GPU is used only for URMP preprocessing (`src/ddsp/preprocess.py`).

---

## Running the notebooks

```bash
conda run -n active-divergence jupyter notebook
```

Or execute headlessly:

```bash
conda run -n active-divergence jupyter nbconvert \
    --to notebook --execute --inplace notebooks/01_ddsp_baseline.ipynb
```

## Data

The DDSP model trains on four URMP violin stems (movements 9, 17, 26, 44). Raw stems are expected at `/mnt/mariadata/datasets/URMP/Dataset`. Pre-processed clips are stored in `samples/processed/` after running `python -m src.ddsp.preprocess`. The inspiring-set experiments use ESC-50 animal vocalization clips.

---

## Documentation

Per-module documentation is in `docs/`:

- [docs/ddsp.md](docs/ddsp.md) — DDSP baseline model (features, synth, decoder, model, loss, dataset, train)
- [docs/fm_ddsp.md](docs/fm_ddsp.md) — Differentiable FM synthesis variant (DDX7-style)
- [docs/divergence.md](docs/divergence.md) — Inference-time active divergence utilities
- [docs/training_divergence.md](docs/training_divergence.md) — Training-time divergence (inspiring set, fine-tuning, loss hacking)
- [docs/plaud.md](docs/plaud.md) — PLAUD latent trajectory generators
- [docs/network_bending.md](docs/network_bending.md) — Network bending and model blending for PLAUD
