# Active Divergence

A research toolkit for applying active divergence techniques to neural audio synthesis. The project spans the full pipeline from training a DDSP instrument model from raw audio, through inference-time and training-time manipulation of that model, to activation-level and weight-level bending of pre-trained black-box TorchScript synthesizers.

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

## Getting started

The archive `active-divergence-materials.tar.gz` contains all models and audio samples.
Unpack it into the root of a cloned copy of this repository:

```bash
tar -xzf active-divergence-materials.tar.gz
```

This populates `models/`, `samples/violin/`, `samples/trumpet/`, and `samples/esc50/`
in place. All notebooks and source code are already in the repository; the archive
supplies only the binary assets.

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
  02_inference_divergence.ipynb   Inference-time active divergence
  03_training_divergence.ipynb    Training-time divergence
  04_network_bending_plaud.ipynb  Network bending and model blending on PLAUD

models/
  ddsp_baseline_violin.pt         Trained DDSP checkpoint
  fm_violin.pt                    Trained FM-DDSP checkpoint
  ts/                             PLAUD TorchScript models

samples/
  violin/                         16 kHz URMP violin clips (.wav + _features.pt)
  trumpet/                        16 kHz URMP trumpet clips (.wav + _features.pt)

tmp/                              Divergent checkpoints written here during training
docs/                             Per-module documentation
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

Audio samples are included in the archive under `samples/violin/` (16 clips) and
`samples/trumpet/` (15 clips). Each clip is a 4-second 16 kHz WAV file with a companion
`_features.pt` containing the pre-extracted F0 and loudness features used by the
notebooks.

The inspiring-set section (§3.1 of notebook 03) requires ESC-50 animal vocalization
clips. Download the dataset from https://github.com/karolpiczak/ESC-50 and set the
`ESC50_DIR` path in the notebook setup cell.

---

## Documentation

Per-module documentation is in `docs/`:

- [docs/ddsp.md](docs/ddsp.md) — DDSP baseline model (features, synth, decoder, model, loss, dataset, train)
- [docs/fm_ddsp.md](docs/fm_ddsp.md) — Differentiable FM synthesis variant (DDX7-style)
- [docs/divergence.md](docs/divergence.md) — Inference-time active divergence utilities
- [docs/training_divergence.md](docs/training_divergence.md) — Training-time divergence (inspiring set, fine-tuning, loss hacking)
- [docs/plaud.md](docs/plaud.md) — PLAUD latent trajectory generators
- [docs/network_bending.md](docs/network_bending.md) — Network bending and model blending for PLAUD
