You are building teaching materials for a 2.5h class on Active Divergence. This is task 3 of 4: produce a notebook on training-time active divergence techniques using DDSP.

## Prerequisites (already done)
Task 1 produced a baseline DDSP implementation in `src/` with a trained checkpoint in `models/` (harmonic + filtered noise model trained on a single instrument from URMP). The conda env `active-divergence` exists. The URMP dataset is at `/mnt/mariadata/datasets/URMP/Dataset` (docs at `/mnt/mariadata/datasets/URMP/README_for_Dataset.tar.pdf`).

If anything from task 1 is missing or broken, inspect `src/` and `notebooks/01_ddsp_baseline.ipynb` first, and fix what you need to in `src/` to make the rest of this task work.

## Goal
Create `notebooks/03_training_divergence.ipynb` covering THREE training-time active divergence techniques. Unlike notebook 2 (which freezes weights), here we modify training itself — the model gets retrained or fine-tuned in deliberately distorted ways.

The first technique reproduces a known paper. The other two are quick CPU-trainable demonstrations students can actually run live during class.

## Topics

### 3.1 Inspiring set (animal vocalizations via deliberately limited synthesizer)
Reproduce the core idea of Hagiwara et al., "Modeling Animal Vocalizations through Synthesizers" (in `papers/`). Read it carefully first.
- Use **torchsynth** (NOT the DDSP harmonic+noise model here — torchsynth provides classical modular synth components). Install it into the conda env.
- Source animal vocalization audio. The core dataset that the authors mention is available at `/mnt/mariadata/datasets/ESC-50`. If you have to ship a tiny demo set, place it in `samples/animals/`. A few species — bird calls, marine mammal vocalizations, primate calls — whatever you can get reliably.
- Build a torchsynth-based model that is intentionally LIMITED (small parameter space, few oscillators, simple envelopes). The "inspiring set" idea: the synthesizer cannot faithfully reproduce the targets, so optimization produces creative approximations rather than copies.
- Optimize synth parameters per target sample (per Hagiwara et al.) using a spectral loss similar to DDSP's. Save resulting parameter sets and rendered audio.
- Notebook should: explain the inspiring-set concept, show target vs. reproduction spectrograms, play both, and let the user explore parameter perturbations of the fitted synth via `ipywidgets`.
- This part may use GPU if available for the optimization, but per-sample fitting on CPU should work for short clips. Make sure the notebook code path is CPU-runnable and exposes the training process.

### 3.2 Divergent fine-tuning
Take the trained baseline DDSP checkpoint from task 1 and continue training it on audio with a completely different timbre. The model gradually drifts from the original instrument's character toward something hybrid and progressively stranger.

Implementation requirements:
- The fine-tuning loop must be **implemented in the notebook itself** (not hidden in `src/`), so students can read it, modify it, and re-run cell-by-cell. Helper functions (data loading, single training step, checkpoint save/load) can live in `src/` and be imported, but the loop and its hyperparameters are notebook-side.
- Must run on **CPU** for a few epochs in reasonable time (a few minutes per epoch at most). Use small batch size, short audio chunks (~1–2 seconds), low frame count, and a small subset of the fine-tuning data. Document the chosen knobs.
- The student must be able to **set the fine-tuning audio directory** via a clearly marked variable / `ipywidgets` text field at the top of the section. Suggest a default path (e.g. a different instrument from URMP, or any folder of `.wav` files).
- After every N steps (configurable, small), save an intermediate checkpoint AND render the same fixed (loudness, f0) probe trajectory through the current model. Append the rendered audio to a list so the student can play "epoch 0", "epoch 1", "epoch 2"... in sequence and hear divergence accumulate.
- Plot loss curves and a strip of probe-output spectrograms across training steps so the divergence trajectory is visible, not just audible.
- Keep the original baseline checkpoint untouched on disk. Save divergent checkpoints to `models/ddsp_divergent_<source>_<target>_step<N>.pt`.
- For the proof of concept, you can use the EMF audio from `/mnt/mariadata/datasets/emf-noises-freesound`. If you need to ship a smaller demo set, place it in `samples/emf/`. The point is just to have something clearly different from the original instrument, so any non-URMP audio will do.

Pedagogical framing in markdown: explain that this is the "find a fertile maladaptation" mode of training-time active divergence — undertraining on a new domain rather than fully retraining gives you the most interesting in-between sonic territory.

### 3.3 Loss hacking (inharmonicity loss)
Keep training data the same as the baseline (same instrument from URMP), but add a strong **inharmonicity** loss term to the training objective. Track the original spectral reconstruction loss separately so students can see reconstruction quality degrading as the inharmonicity term pulls the model toward non-harmonic territory.

Implementation requirements:
- The inharmonicity loss must be **built into the original model/loss module in `src/`** (e.g. extend the loss class with an optional `inharmonicity_weight` argument, default 0). The notebook just imports it and toggles it on. Do not duplicate the loss in the notebook.
- Definition: a sensible inharmonicity loss for this DDSP model rewards the harmonic oscillator outputting harmonic distributions that *deviate* from the standard integer-multiple stack — e.g. penalize concentration of energy at integer multiples of f0, or maximize spread/entropy across non-integer frequency content. Pick one clean definition, document it in markdown with the math, and justify it briefly. Either operate on the predicted harmonic amplitudes/distribution directly (cheap, no FFT needed) or on the synthesized output spectrum. The former is faster on CPU; prefer it.
- The notebook starts from the task 1 checkpoint (so students don't wait for from-scratch training), enables the inharmonicity term with a configurable weight via an `ipywidgets` slider or a clearly marked variable, and trains for a small number of epochs on CPU.
- Track and plot **two curves**: original reconstruction loss and inharmonicity loss, on the same axes (twin y-axis if scales differ). The point is to make the trade-off visceral.
- Same probe-trajectory rendering as in 3.2: render the same fixed (loudness, f0) probe at intervals and let the student listen to the progression — reconstruction degrades, sound gets weird and bell-like / metallic / inharmonic.
- Save divergent checkpoints to `models/ddsp_inharmonic_w<weight>_step<N>.pt`. Do not overwrite the baseline.

## Implementation rules
- Add modules under `src/` as needed (e.g. `src/training_divergence/inspiring_set.py`, `src/training_divergence/finetune_utils.py`). The inharmonicity loss extends the existing loss code in `src/` rather than living in a new file if the existing layout permits.
- Notebooks 3.2 and 3.3 should both train on **CPU**, in the notebook, in a few minutes per run. Be conservative with batch size, audio length, and dataset subset.
- 3.1 (inspiring set) may use GPU for fitting if available; ensure the CPU code path also works.
- All audio playback uses `IPython.display.Audio` inline. All comparisons get spectrograms.
- Use `ipywidgets` for: directory picker (3.2), inharmonicity weight slider (3.3), and parameter exploration of the fitted synth (3.1).
- The notebook is for explanation and interactive demonstration. Heavy markdown explaining what each technique is, why it produces divergence, and how to listen for it.

## Project structure rules (strict)
- Only `src/` and `notebooks/` may be modified. Writes to `models/`, `samples/`, `training/` are allowed.
- Do not modify `papers/` or the original URMP dataset. The baseline checkpoint from task 1 must remain untouched on disk.

## Environment
- Use the `active-divergence` conda env. Add `torchsynth` and any other missing deps. Document what you added in the notebook's first cell.

## Reading (do this before coding)
In `papers/`:
- Broad et al., Active Divergence — for the conceptual framing of training-time techniques (especially "fine-tuning to a divergent target" and "loss modification").
- Engel et al., DDSP — for the loss formulation you'll be extending.
- Hagiwara et al., Modeling Animal Vocalizations through Synthesizers — for section 3.1.

## Workflow
- Be fully autonomous. No questions. Make decisions and document them in markdown.
- Verify the notebook runs end-to-end on CPU, restart kernel, run all. Confirm 3.2 and 3.3 actually finish a few epochs in a few minutes on CPU. If they don't, reduce data/model size until they do — didactic responsiveness matters more than training thoroughness here.
