You are building teaching materials for a 2.5h class on Active Divergence. This is task 2 of 4: produce a notebook demonstrating inference-time active divergence techniques on the DDSP model from task 1.

## Prerequisites (already done)
Task 1 has produced:
- A trained baseline DDSP model (harmonic + filtered noise + reverb, encoder/decoder mapping loudness + f0 to synthesis params) at `models/ddsp_baseline_<instrument>.pt`.
- Importable modules in `src/` for the model and synthesis chain.
- A curated audio subset in `samples/`.
- A conda environment named `active-divergence`.

If anything is missing or broken, inspect `src/` and the task 1 notebook (`notebooks/01_ddsp_baseline.ipynb`), then proceed. Do not retrain from scratch unless absolutely necessary.

## Goal
Create `notebooks/02_inference_divergence.ipynb` covering two families of inference-time active divergence techniques on the trained DDSP model. The model weights are FROZEN throughout — divergence comes from manipulating inputs and the synthesis chain, not retraining.

## Topics to cover (in this order)

### 2.1 Out-of-distribution parameter sampling / latent space exploration
Generate audio from (loudness, f0) trajectories sampled from distributions that differ from the training data. Demonstrate at minimum:
- Pitches far outside the instrument's natural range (sub-bass, ultra-high, including microtonal glides).
- Loudness envelopes the instrument can't physically produce (e.g. extremely fast amplitude modulation, square-wave-like envelopes, sustained extreme dynamics).
- Random walks and noise-driven trajectories vs. structured/periodic trajectories.
- If a Z/timbre latent exists, sample it from broader distributions (Gaussian with larger σ, uniform, interpolations between extracted z's).
For each, render audio, plot the spectrogram, and discuss what the model does with these unfamiliar inputs.

### 2.2 Synthesis chain hacking
Modify the synthesis chain at inference. The decoder outputs harmonic amplitudes, harmonic distribution, overall amplitude, and noise filter coefficients — intercept these and transform them before they hit the synth. Implement and demonstrate at minimum:
- **Base function manipulation**: experiment with:
  * waveshaping the base sinusoids (e.g. apply nonlinearities like tanh, abs, clipping, folding; change to another waveform, etc) before scaling by the harmonic amplitudes. Show how this affects outcome timbre.
  * frequency modulation of base sinusoids.
- **Amplitude rolling**: cyclically shift the per-harmonic amplitude vector along the harmonic axis (turning the Nth harmonic's amplitude into the (N+k)th's). Vary k. Show what happens to perceived pitch/timbre.
- **Component limiting**: zero out subsets of harmonics (only odd, only even, only first K, only above K, random masks). Same for noise bands. Compare audio output.
- **FIR-filter coefficient manipulation for the noise component**: mess with the noise-filter magnitude response — invert it, randomize it, low/high/notch-shape it manually, swap coefficients between frames, smooth or sharpen them.
For each manipulation, expose at least one slider/control with `ipywidgets` so the audience can play live.
- **FIR-reverb coefficients manipulation**: if the model includes a reverb component with FIR coefficients, apply similar manipulations to them (e.g. invert, randomize, low/high/notch-shape, swap/smooth/sharpen). Show how this affects the perceived space and timbre.

## Implementation rules
- Add divergence utilities to `src/` (e.g. `src/divergence/inference.py`) — the notebook imports them, doesn't redefine them.
- The notebook is the teaching artifact: heavy markdown, brief code cells, every audio result audible inline (`IPython.display.Audio`), every transformation visualized (spectrograms, harmonic-amplitude heatmaps).
- You might need to implement part of the code in the original DDSP model (e.g. to expose intermediate synthesis parameters) — if so, do it in `src/` and keep the notebook clean.
- Use `ipywidgets` aggressively — sliders, dropdowns, checkboxes. The class is interactive.
- All work is on CPU. Load the checkpoint with `map_location='cpu'`.
- Reference and briefly cite (in markdown) the relevant papers in `papers/`: Broad et al. on Active Divergence, Yee-King et al. on Network Bending DDSP. Read them before writing.

## Project structure rules (strict)
- Only modify `src/` and `notebooks/`. Reading from `models/`, `samples/`, `papers/` is fine. Do not modify them.
- Do not retrain. Do not modify model weights — this notebook is purely inference-time.

## Workflow
- Be fully autonomous. Do not ask questions. Make decisions and note them in markdown.
- Verify the notebook runs end-to-end on CPU (restart kernel, run all). Fix any breakage.
- Use the `active-divergence` conda environment. If extra packages are needed, add them to the env and document in the notebook.

Read `papers/` (Active Divergence by Broad et al., DDSP by Engel et al., Network Bending DDSP by Yee-King et al.) first. Then implement.
