from .inference import (
    load_model, n_frames,
    constant, pitch_glide, microtonal_glide, fast_am, square_gate,
    random_walk, periodic_pitch,
    get_controls,
    harmonic_oscillator_bent, WAVEFORM_NAMES, make_waveform_fn,
    mask_harmonics,
    bend_noise_mag, get_ir, bend_ir, apply_ir,
    synth_bent,
    specshow, play, heatmap,
)
