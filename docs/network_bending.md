# `src/network_bending` — Network Bending and Model Blending for PLAUD

Activation-level and weight-level bending for PLAUD TorchScript models, plus linear weight interpolation (model blending) between compatible checkpoints. Accompanies the paper:

> Kotowski, B. & Font, F. (2026). *Network Bending as Circuit Bending Inspired Live Neural Synthesis Hacking*. NIME.

---

## Module overview

| File | Purpose |
|---|---|
| `activations.py` | Manual layer-by-layer decode with activation interception |
| `weights.py` | In-place weight mutations |
| `blending.py` | Linear weight interpolation between two models |
| `compatibility.py` | Parameter shape comparison between model pairs |

---

## TorchScript constraints

PLAUD models are TorchScript-exported ScriptModules. Two important constraints affect the implementation:

1. **`register_forward_hook` is not supported** on ScriptModules. Activation interception requires manually stepping through the decoder layer by layer in Python.

2. **Direct parameter assignment raises an error** (`leaf Variable requires grad in in-place op`). All weight mutations must use `param.data.copy_(new_value)` and must operate on a freshly loaded model copy to avoid mutating a shared reference.

---

## `activations.py`

### Decode layers

```python
DECODE_LAYERS = ["input_bottleneck", "gru", "inter_mlp", "output_params"]
```

The four interception points in PLAUD's decoder:

| Layer | What it processes | Bending effect |
|---|---|---|
| `input_bottleneck` | High-level latent representation | Structural / timbral |
| `gru` | Recurrent hidden state | Temporal / rhythmic |
| `inter_mlp` | Pre-output feature vector | Spectral shaping |
| `output_params` | Pre-synthesis logits | Strong spectral reshaping |

### `_scaled_sigmoid(x) → Tensor`
Matches PLAUD's internal `_scaled_sigmoid` function, decoded from the TorchScript inlined graph:
```
sigmoid(x)^ln(10) × 2 + 1e-18
```

### `_reset_hidden(model)`
Zeros the GRU's `_hidden_state` buffer under `torch.no_grad()`. Must be called before each decode to prevent state leaking between calls.

### `decode(model, z) → (1, 1, T·32)`
Full decoder forward pass equivalent to calling the model directly:
1. Reset hidden state.
2. `input_bottleneck(z)` → `gru.forward__0(x, h0)` → `inter_mlp(x)` → `output_params(x)`.
3. Apply `_scaled_sigmoid` and permute to `(1, C, T)`.
4. Call `model.pretrained._synthesize(synth_params)`.

Note: `gru.forward__0` is the TorchScript-generated method name for the GRU's forward pass with explicit hidden state.

### `bent_decode(model, z, layer, transform_fn) → (1, 1, T·32)`
Same as `decode` but applies `transform_fn` to the activation tensor immediately after `layer`. Only one layer is bent per call; to combine bending at multiple layers, call `bent_decode` in a chain.

### Transform factories

Ready-made `transform_fn` factories for use with `bent_decode`:

| Factory | Effect |
|---|---|
| `scale_fn(factor)` | Multiply activation by scalar |
| `shift_fn(offset)` | Add constant offset |
| `noise_fn(std, seed)` | Add Gaussian noise |
| `channel_shuffle_fn(seed)` | Randomly permute feature channels |
| `zero_fn()` | Zero the entire activation |
| `roll_fn(shifts, dim=-1)` | Cyclic shift along channel dimension |
| `flip_fn(dim=-1)` | Reverse channel order |

---

## `weights.py`

All operations mutate the model **in place** via `param.data.copy_()`. Always operate on a freshly loaded copy (`torch.jit.load(path).eval()`) so the original checkpoint on disk stays clean.

#### `_find_param(model, name) → Parameter`
Walks `model.named_parameters()` to find a parameter by exact name. Raises `KeyError` if not found.

#### `scale(model, param_name, factor)`
Multiplies all values in the named parameter tensor by `factor`. Values < 1 compress the weight distribution toward zero; values > 1 amplify it.

#### `shift(model, param_name, offset)`
Adds `offset` to every value. Non-zero offsets break the zero-mean assumption common in weight initialization schemes and can significantly alter the output bias.

#### `add_noise(model, param_name, std, seed=None)`
Adds zero-mean Gaussian noise with standard deviation `std`. `seed` makes the noise reproducible. Mild noise (std << weight std) produces subtle detuning; large noise destroys weight structure.

#### `zero_out(model, param_name)`
Zeros all values — effectively ablates that parameter (kills the layer's contribution). Useful for identifying which weights are load-bearing.

#### `roll(model, param_name, shifts, dim=0)`
Cyclically shifts weight values along `dim` by `shifts` positions. Displaces which neuron connects to which, causing timbral smearing and filtering-like spectral changes without changing the statistical distribution of the weights.

#### `flip(model, param_name, dim=0)`
Reverses the order of weight values along `dim`.

#### `squeeze(model, param_name, factor=0.5)`
Alias for `scale(factor)` with `factor < 1`. Compresses the weight distribution toward zero.

---

## `blending.py`

### `blend(model_a, model_b, alpha, base_path, fallback='a') → model`
Returns a new model whose weights are `α·W_A + (1−α)·W_B`:
- `alpha=1.0` → pure model A
- `alpha=0.0` → pure model B

A fresh copy is loaded from `base_path` (avoids mutating either source model). Compatible parameters (same name and shape in both models) are blended. Incompatible parameters fall back to either model A or B according to `fallback`.

**Buffer handling:** unlike `load_state_dict`, `named_parameters()` does not cover registered buffers. `blend` explicitly iterates `named_buffers()` and copies non-hidden-state buffers (notably the synthesis oscillator phases `pretrained.synths.0._phases`) using the same blending rule. Hidden state buffers (`_hidden_state`) are excluded because `decode()` resets them before each call. Without this, `alpha=0` would not produce output identical to the unmodified model B.

### `alpha_sweep(model_a, model_b, alphas, base_path, fallback='a') → list[(alpha, model)]`
Returns `[(alpha, blended_model), ...]` for each alpha in `alphas`. Useful for rendering an array of blended outputs to compare at different mix ratios.

---

## `compatibility.py`

### `param_shapes(model) → dict[str, tuple]`
Returns `{param_name: shape_tuple}` for all named parameters in the model.

### `pairwise_compatibility(models) → DataFrame`
Computes a pairwise compatibility matrix for a dict of models. Each row reports:
- `matching_params`: parameters with the same name **and** shape in both models
- `total_params`: max parameter count across the pair
- `fully_compatible`: True if all parameters match

Returns a pandas DataFrame indexed by `(model_a, model_b)`. Fully compatible pairs can be blended completely; partial pairs blend only the matching subset.

Known compatibility among the included PLAUD checkpoints:
- `iclc-guitar-loops`, `plaud-melody`, `tarta-lows-avg`: fully compatible (50/50)
- `plaud-drums`: partially compatible with the above (48/50 — `output_params` shape differs: 1065×512 vs 665×512)

### `mismatched_params(model_a, model_b) → list[(name, shape_a, shape_b)]`
Lists parameters whose shapes differ or are absent in one model. The shape is `None` if the parameter exists only in the other model.

### `compatible_keys(model_a, model_b) → list[str]`
Returns parameter names that are safe to blend — present in both models with identical shapes.
