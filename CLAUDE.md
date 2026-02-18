# CLAUDE.md — AI Assistant Guide for gemma_pytorch

This file provides essential context for AI assistants working on this codebase.

## Project Overview

This is a **fork** of Google's official PyTorch implementation of the **Gemma** family of large language models. It supports text-only and multimodal (text + vision) inference using PyTorch and PyTorch/XLA, targeting CPU, GPU, and TPU.

The fork includes exploratory work on:
- Running Gemma models on Apple Silicon via **CoreML** (`coremltools`)
- Removing complex-number operations from RoPE (rotary position embeddings) for CoreML compatibility
- Custom CoreML operator registration (e.g. `index_copy`)

## Supported Model Variants

| Variant   | Architecture | Type        | Vocab Size |
|-----------|-------------|-------------|------------|
| `2b`      | Gemma 1     | Text-only   | 256,000    |
| `7b`      | Gemma 1     | Text-only   | 256,000    |
| `2b-v2`   | Gemma 2     | Text-only   | 256,000    |
| `9b`      | Gemma 2     | Text-only   | 256,000    |
| `27b`     | Gemma 2     | Text-only   | 256,000    |
| `1b`      | Gemma 3     | Text-only   | 262,144    |
| `4b`      | Gemma 3     | Multimodal  | 262,144    |
| `12b`     | Gemma 3     | Multimodal  | 262,144    |
| `27b_v3`  | Gemma 3     | Multimodal  | 262,144    |

## Repository Structure

```
gemma_pytorch_FORK/
├── gemma/                        # Core model package
│   ├── __init__.py
│   ├── config.py                 # GemmaConfig dataclass + per-variant factory functions
│   ├── model.py                  # Gemma 1/2 text-only model (GemmaForCausalLM)
│   ├── gemma3_model.py           # Gemma 3 multimodal model (Gemma3ForMultimodalLM)
│   ├── gemma3_preprocessor.py    # Input preprocessing for Gemma 3 (text + image)
│   ├── model_xla.py              # PyTorch/XLA variant (tensor-parallel, Google-internal imports)
│   ├── tokenizer.py              # SentencePiece tokenizer wrapper
│   ├── xla_model_parallel.py     # XLA tensor-parallel primitives
│   └── siglip_vision/            # Vision encoder sub-package (for multimodal Gemma 3)
│       ├── __init__.py
│       ├── config.py             # SiglipVisionModelConfig
│       ├── siglip_vision_model.py # SiglipVisionModel (ViT-style encoder)
│       ├── preprocessor.py       # Image resizing/normalization for SigLIP
│       └── pan_and_scan.py       # Pan-and-scan image cropping
├── ops/                          # Custom operator experiments
│   ├── __init__.py
│   ├── tensor_transformation.py  # CoreML custom op: index_copy via coremltools MIL
│   ├── standalone_test_runner.py
│   └── tests/
│       ├── __init__.py
│       ├── test_tensor_transformation.py  # CoreML conversion tests (requires MPS device)
│       └── test_util.py
├── scripts/                      # Runnable inference scripts
│   ├── run.py                    # Text-only inference (Gemma 1/2/3 text variants)
│   ├── run_multimodal.py         # Multimodal inference (Gemma 3 4b/12b/27b_v3)
│   ├── run_xla.py                # XLA inference (CPU/TPU/GPU via PyTorch/XLA)
│   ├── run_coreml_trace.py       # CoreML model tracing (Apple Silicon)
│   ├── run_coreml_trace_layer.py # CoreML layer-level tracing
│   └── images/                   # Sample images for multimodal tests
├── docker/
│   ├── Dockerfile                # PyTorch CPU/GPU container
│   ├── xla.Dockerfile            # PyTorch/XLA CPU/TPU container
│   └── xla_gpu.Dockerfile        # PyTorch/XLA GPU container
├── tokenizer/
│   ├── tokenizer.model           # SentencePiece model (Gemma 1/2, vocab=256k)
│   └── gemma3_cleaned_262144_v2.spiece.model  # SentencePiece model (Gemma 3, vocab=262k)
├── environment.yaml              # Conda environment definition (Python ≤3.12)
├── requirements.txt              # Core pip dependencies
├── requirements-local.txt        # Local dev extras (kaggle, coremltools)
├── requirements-torch.txt        # PyTorch version pins
└── setup.py                      # Package installation (pip install -e .)
```

## Key Modules and Classes

### `gemma/config.py`
- `GemmaConfig` — central dataclass holding all model hyperparameters.
- `Architecture` enum: `GEMMA_1`, `GEMMA_2`, `GEMMA_3`.
- `AttentionType` enum: `GLOBAL`, `LOCAL_SLIDING`.
- Factory functions: `get_config_for_1b()`, `get_config_for_4b()`, etc.
- Entry point: `get_model_config(variant, dtype)` — use this to obtain a config.

### `gemma/model.py` — Gemma 1 & 2 (text-only)
Key classes (in dependency order):
- `Linear` / `Embedding` — quantization-aware wrappers around `nn.Linear` / `nn.Embedding`.
- `RMSNorm` — RMS layer normalization with unit offset (Gemma convention).
- `GemmaMLP` — gated MLP (GeGLU activation via `F.gelu(..., approximate="tanh")`).
- `GemmaKvCache` — KV cache stored as registered buffers per layer; initialised once per `generate()` call.
- `GemmaAttention` — multi-head attention with GQA (grouped-query attention), RoPE, optional QK-norm, optional sliding window and logit softcapping.
- `GemmaDecoderLayer` — Gemma 1 decoder block (pre-norm, post-attn residual, no extra norms).
- `Gemma2DecoderLayer` — Gemma 2/3 decoder block (pre/post feedforward norms optional).
- `GemmaModel` — stack of decoder layers, dispatches layer type based on `Architecture`.
- `GemmaForCausalLM` — top-level model: embedding, model stack, sampler, RoPE freq tables.
  - `generate(prompts, device, output_len, temperature, top_p, top_k)` — autoregressive generation.
  - `load_weights(model_path)` — loads from a single `.ckpt` or a sharded HuggingFace directory.

### `gemma/gemma3_model.py` — Gemma 3 multimodal
- `Gemma3ForMultimodalLM` — wraps text model + `SiglipVisionModel` + projection layers.
  - Accepts interleaved text + image inputs via `generate(prompts, device, ...)`.
  - Prompts are `Sequence[Sequence[str | PIL.Image.Image]]`.
  - Images are converted to patch tokens and injected into the hidden state.

### `gemma/tokenizer.py`
- `Tokenizer` — thin wrapper around `sentencepiece.SentencePieceProcessor`.
  - Special tokens: `bos_id`, `eos_id`, `pad_id`, `boi_id` (255999), `eoi_id` (256000).
  - `image_token_placeholder_id` = `pad_id` (images use pad tokens as placeholders in the sequence).

### `gemma/siglip_vision/`
- `SiglipVisionModel` — ViT-style encoder:
  1. Conv2D patch embedding (14×14 patches, 896×896 input → 64×64 patch grid).
  2. Positional embedding (learnable).
  3. 27 `SiglipEncoderBlock` layers (pre-LN, full attention, no sliding window).
  4. Final layer norm.
  5. `AveragePool2D` (4×4 → 256 tokens output per image).
- Images are normalized to mean=0.5, std=0.5, clipped to `[-1, 1]`.
- `pan_and_scan.py` — crops large images into at most 4 sub-crops for better resolution.

### `ops/` — CoreML custom operator package

The `ops/` package bridges PyTorch ops that have no CoreML equivalent, enabling the model to be converted and run on Apple Silicon via `coremltools`.

**`ops/__init__.py`**
Imports `index_copy` from `tensor_transformation`, which triggers `@register_torch_op` side-effect registration. Importing `ops` is sufficient to make all custom ops available before any CoreML conversion call.

**`ops/tensor_transformation.py`**
Registers `index_copy` as a custom CoreML MIL op using the `@register_torch_op` decorator from `coremltools.converters.mil.frontend.torch`. The op translates `torch.Tensor.index_copy_` (scatter-write along an arbitrary dimension) into CoreML's `mb.slice_update` primitive.

Needed because `GemmaKvCache.update()` calls `index_copy_` along the sequence dimension (dim=1) to write new K/V entries into the buffer cache — a pattern that CoreML does not support natively.

> **Status (in progress):** The `index_copy_` path is not yet fully exercised end-to-end. In `ops/tests/test_tensor_transformation.py`, the `TestTensorIndexCopySlice.forward()` currently performs a broadcast multiply instead of the real `index_copy_` call (which is commented out). The custom op translation is implemented but the full round-trip through CoreML conversion and execution is still being validated.

**`ops/tests/test_util.py`** — CoreML conversion pipeline helpers
Three utility functions used by all CoreML tests:

| Function | Purpose |
|---|---|
| `torch_export_model(model, inputs, dynamic_shapes)` | Exports the PyTorch model to ATen IR via `torch.export.export`, then runs `run_decompositions({})` to lower to a form CoreML can consume |
| `coreml_convert_model(aten_program, states=None)` | Converts to CoreML MLProgram format; targets **iOS 18** (`ct.target.iOS18`), uses `ct.ComputeUnit.ALL`, and accepts a `states` list of `ct.StateType` descriptors for persistent tensors |
| `run_coreml_model(model, inputs, state_kv=None)` | Runs inference; when a `state_kv=(name, numpy_array)` pair is supplied it calls `model.make_state()`, `state.write_state(name, value)`, then `model.predict(inputs, state=state)` — the standard CoreML State API |

The `states` / `ct.StateType` mechanism maps directly onto PyTorch `register_buffer`: a buffer declared in Python becomes a CoreML State tensor that persists across inference calls.

**`ops/standalone_test_runner.py`**
Runs `test_index_copy()` directly as a `__main__` script without pytest, so that full Python stack traces (rather than pytest's captured output) are visible during development.

**`ops/tests/test_tensor_transformation.py`**
End-to-end test of the buffer ↔ CoreML State round-trip:
1. Creates a model with a `register_buffer` target (mimicking `GemmaKvCache`).
2. Exports with `torch.export` using dynamic shapes for batch and index dimensions.
3. Converts to CoreML with `ct.StateType(ct.TensorType(shape=..., dtype=float16))` wrapping the buffer.
4. Runs CoreML inference via `run_coreml_model(..., state_kv=('target', ...))`.

Note: CoreML State requires `float32` for `write_state` even when the underlying dtype is `float16` (see inline comment `### Bah! use float32 not float16`). MPS device is required; tests fail on Linux/CUDA.

## RoPE (Rotary Position Embedding) Implementation

The upstream implementation uses `torch.polar` / `torch.view_as_complex` to treat the rotation as complex-number multiplication. CoreML has poor support for complex (imaginary-number) tensors, so **this fork rewrites RoPE entirely in real arithmetic** — no complex dtypes are used anywhere.

`precompute_freqs_cis()` returns a `(real, imag)` tuple of ordinary float tensors (cos and sin of the frequency angles). `apply_rotary_emb()` performs the rotation using the standard 2D rotation formula:

```python
# precompute_freqs_cis() — returns two real tensors instead of one complex tensor
freqs_cis_real = cos(freqs)   # was: torch.polar(...) → complex64
freqs_cis_imag = sin(freqs)

# apply_rotary_emb() — explicit rotation, no view_as_complex / view_as_real
x_real, x_imag = chunk(x, 2, dim=-1)
rot_real = x_real * freqs_cis_real - x_imag * freqs_cis_imag
rot_imag = x_real * freqs_cis_imag + x_imag * freqs_cis_real
```

The commented-out original code (`torch.polar`, `torch.view_as_complex`, `torch.view_as_real`) is preserved inline as a reference.

Gemma 3 uses **two separate RoPE tables**, each stored as a `(real, imag)` buffer pair:
- `local_freqs_cis` — `theta=10_000` for `LOCAL_SLIDING` layers.
- `global_freqs_cis` — `theta=1_000_000` (divided by `rope_scaling_factor` when set) for `GLOBAL` layers.

## KV Cache Design

CoreML (since coremltools ≥9.0) supports a **State** concept: tensors that are part of the model graph and persist across inference calls, analogous to PyTorch `register_buffer`. This fork maps the KV cache directly onto that abstraction.

### Gemma 1/2 — buffer-based (`GemmaKvCache`)

`GemmaKvCache` is an `nn.Module` that owns the cache as **registered buffers** (`register_buffer`), one `k_cache_i` / `v_cache_i` pair per layer. This makes the cache part of the model's state dict and moves with the model when `.to(device)` is called — the same contract as CoreML State.

Lifecycle:
1. `GemmaModel.__init__` creates a single `GemmaKvCache` instance and holds a reference to it.
2. Before each `generate()` call, `model.model.initialise_cache(batch_size, max_seq_len, device)` allocates the buffers (zeros, `float16`).
3. During the forward pass, `GemmaKvCache.update(layer_index, kv_write_indices, keys, values)` writes new K/V entries via `index_copy_` (dim=1, sequence dimension).
4. The `GemmaAttention` layer holds a reference to the shared `GemmaKvCache` and calls `update()` directly — there is no per-layer cache argument threaded through `forward()`.

### Gemma 3 — local tensors (`Gemma3ForMultimodalLM`)

Gemma 3's `generate()` allocates KV caches as plain `torch.zeros` tensors at the start of each call and threads them as a `List[Tuple[Tensor, Tensor]]` through the forward stack. These are **not** registered buffers. The design predates the buffer refactor and has not yet been migrated.

## Attention Variants

- **Gemma 1**: Global attention only, no softcapping, no QK norm.
- **Gemma 2**: Alternating `LOCAL_SLIDING` / `GLOBAL` (pattern repeats), logit softcapping (attn + final), pre/post FFW norms.
- **Gemma 3**: Same alternating pattern (5 local + 1 global cycling), QK normalization, dual RoPE theta, bidirectional attention over image token spans.

## Inference Scripts

### Text-only (`scripts/run.py`)
```bash
python scripts/run.py \
  --ckpt=/path/to/model.ckpt \
  --variant=1b \          # 2b | 7b | 2b-v2 | 9b | 27b | 1b
  --device=cpu \          # or cuda
  --output_len=100 \
  --prompt="What are large language models?"
```

### Multimodal (`scripts/run_multimodal.py`)
```bash
python scripts/run_multimodal.py \
  --ckpt=/path/to/model.ckpt \
  --variant=4b \          # 4b | 12b | 27b_v3
  --device=cpu \          # or cuda
  --output_len=100
```
Prompt format for multimodal: interleave `<start_of_turn>user\n`, `PIL.Image`, text, `<end_of_turn>\n<start_of_turn>model`.

### XLA (`scripts/run_xla.py`)
```bash
# CPU
PJRT_DEVICE=CPU python scripts/run_xla.py --ckpt=... --variant=...
# TPU
PJRT_DEVICE=TPU python scripts/run_xla.py --ckpt=... --variant=...
# GPU
USE_CUDA=1 PJRT_DEVICE=CUDA python scripts/run_xla.py --ckpt=... --variant=...
```

### CoreML tracing (`scripts/run_coreml_trace.py`)
Hardcoded to variant `1b`, `mps` device. Edit the script directly for other configs.

## Environment Setup

### Conda (recommended for local dev)
```bash
conda env create --file environment.yaml --yes
conda activate gemma-pytorch
pip install -r requirements.txt
pip install -r requirements-local.txt  # adds kaggle, coremltools
pip install torch  # see requirements-torch.txt for pinned versions
pip install -e .
```

### Docker (CPU/GPU)
```bash
docker build -f docker/Dockerfile . -t gemma:local
docker run --gpus all -v /path/to/ckpt:/tmp/ckpt gemma:local \
    python scripts/run_multimodal.py --ckpt=/tmp/ckpt --variant=4b --device=cuda
```

## Dependencies

| Package        | Version (requirements.txt) | Purpose                         |
|----------------|----------------------------|---------------------------------|
| `numpy`        | 2.2.3                      | Array operations                |
| `pillow`       | 11.1.0                     | Image loading/resizing          |
| `sentencepiece`| 0.2.0                      | Tokenization                    |
| `absl-py`      | 2.1.0                      | CLI flags (`absl.flags`)        |
| `torch`        | 2.6.0 or 2.8.0             | Core ML framework               |
| `coremltools`  | ≥9.0 (requirements-local)  | CoreML conversion (Apple only)  |

Python requirement: **≥3.11** (setup.py), **≤3.12** (environment.yaml).

## Code Conventions

- **Apache 2.0 license header** required on all new Python files.
- **Google-style docstrings** where present; many implementation files have minimal or no docstrings.
- **`dataclasses.dataclass`** for configuration objects (`GemmaConfig`, `SiglipVisionModelConfig`).
- **`absl.flags`** used for CLI argument parsing in all `scripts/` files.
- **`@torch.no_grad()`** / **`@torch.inference_mode`** decorators on `forward()` methods — this is inference-only code; no training support.
- **KV cache as buffers** (`register_buffer`) in `GemmaKvCache` for Gemma 1/2; as local tensors for Gemma 3.
- Commented-out code is common (e.g., original `index_select`/`view_as_complex` approaches preserved for reference).
- The `model_xla.py` file contains **Google-internal import paths** (`google3.third_party...`) and cannot be run outside Google's Piper monorepo.

## Tokenizer Notes

- **Gemma 1/2**: `tokenizer/tokenizer.model` — 256,000 vocab, 99 unused tokens (`<unused0>`–`<unused98>`, IDs 7–104).
- **Gemma 3**: `tokenizer/gemma3_cleaned_262144_v2.spiece.model` — 262,144 vocab.
- Image tokens use IDs 255999 (`<boi>`) and 256000 (`<eoi>`) as delimiters; image content is represented by repeating `pad_id` placeholder tokens (256 per image after pooling).

## Checkpoint Loading

`load_weights(model_path)` handles two formats:
1. **Single file**: `model_path` is a `.ckpt` file — loads `state_dict['model_state_dict']` with `strict=False`.
2. **Sharded HuggingFace format**: `model_path` is a directory containing `pytorch_model.bin.index.json` and shard files.

Always use `strict=False` when loading — there may be missing or extra keys (e.g. RoPE buffers not in the checkpoint).

## Common Pitfalls

- `GemmaDecoderLayer` (Gemma 1) still has the old `kv_cache` parameter in its `forward()` signature but is incompatible with the new `GemmaKvCache`-based approach used in `Gemma2DecoderLayer`. The `GemmaModel.forward()` routes around this by not passing `kv_cache` to Gemma 1 layers via `**kwargs`.
- `model_xla.py` cannot be used standalone — it requires Google-internal `google3` package paths.
- CoreML tests in `ops/tests/` require an Apple Silicon Mac with MPS device; they will fail on Linux/CUDA.
- The `run_coreml_trace.py` script has hardcoded paths (`./model/1b/...`) — update before running.
- `max_seq_len` must not exceed `config.max_position_embeddings` (checked via `assert` in `GemmaForCausalLM.generate()`).
