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

### `ops/tensor_transformation.py`
- Registers a custom CoreML MIL op (`index_copy`) via `coremltools`.
- Used when converting the model to CoreML format on Apple Silicon.

## RoPE (Rotary Position Embedding) Implementation

This fork **avoids complex number tensors** (no `torch.view_as_complex`) for CoreML compatibility. Instead, real and imaginary parts are tracked separately:

```python
# In precompute_freqs_cis():
freqs_cis_real, freqs_cis_imag = cos(freqs), sin(freqs)

# In apply_rotary_emb():
x_real, x_imag = chunk(x, 2, dim=-1)
rot_real = x_real * freqs_cis_real - x_imag * freqs_cis_imag
rot_imag = x_real * freqs_cis_imag + x_imag * freqs_cis_real
```

Gemma 3 uses **two separate RoPE tables**:
- `local_freqs_cis` — `theta=10_000` for `LOCAL_SLIDING` layers.
- `global_freqs_cis` — `theta=1_000_000` (with optional `rope_scaling_factor`) for `GLOBAL` layers.

## KV Cache Design

- Gemma 1/2 (`GemmaForCausalLM`): `GemmaKvCache` is stored as model buffers, initialised on each `generate()` call via `model.initialise_cache(batch_size, max_seq_len, device)`.
- Gemma 3 (`Gemma3ForMultimodalLM`): KV caches are created as local tensors per `generate()` call and passed through the forward call stack.

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
