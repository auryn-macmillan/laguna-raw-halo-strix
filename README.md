# laguna-raw

A lean Rust + Vulkan compute engine for DFlash speculative decoding on AMD GPUs.

## Overview

This project implements the DFlash draft model architecture (used for speculative
decoding) from scratch in Rust, using Vulkan (Mesa RADV) compute shaders for
GPU acceleration. It is designed for AMD Ryzen AI 300 series (Strix Halo) APUs.

### Architecture

```
Target Model (Llama) → DFlash Encoder → DFlash Decoder → Draft Logits → Verify
                    (aux_norm + fc)   (6 transformer layers)
```

The DFlash model captures hidden states from specific target model layers,
normalizes and projects them through an encoder, then runs a lightweight
transformer decoder to generate draft tokens. These drafts are then verified
by the target model for accelerated generation.

## Requirements

- Vulkan 1.2+ ( Mesa RADV for AMD )
- `glslangValidator` (for shader compilation)
- Rust 1.70+

## Usage

```bash
# Show model metadata
cargo run --release -- info model.gguf

# Show tensor shapes
cargo run --release -- shapes model.gguf

# Run decoder forward pass (zero input)
cargo run --release -- forward model.gguf 0

# Run encoder + decoder chain (dummy zero features)
cargo run --release -- encode model.gguf

# GPU tests
cargo run --release -- test-dequant
cargo run --release -- test-matmul
cargo run --release -- test-rmsnorm
cargo run --release -- test-rope

# Unit tests
cargo test --lib
```

## Model Format

Supports GGUF v3 with the following tensor types:
- F32, BF16, F16 (full precision)
- Q8_0, Q4_K (quantized)

The model must be a DFlash architecture with:
- `enc.aux_norm.weight` - encoder per-feature normalization
- `fc.weight` - encoder projection
- `enc.output_norm.weight` - encoder output norm
- `output_norm.weight` - decoder final norm
- `blk.N.*` - decoder layer weights (6 layers)

## Compute Shaders

Shaders are in `kernels/*.comp`, compiled to SPIR-V via `build.rs`:

| Shader | Purpose |
|--------|---------|
| `rmsnorm.comp` | Global RMSNorm |
| `qk_norm.comp` | Per-head RMSNorm (Q/K normalization) |
| `enc_aux_norm.comp` | Encoder per-feature RMSNorm + aux_norm multiply |
| `rope.comp` | GPT-NeoX RoPE |
| `proj.comp` | Matrix-vector projection (GGUF column-major) |
| `repeat_v.comp` | GQA V repetition |
| `gate_mul.comp` | Element-wise gate multiplication |
| `attention.comp` | Q dot K score computation |
| `softmax.comp` | Softmax + V weighted sum |
| `activation.comp` | SiLU(gate) * up |
| `add.comp` | Residual addition |

## Project Structure

```
src/
  lib.rs       - Module definitions, CLI functions, include_spv! macro
  main.rs      - CLI entry point
  vulkan.rs    - Vulkan context, buffer, pipeline management
  gguf.rs      - GGUF v3 reader (metadata + tensor parsing)
  model.rs     - Model metadata abstraction
  dflash.rs    - DFlash model: encoder, decoder, dispatch functions
kernels/       - GLSL compute shaders
examples/      - Integration test examples
```

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Research & architecture | Done |
| 2 | GGUF reader, Vulkan infra, shaders | Done |
| 3 | Per-layer kernels (decoder forward) | Done |
| 4 | Chain DFlash encoder + decoder | Done |
| 5 | Megakernel fusion + DDTree stretch | Pending |
