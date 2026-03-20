# STORM

Embedded transformer inference engine in pure C for Human Activity Recognition (HAR) on resource-constrained microcontrollers.

## Overview

STORM-C implements a quantized Transformer (TinyTransformer) as a **header-only C library**, designed for edge inference on RISC-V platforms (X-HEEP). It classifies IMU sensor data (accelerometer + gyroscope) into 8 activity categories using INT8 quantized weights and activations.

## Architecture

```
Input [6ch × 64 samples]
  → ConvStem (1D pointwise conv + SiLU)
  → PosMix (depthwise residual conv)
  → Transformer Block ×2 (MHSA + MLP with GELU)
  → LayerNorm → AttnPool → Classifier
  → Logits [8 classes]
```

## Project Structure

```
storm-c/
├── main.c                  # Test harness & inference entry point
├── storm.h                 # Main wrapper (workspace, params, caches)
├── kernels/                # Low-level compute kernels
│   ├── ker_conv1d.h        # 1D convolution (pointwise & depthwise)
│   ├── ker_linear.h        # Matrix-vector operations (tiled, streaming)
│   ├── ker_layernorm.h     # Floating-point LayerNorm
│   ├── ker_layernorm_int.h # Integer-only LayerNorm with LUT
│   ├── ker_softmax.h       # Q15 softmax with LUT
│   ├── ker_gelu.h          # GELU activation (LUT-based)
│   ├── ker_silu.h          # SiLU activation (LUT-based)
│   ├── ker_dotproduct.h    # Dot product utilities
│   └── ker_xpulp.h         # RISC-V XPULP extension helpers
├── modules/                # Transformer block modules
│   ├── mod1_convstem.h     # Input stem
│   ├── mod2_posmix.h       # Positional mixing
│   ├── mod3_mhsa.h         # Multi-head self-attention
│   ├── mod4_mlp.h          # Feed-forward network
│   ├── mod5_attnpool.h     # Attention pooling
│   └── mod6_classifier.h   # Classification head
├── luts/                   # Lookup tables (exp, gelu, sigmoid, reciprocal)
└── utils/
    ├── build_config.h      # Compile-time feature toggles
    ├── model.h             # Auto-generated model weights & quantization params
    ├── model_flash.h       # Flash-optimized model data
    ├── flash_reader.h      # External SPI FLASH interface
    ├── profiler.h          # Cycle-accurate performance profiling
    └── test_vector.h       # Test input vectors
```

## Key Features

- **No malloc, no VLAs** — all buffers are statically allocated
- **INT8 symmetric quantization** with INT32 accumulators and per-channel weight scales
- **Two execution modes:** weights in RAM (`USE_FLASH=0`) or streamed from external SPI FLASH (`USE_FLASH=1`)
- **LUT-based activations** (GELU, SiLU, softmax) for integer-only compute
- **Built-in profiling** with per-module cycle counters
- **Optional XPULP extensions** for accelerated RISC-V integer operations

## Build Configuration

Edit `utils/build_config.h` to toggle features:

| Flag | Description |
|------|-------------|
| `USE_FLASH` | `0` = weights in RAM, `1` = read from SPI FLASH |
| `FLASH_USE_QUAD` | Enable quad-SPI reads for faster FLASH access |
| `USE_INT_LAYERNORM` | Use integer-only LayerNorm (no floating point) |
| `ENABLE_PROFILER` | Enable cycle counting instrumentation |
| `PRELOAD_CONSTANTS_EACH_INFERENCE` | Cache constants to avoid repeated FLASH reads |

## Building

The project is designed to be compiled as part of the X-HEEP SDK build system.

## Dependencies

- **X-HEEP SDK** — platform headers (`x-heep.h`, `csr.h`, `w25q128jw.h`)
- **RISC-V toolchain** with optional XPULP extension support
- **Standard C library** (`stdio.h`, `stdlib.h`, `string.h`, `math.h`)

## Input Format

6-channel IMU data (3-axis accelerometer + 3-axis gyroscope), 64 time steps. Raw sensor values are normalized per-channel and quantized to INT8 before inference.

## License

See repository for license information.
