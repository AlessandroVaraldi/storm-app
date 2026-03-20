#pragma once

// build_config.h
// Single place to control optional (alternative) optimized paths.
// IMPORTANT: Defaults are set to preserve the current execution flow.

// ---------------------------
// Profiling / instrumentation
// ---------------------------
#ifndef ENABLE_PROFILER
#define ENABLE_PROFILER 1
#endif

#ifndef PROFILER_USE_GLOBAL
#define PROFILER_USE_GLOBAL 1
#endif

// ---------------------------
// External FLASH transfers
// ---------------------------
#ifndef FLASH_USE_QUAD
#define FLASH_USE_QUAD 1
#endif

// ---------------------------
// Preloading constants
// ---------------------------
#ifndef PRELOAD_CONSTANTS_EACH_INFERENCE
#define PRELOAD_CONSTANTS_EACH_INFERENCE 0
#endif

// ---------------------------
// Integer-only LayerNorm
// ---------------------------
#ifndef USE_INT_LAYERNORM
#define USE_INT_LAYERNORM 1
#endif

// ---------------------------
// Model data storage location
// ---------------------------
#ifndef USE_FLASH
#define USE_FLASH 0
#endif
