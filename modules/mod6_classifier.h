#pragma once
#ifndef MOD6_CLASSIFIER_H_
#define MOD6_CLASSIFIER_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ker_layernorm.h"
#include "ker_linear.h"

// INT8 → (LN) → INT32 logits (in-RAM weights)
static inline void classifier_forward_i8_to_i32(
    const int8_t *__restrict feat_i8,
    int D,
    float s_in_feat,
    const float *__restrict gamma,   // [D] or NULL (SRAM)
    const float *__restrict beta,    // [D] or NULL (SRAM)
    float ln_eps,
    float s_after_ln,
    const int8_t *__restrict W_i8,   // [C,D] row-major (SRAM)
    const int32_t *__restrict B_i32, // [C] (nullable)
    int32_t *__restrict Y32,         // [1,C] scratch
    int8_t *__restrict tmp_i8,       // [D]   scratch (LN output)
    int C,
    int32_t *__restrict logits_i32   // [C]
)
{
    // LN on the 1×D vector (treat as [C=D, T=1])
    layernorm_forward_f32_i8io(
        feat_i8, D, 1,
        s_in_feat,
        gamma, beta,
        ln_eps,
        s_after_ln,
        tmp_i8);

    // FC without requant: logits_i32 = W * x + b
    fc_core_outtile_one_row(tmp_i8, D, W_i8, C, B_i32, Y32);

    for (int n = 0; n < C; ++n) logits_i32[n] = Y32[n];
}

// INT8 → (LN) → INT8 logits (in-RAM weights; per-column requant)
static inline void classifier_forward_i8_to_i8(
    const int8_t *__restrict feat_i8,
    int D,
    float s_in_feat,
    const float *__restrict gamma,   // [D] or NULL (SRAM)
    const float *__restrict beta,    // [D] or NULL (SRAM)
    float ln_eps,
    float s_after_ln,
    const int8_t *__restrict W_i8,   // [C,D]
    const int32_t *__restrict B_i32, // [C] (nullable)
    const int32_t *__restrict Mv,    // [C]
    const int32_t *__restrict Rv,    // [C] (R>=0)
    int32_t *__restrict Y32,         // [1,C] scratch
    int8_t *__restrict tmp_i8,       // [D]   scratch
    int C,
    int8_t *__restrict logits_i8     // [C]
)
{
    layernorm_forward_f32_i8io(
        feat_i8, D, 1,
        s_in_feat,
        gamma, beta,
        ln_eps,
        s_after_ln,
        tmp_i8);

    // FC in int32 then per-channel requant to int8.
    fc_core_outtile_one_row(tmp_i8, D, W_i8, C, B_i32, Y32);
    fc_requant_outtile(Y32, Mv, Rv, C, D, logits_i8);
}

typedef struct {
    void* user;
    int (*read)(void* user, uint32_t off, void* dst, size_t nbytes);
} WeightReader;

static inline int classifier_forward_i8_streamed_fc_only(
    const int8_t* __restrict x_i8, // [D] (already LN'ed if you want LN)
    int D,
    WeightReader reader,
    uint32_t off_W,
    uint32_t off_B, // 0 -> no bias
    int C,
    int OC_BLK,
    int8_t*  __restrict w_tile, // [OC_BLK * D]
    int32_t* __restrict b_tile, // [OC_BLK]
    int32_t* __restrict logits_i32 // [C]
){
    if (!x_i8 || D<=0 || C<=0 || OC_BLK<=0 || !reader.read || !w_tile || !b_tile || !logits_i32)
        return -1;

    // Process classes in tiles
    for (int c0 = 0; c0 < C; c0 += OC_BLK) {
        const int cn = (c0 + OC_BLK <= C) ? OC_BLK : (C - c0);

        // Read bias tile or zero it
        if (off_B) {
            if (reader.read(reader.user, off_B + (uint32_t)(c0 * 4), b_tile, (size_t)cn * 4))
                return -2;
        } else {
            for (int j = 0; j < cn; ++j) b_tile[j] = 0;
        }

        // Read weight tile: rows [c0..c0+cn-1], each row has D int8
        const uint32_t w_off = off_W + (uint32_t)((size_t)c0 * (size_t)D);
        if (reader.read(reader.user, w_off, w_tile, (size_t)cn * (size_t)D))
            return -3;

        // Compute logits for this tile: logits_i32[c0: c0+cn) = W_tile * x_i8 + b_tile
        fc_core_outtile_one_row(x_i8, D, w_tile, cn, b_tile, &logits_i32[c0]);
    }
    return 0;
}

static inline int classifier_forward_i8_streamed(
    const int8_t* __restrict feat_i8, // [D]
    int D,
    // LayerNorm
    float s_in_feat,
    const float* __restrict gamma,  // [D] or NULL (SRAM)
    const float* __restrict beta,   // [D] or NULL (SRAM)
    float ln_eps,
    float s_after_ln,
    // FLASH reader + offsets
    WeightReader reader,
    uint32_t off_W,
    uint32_t off_B,
    // Geometry / tiling
    int C,
    int OC_BLK,
    // tiles
    int8_t*  __restrict w_tile, // [OC_BLK * D]
    int32_t* __restrict b_tile, // [OC_BLK]
    // scratch + out
    int8_t*  __restrict tmp_i8,     // [D]  (LN output)
    int32_t* __restrict logits_i32  // [C]
){
    if (!feat_i8 || D<=0 || C<=0 || !tmp_i8) return -10;

    // 1) LayerNorm: feat_i8 (T=1)
    layernorm_forward_f32_i8io(feat_i8, D, 1, s_in_feat, gamma, beta, ln_eps, s_after_ln, tmp_i8);

    // 2) Streamed FC: tmp_i8 × W^T + bias → logits_i32
    return classifier_forward_i8_streamed_fc_only(tmp_i8, D, reader, off_W, off_B, C, OC_BLK, w_tile, b_tile, logits_i32);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MOD6_CLASSIFIER_H_
