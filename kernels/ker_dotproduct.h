#pragma once
#ifndef KER_DOTPRODUCT_H_
#define KER_DOTPRODUCT_H_

#include <stdint.h>
#include <stddef.h>

#include "ker_xpulp.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef LIKELY
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

// Weighted sum with Q15 probs along a row: sum_j p[j]*x[j] >> 15 → int8
static inline int8_t dot_q15_i8_to_i8(const int16_t* __restrict p_q15,
                                      const int8_t*  __restrict x_i8,
                                      int n)
{
    int32_t acc = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc += (int32_t)p_q15[i+0] * x_i8[i+0];
        acc += (int32_t)p_q15[i+1] * x_i8[i+1];
        acc += (int32_t)p_q15[i+2] * x_i8[i+2];
        acc += (int32_t)p_q15[i+3] * x_i8[i+3];
    }
    for (; i < n; ++i) acc += (int32_t)p_q15[i] * x_i8[i];
    // Q15 rounding shift (symmetric): preserves small non-zero means better than truncation.
    if (acc >= 0) acc += (1 << 14);
    else          acc -= (1 << 14);
    acc >>= 15;
    return SATURATE_INT8(acc);
}

// Batched weighted sums over T for all channels Z:[C,T]
static inline void attnpool_weighted_sum_ct_q15_to_i8(
    const int8_t*  __restrict Z, // [C, T] channel-major
    int C, int T,
    const int16_t* __restrict probs_q15, // [T]
    int8_t*       __restrict feat_i8     // [C]
)
{
    for (int c = 0; c < C; ++c) {
        const int8_t* zrow = &Z[(size_t)c * (size_t)T];
        feat_i8[c] = dot_q15_i8_to_i8(probs_q15, zrow, T);
    }
}

// With per-channel requant (single-step combined (M,R))
static inline void attnpool_weighted_sum_ct_q15_to_i8_requant(
    const int8_t*  __restrict Z, // [C, T]
    int C, int T,
    const int16_t* __restrict probs_q15, // [T]
    const int32_t* __restrict M,         // [C]
    const int32_t* __restrict R,         // [C]
    int8_t*        __restrict feat_i8    // [C]
)
{
    for (int c = 0; c < C; ++c) {
        const int8_t* zrow = &Z[(size_t)c * (size_t)T];
        int32_t acc = 0;
        for (int j = 0; j < T; ++j) acc += (int32_t)probs_q15[j] * (int32_t)zrow[j];
        int64_t prod = (int64_t)acc * (int64_t)M[c];
        int32_t rq;
        if (R[c] > 0)      rq = (int32_t)((prod + ((int64_t)1 << (R[c]-1))) >> R[c]);
        else if (R[c] < 0) rq = (int32_t)(prod << (-R[c]));
        else               rq = (int32_t)prod;
        feat_i8[c] = SATURATE_INT8(rq);
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // KER_DOTPRODUCT_H_
