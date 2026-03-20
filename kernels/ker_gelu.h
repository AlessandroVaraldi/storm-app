#pragma once
#ifndef GELU_LUT_H_
#define GELU_LUT_H_

#include <stdint.h>
#include <stddef.h>

#include "lut_gelu.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

static inline void gelu_lut_q15(
    const int8_t *__restrict x,              // in  : int8, length N
    size_t N,
    const int16_t *__restrict lut,           // in  : int16 Q15, length L
    int L,
    int32_t alpha, int32_t beta, int rshift, // index mapping params
    int32_t M_out, int32_t R_out,            // requant Q15 -> int8
    int8_t *__restrict y                     // out : int8, length N
)
{
    if (!x || !y || !lut || L <= 0)
        return;
    const int idx_max = L - 1;
    const int64_t rnd_R = (R_out > 0) ? ((int64_t)1 << (R_out - 1)) : 0;
    for (size_t i = 0; i < N; ++i)
    {
        // 1) Integer-only LUT indexing
        const int32_t xi = (int32_t)x[i];
        int32_t idx = (int32_t)(((int64_t)xi * (int64_t)alpha + (int64_t)beta) >> rshift);
        if (idx < 0)
            idx = 0;
        else if (idx > idx_max)
            idx = idx_max;
        // 2) Fetch GELU(real_x) in Q15
        const int32_t gelu_q15 = (int32_t)lut[idx]; // can be negative
        // 3) Requant to int8: (gelu_q15 * M_out) >> R_out  (with rounding if R_out>0)
        int64_t prod = (int64_t)gelu_q15 * (int64_t)M_out;
        int32_t rq;
        if (R_out > 0)
            rq = (int32_t)((prod + rnd_R) >> R_out);
        else if (R_out < 0)
            rq = (int32_t)(prod << (-R_out));
        else
            rq = (int32_t)prod;
        y[i] = SATURATE_INT8(rq);
    }
}

// Convenience wrapper bound to the model LUT `lut_gelu` and its macros.
#ifdef LUT_GELU_SIZE
    static inline void gelu_lut_q15_model(
        const int8_t *__restrict x, size_t N,
        int32_t alpha, int32_t beta, int rshift, // index mapping (from s_in and LUT domain)
        int32_t M_out, int32_t R_out,            // requant from Q15 -> int8 at desired s_out
        int8_t *__restrict y)
    {
        extern const int16_t lut_gelu[LUT_GELU_SIZE];
        gelu_lut_q15(x, N, lut_gelu, (int)LUT_GELU_SIZE, alpha, beta, rshift, M_out, R_out, y);
    }
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GELU_LUT_H_
