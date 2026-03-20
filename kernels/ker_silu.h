#pragma once
#ifndef SILU_LUT_H_
#define SILU_LUT_H_

#include <stdint.h>
#include <stddef.h>

#include "lut_sigmoid.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

static inline void silu_lut_q15(
    const int8_t *__restrict x, size_t N,
    const int16_t *__restrict lut, int L,
    int32_t alpha, int32_t beta, int rshift,
    int8_t *__restrict y)
{
    if (!x || !y || !lut || L <= 0)
        return;
    const int idx_max = L - 1;
    for (size_t i = 0; i < N; ++i)
    {
        const int32_t xi = (int32_t)x[i];
        int32_t idx = (int32_t)(((int64_t)xi * (int64_t)alpha + (int64_t)beta) >> rshift);
        if (idx < 0)
            idx = 0;
        else if (idx > idx_max)
            idx = idx_max;
        const int32_t sig_q15 = (int32_t)lut[idx];
        const int32_t prod = xi * sig_q15;
        const int32_t rq = (prod + (1 << 14)) >> 15;
        y[i] = SATURATE_INT8(rq);
    }
}

#ifdef LUT_SIGMOID_SIZE
    static inline void silu_lut_q15_model(
        const int8_t *__restrict x, size_t N,
        int32_t alpha, int32_t beta, int rshift,
        int8_t *__restrict y)
    {
        extern const int16_t lut_sigmoid[LUT_SIGMOID_SIZE];
        silu_lut_q15(x, N, lut_sigmoid, (int)LUT_SIGMOID_SIZE, alpha, beta, rshift, y);
    }
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SILU_LUT_H_
