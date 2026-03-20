#pragma once
#ifndef SOFTMAX_LUT_H_
#define SOFTMAX_LUT_H_

#include <stdint.h>
#include <stddef.h>

#include "lut_exp.h"
#include "lut_recip.h" 

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef SOFTMAX_USE_INTBINS
#define SOFTMAX_USE_INTBINS 1
#endif

#ifndef SM_SOFTMAX_INTBINS_RSHIFT
#define SM_SOFTMAX_INTBINS_RSHIFT 20
#endif

static inline float sm__clampf(float x, float lo, float hi)
{
    return (x < lo) ? lo : (x > hi ? hi : x);
}

static inline int sm__lut_index_from_x(float x)
{
    const float xmin = (float)LUT_EXP_XMIN;
    const float xmax = (float)LUT_EXP_XMAX;
    if (x <= xmin)
        return 0;
    if (x >= xmax)
        return (int)LUT_EXP_SIZE - 1;
    const float t = (x - xmin) * ((float)(LUT_EXP_SIZE - 1)) / (xmax - xmin);
    int idx = (int)(t + 0.5f);
    if (idx < 0)
        idx = 0;
    if (idx > (int)LUT_EXP_SIZE - 1)
        idx = (int)LUT_EXP_SIZE - 1;
    return idx;
}

static inline int sm__recip_lut_index_from_q15(int32_t mant_q15)
{
    // Center at 0.5 → subtract 0.5 in Q15 (1<<14)
    int32_t delta = mant_q15 - (1 << 14);
    if (delta < 0)
        delta = 0;
    // scale by (size-1)/0.5 = 2*(size-1); then / 2^15
    const int32_t scale = 2 * ((int32_t)LUT_RECIP_SIZE - 1);
    int32_t idx = (int32_t)(((int64_t)delta * (int64_t)scale + (1 << 14)) >> 15);
    if (idx < 0)
        idx = 0;
    if (idx > (int32_t)LUT_RECIP_SIZE - 1)
        idx = (int32_t)LUT_RECIP_SIZE - 1;
    return (int)idx;
}

// Compute floor(log2(x)) for x>0 (portable)
static inline int sm__flog2_u64(uint64_t x)
{
    int b = -1;
    while (x)
    {
        x >>= 1;
        ++b;
    }
    return b;
}

// Normalize sum_q15 to mantissa in [0.5,1.0) (Q15) and exponent k
static inline void sm__normalize_sum_q15(uint64_t sum_q15, int32_t *__restrict mant_q15_out, int *__restrict k_out)
{
    if (sum_q15 == 0)
    {
        *mant_q15_out = 0;
        *k_out = 0;
        return;
    }
    int b = sm__flog2_u64(sum_q15); 
    int k = b - 14;
    int32_t mant_q15;
    if (k >= 0)
        mant_q15 = (int32_t)(sum_q15 >> k);
    else
        mant_q15 = (int32_t)(sum_q15 << (-k));
    if (mant_q15 < (1 << 14))
        mant_q15 = (1 << 14);
    if (mant_q15 > (1 << 15) - 1)
        mant_q15 = (1 << 15) - 1;
    *mant_q15_out = mant_q15;
    *k_out = k;
}

static inline int16_t sm__apply_recip_norm_q15(uint16_t e_q15, int16_t inv_mant_q14, int k)
{
    int64_t prod = (int64_t)e_q15 * (int64_t)inv_mant_q14;
    int32_t q15 = (int32_t)((prod + (1LL << (LUT_RECIP_QBITS - 1))) >> LUT_RECIP_QBITS);
    if (k > 0)
    {
        int32_t add = 1 << (k - 1);
        q15 = (q15 + add) >> k;
    }
    else if (k < 0)
    {
        int shift = -k;
        if (shift >= 16)
        {
            q15 = 0x7FFF;
        }
        else
        {
            int32_t v = q15 << shift;
            if (v > 0x7FFF)
                v = 0x7FFF;
            q15 = v;
        }
    }
    if (q15 < 0)
        q15 = 0;
    if (q15 > 0x7FFF)
        q15 = 0x7FFF;
    return (int16_t)q15;
}

static inline void softmax_row_q15f(
    const int32_t *__restrict logits,
    size_t N,
    float scale,
    int16_t *__restrict out_q15)
{
    if (N == 0) return;

    int32_t m = logits[0];

    for (size_t i = 1; i < N; ++i)
        if (logits[i] > m)
            m = logits[i];
    uint64_t sum_q15 = 0;

    for (size_t i = 0; i < N; ++i)
    {
        float x = ((float)(logits[i] - m)) * scale;
        x = sm__clampf(x, (float)LUT_EXP_XMIN, (float)LUT_EXP_XMAX);
        const int idx = sm__lut_index_from_x(x);
        int16_t e_q15 = lut_exp[idx];
        if (e_q15 < 0)
            e_q15 = 0;
        out_q15[i] = e_q15;
        sum_q15 += (uint16_t)e_q15;
    }

    if (sum_q15 == 0)
    {
        const int16_t uni = (int16_t)((1u << 15) / (uint32_t)N);
        for (size_t i = 0; i < N; ++i)
            out_q15[i] = uni;
        return;
    }

    int32_t mant_q15;
    int k;

    sm__normalize_sum_q15(sum_q15, &mant_q15, &k);

    const int recip_idx = sm__recip_lut_index_from_q15(mant_q15);
    const int16_t inv_mant_q14 = lut_recip[recip_idx];

    for (size_t i = 0; i < N; ++i) out_q15[i] = sm__apply_recip_norm_q15((uint16_t)out_q15[i], inv_mant_q14, k);
}
static inline void softmax_row_q15_q31(
    const int32_t *__restrict logits,
    size_t N,
    int32_t mul,
    int rshift,
    int16_t *__restrict out_q15)
{
    if (N == 0)
        return;
    int32_t m = logits[0];
    for (size_t i = 1; i < N; ++i)
        if (logits[i] > m) m = logits[i];
    uint64_t sum_q15 = 0;
    for (size_t i = 0; i < N; ++i)
    {
        const int32_t d = logits[i] - m;
        const int64_t prod = (int64_t)d * (int64_t)mul;
        const int32_t xs = (rshift >= 0) ? (int32_t)(prod >> rshift) : (int32_t)(prod << (-rshift));
        float x = (float)xs;

        x = sm__clampf(x, (float)LUT_EXP_XMIN, (float)LUT_EXP_XMAX);

        const int idx = sm__lut_index_from_x(x);

        int16_t e_q15 = lut_exp[idx];

        if (e_q15 < 0) e_q15 = 0;

        out_q15[i] = e_q15;
        sum_q15 += (uint16_t)e_q15;
    }

    if (sum_q15 == 0)
    {
        const int16_t uni = (int16_t)((1u << 15) / (uint32_t)N);
        for (size_t i = 0; i < N; ++i) out_q15[i] = uni;
        return;
    }

    int32_t mant_q15;
    int k;

    sm__normalize_sum_q15(sum_q15, &mant_q15, &k);

    const int recip_idx = sm__recip_lut_index_from_q15(mant_q15);
    const int16_t inv_mant_q14 = lut_recip[recip_idx];

    for (size_t i = 0; i < N; ++i) out_q15[i] = sm__apply_recip_norm_q15((uint16_t)out_q15[i], inv_mant_q14, k);
}

static inline void softmax_row_q15_bins(
    const int32_t *__restrict logits,
    size_t N,
    int32_t mul_bins,
    int rshift,
    int16_t *__restrict out_q15)
{
    if (N == 0) return;

    const int xmin_bins = -((int)LUT_EXP_SIZE - 1);

    int32_t m = logits[0];

    for (size_t i = 1; i < N; ++i)
        if (logits[i] > m) m = logits[i];

    uint64_t sum_q15 = 0;

    for (size_t i = 0; i < N; ++i)
    {
        const int32_t d = logits[i] - m;
        const int64_t prod = (int64_t)d * (int64_t)mul_bins; 
        const int32_t x_bins = (rshift >= 0) ? (int32_t)(prod >> rshift) : (int32_t)(prod << (-rshift));
        int idx = x_bins - xmin_bins;
        if (idx < 0)
            idx = 0;
        if (idx > (int)LUT_EXP_SIZE - 1)
            idx = (int)LUT_EXP_SIZE - 1;
        int16_t e_q15 = lut_exp[idx];
        if (e_q15 < 0)
            e_q15 = 0;
        out_q15[i] = e_q15;
        sum_q15 += (uint16_t)e_q15;
    }
    if (sum_q15 == 0)
    {
        const int16_t uni = (int16_t)((1u << 15) / (uint32_t)N);
        for (size_t i = 0; i < N; ++i)
            out_q15[i] = uni;
        return;
    }
    int32_t mant_q15;
    int k;
    sm__normalize_sum_q15(sum_q15, &mant_q15, &k);
    const int recip_idx = sm__recip_lut_index_from_q15(mant_q15);
    const int16_t inv_mant_q14 = lut_recip[recip_idx];
    for (size_t i = 0; i < N; ++i)
    {
        out_q15[i] = sm__apply_recip_norm_q15((uint16_t)out_q15[i], inv_mant_q14, k);
    }
}

static inline void softmax_row_q15f_intbins(
    const int32_t *__restrict logits,
    size_t N,
    float scale,
    int16_t *__restrict out_q15)
{
    static int init = 0;
    static float cached_scale = 0.0f;
    static int32_t cached_mul_bins = 0;
    if (!init || !(scale == cached_scale)) {
        const float bpu = (float)((int)LUT_EXP_SIZE - 1) / (float)((float)LUT_EXP_XMAX - (float)LUT_EXP_XMIN);
        const float v = scale * bpu;
        const float s = v * (float)(1u << (unsigned)SM_SOFTMAX_INTBINS_RSHIFT);
        cached_mul_bins = (int32_t)(s >= 0.0f ? (s + 0.5f) : (s - 0.5f));
        cached_scale = scale;
        init = 1;
    }
    softmax_row_q15_bins(logits, N, cached_mul_bins, SM_SOFTMAX_INTBINS_RSHIFT, out_q15);
}

static inline void softmax_rows_q15f(
    const int32_t *__restrict logits, size_t R, size_t C,
    float scale,
    int16_t *__restrict out_q15)
{
    for (size_t r = 0; r < R; ++r) softmax_row_q15f(&logits[r * C], C, scale, &out_q15[r * C]);
}
static inline void softmax_rows_q15_bins(
    const int32_t *__restrict logits, size_t R, size_t C,
    int32_t mul_bins, int rshift,
    int16_t *__restrict out_q15)
{
    for (size_t r = 0; r < R; ++r) softmax_row_q15_bins(&logits[r * C], C, mul_bins, rshift, &out_q15[r * C]);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SOFTMAX_LUT_H_
