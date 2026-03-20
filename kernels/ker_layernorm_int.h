#pragma once
#ifndef KER_LAYERNORM_INT_H_
#define KER_LAYERNORM_INT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *lut;
    int          size; 
    float        log_min;
    float        idx_scale;
} kerln_int_rsqrt_lut_t;

static kerln_int_rsqrt_lut_t g_kerln_rsqrt_lut = { 0, 0, 0.0f, 0.0f };

static inline void kerln_int_lut_init(
    const float *lut,
    int          size,
    float        log_min,
    float        log_range)
{
    g_kerln_rsqrt_lut.lut       = lut;
    g_kerln_rsqrt_lut.size      = size;
    g_kerln_rsqrt_lut.log_min   = log_min;
    g_kerln_rsqrt_lut.idx_scale = (size > 1) ? (float)(size - 1) / log_range : 0.0f;
}

static inline float kerln_fast_lnf(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    /* log2(x) ≈ (float_bits_as_int32 − 0x3F800000) / 2^23 */
    float log2_approx = (float)((int32_t)v.u - (int32_t)0x3F800000) * (1.0f / 8388608.0f);
    return log2_approx * 0.6931471805599453f;
}

static inline float kerln_lut_rsqrt(float var_plus_eps)
{
    const kerln_int_rsqrt_lut_t *s = &g_kerln_rsqrt_lut;
    float ln_val = kerln_fast_lnf(var_plus_eps);
    int   idx    = (int)((ln_val - s->log_min) * s->idx_scale);

    if (idx < 0)         idx = 0;
    if (idx >= s->size)  idx = s->size - 1;

    return s->lut[idx];
}

#ifndef KER_LN_INT_MAX_C
  #if defined(MODEL_D_MODEL)
    #define KER_LN_INT_MAX_C MODEL_D_MODEL
  #else
    #define KER_LN_INT_MAX_C 128
  #endif
#endif

#define KER_LN_INT_FRAC   14
#define KER_LN_INT_SCALE  (1 << KER_LN_INT_FRAC)               /* 16384       */
#define KER_LN_INT_SCALEF ((float)KER_LN_INT_SCALE)            /* 16384.0f    */
#define KER_LN_INT_TOTAL  (KER_LN_INT_FRAC + KER_LN_INT_FRAC)  /* 28      */
#define KER_LN_INT_ROUND  (1LL << (KER_LN_INT_TOTAL - 1))      /* rounding    */

static inline void kerln_int_precompute_coeffs(
    const float *__restrict gamma,
    const float *__restrict beta,
    int          C,
    float        inv_sy,
    int32_t     *__restrict out_gq,
    int32_t     *__restrict out_bq)
{
    for (int c = 0; c < C; ++c) {
        float g = gamma ? gamma[c] : 1.0f;
        float b = beta  ? beta[c]  : 0.0f;
        out_gq[c] = (int32_t)lrintf(g * KER_LN_INT_SCALEF);
        out_bq[c] = (int32_t)lrintf(b * inv_sy * KER_LN_INT_SCALEF);
    }
}

static inline void kerln_int_affine_token(
    const int8_t  *__restrict x_data,
    int            x_stride,
    int            C,
    int32_t        K_q14,
    int32_t        Km_q14,
    const int32_t *__restrict gq14,
    const int32_t *__restrict bq14,
    int8_t        *__restrict y_data,
    int            y_stride)
{
    for (int c = 0; c < C; ++c) {
        int32_t centered = K_q14 * (int32_t)x_data[c * x_stride] - Km_q14;
        int64_t val = (int64_t)gq14[c] * (int64_t)centered + ((int64_t)bq14[c] << KER_LN_INT_FRAC);

        int32_t y_raw = (int32_t)((val + KER_LN_INT_ROUND) >> KER_LN_INT_TOTAL);

        if (y_raw >  127) y_raw =  127;
        if (y_raw < -128) y_raw = -128;
        y_data[c * y_stride] = (int8_t)y_raw;
    }
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KER_LAYERNORM_INT_H_ */
