#pragma once
#ifndef LAYERNORM_H_
#define LAYERNORM_H_

#include <stdint.h>
#include <stddef.h>
#include <math.h>   // lrintf

#ifdef __cplusplus
extern "C" {
#endif

#ifndef KER_LN_USE_XBUF64
#define KER_LN_USE_XBUF64 1
#endif

#define KER_LN_FWD_ATTR __attribute__((noinline))
#define KER_LN_FWD_STATIC static

static inline int8_t ln__saturate_i8(int32_t v) {
    if (v > 127)  return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}


static inline float kerln_rsqrt_fast(float x) {
    union { float f; uint32_t i; } u = { x };
    u.i = 0x5f3759dfu - (u.i >> 1);
    float y = u.f;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}

#ifndef rsqrtf
#define rsqrtf(z) kerln_rsqrt_fast((z))
#endif
#ifndef sqrtf
#define sqrtf(z) (1.0f / kerln_rsqrt_fast((z)))
#endif

#if defined(USE_INT_LAYERNORM) && USE_INT_LAYERNORM \
 && defined(LN_INT_ENABLED) && LN_INT_ENABLED
#include "ker_layernorm_int.h"
#undef rsqrtf
#define rsqrtf(z) kerln_lut_rsqrt((z))
#undef sqrtf
#define sqrtf(z) (1.0f / kerln_lut_rsqrt((z)))
#endif

static inline void ln__mean_var_column_i8_accum(
    const int8_t *__restrict x_col,
    int C, int T, float scale_x,
    float *out_mean, float *out_var,
    int8_t *__restrict xbuf_or_null)
{
    int32_t sum_i32 = 0;
    int32_t sumsq_i32 = 0;

    const int8_t *__restrict xp = x_col;

    for (int c = 0; c < C; ++c) {
        const int8_t xv = *xp;
        if (xbuf_or_null) xbuf_or_null[c] = xv;

        sum_i32 += (int32_t)xv;
        sumsq_i32 += (int32_t)xv * (int32_t)xv;
        xp += T;
    }

    if (C > 0) {
        const float invC = 1.0f / (float)C;
        const float mean_i = (float)sum_i32 * invC;
        float var_i = (float)sumsq_i32 * invC - mean_i * mean_i;
        if (var_i < 0.0f) var_i = 0.0f;

        const float sx = scale_x;
        const float mean = sx * mean_i;
        float var = (sx * sx) * var_i;
        if (var < 0.0f) var = 0.0f;
        *out_mean = mean;
        *out_var  = var;
    } else {
        *out_mean = 0.0f;
        *out_var  = 0.0f;
    }
}


static inline void layernorm_token_f32_i8o(
    const int8_t *__restrict x,
    int C,
    float scale_x,
    const float *__restrict gamma,
    const float *__restrict beta,
    float eps,
    float scale_y,
    int8_t *__restrict y
){
    int32_t sum_i32 = 0;
    int32_t sumsq_i32 = 0;
    for (int c = 0; c < C; ++c) {
        const int32_t xv = (int32_t)x[c];
        sum_i32 += xv;
        sumsq_i32 += xv * xv;
    }

    float mean = 0.0f;
    float var  = 0.0f;
    if (C > 0) {
        const float invC = 1.0f / (float)C;
        const float mean_i = (float)sum_i32 * invC;
        float var_i = (float)sumsq_i32 * invC - mean_i * mean_i;
        if (var_i < 0.0f) var_i = 0.0f;
        mean = scale_x * mean_i;
        var  = (scale_x * scale_x) * var_i;
    }

    const float inv_std     = rsqrtf(var + eps);
    const float inv_scale_y = (scale_y != 0.0f) ? (1.0f / scale_y) : 0.0f;

#if defined(USE_INT_LAYERNORM) && USE_INT_LAYERNORM \
 && defined(LN_INT_ENABLED) && LN_INT_ENABLED
    {
        int32_t _gq14[KER_LN_INT_MAX_C];
        int32_t _bq14[KER_LN_INT_MAX_C];
        kerln_int_precompute_coeffs(gamma, beta, C, inv_scale_y, _gq14, _bq14);
        int32_t K_q14  = (int32_t)lrintf(inv_std * scale_x * inv_scale_y * KER_LN_INT_SCALEF);
        int32_t Km_q14 = (int32_t)lrintf(inv_std * mean * inv_scale_y * KER_LN_INT_SCALEF);
        kerln_int_affine_token(x, 1, C, K_q14, Km_q14, _gq14, _bq14, y, 1);
    }
#else
    for (int c = 0; c < C; ++c) {
        const float xc   = scale_x * (float)x[c];
        const float norm = (xc - mean) * inv_std;

        const float g = gamma ? gamma[c] : 1.0f;
        const float b = beta  ? beta[c]  : 0.0f;

        const float yf = norm * g + b;
        const int32_t q = (int32_t)lrintf(yf * inv_scale_y);

        y[c] = ln__saturate_i8(q);
    }
#endif
}

KER_LN_FWD_STATIC KER_LN_FWD_ATTR void layernorm_forward_f32_i8io(
    const int8_t *__restrict x,    // [C, T]
    int C,
    int T,
    float scale_x,                 // dequant of x
    const float *__restrict gamma, // [C] or NULL
    const float *__restrict beta,  // [C] or NULL
    float eps,
    float scale_y,                 // requant of y
    int8_t *__restrict y           // [C, T]
){
    const float inv_scale_y = (scale_y != 0.0f) ? (1.0f / scale_y) : 0.0f;

    int t = 0;
    float mean = 0.0f;
    float var  = 0.0f;
    float inv_std = 0.0f;

    const int8_t *__restrict x_col = NULL;

#if defined(USE_INT_LAYERNORM) && USE_INT_LAYERNORM \
 && defined(LN_INT_ENABLED) && LN_INT_ENABLED
    int32_t _ln_gq14[KER_LN_INT_MAX_C];
    int32_t _ln_bq14[KER_LN_INT_MAX_C];
    const float _ln_sx_over_sy = scale_x * inv_scale_y;
    kerln_int_precompute_coeffs(gamma, beta, C, inv_scale_y, _ln_gq14, _ln_bq14);
#else
    int c = 0;
    int8_t *__restrict yw = NULL;
    const int8_t *__restrict xr = NULL;
    float xc   = 0.0f;
    float norm = 0.0f;
    float g    = 1.0f;
    float b    = 0.0f;
    float yf   = 0.0f;
    int32_t q  = 0;
#endif

#if KER_LN_USE_XBUF64
    int8_t xbuf64[64];
#endif

    for (t = 0; t < T; ++t) {
        x_col = x + t;

        const int use_buf = (KER_LN_USE_XBUF64 && C > 0 && C <= 64);
        ln__mean_var_column_i8_accum(
            x_col, C, T, scale_x, &mean, &var,
    #if KER_LN_USE_XBUF64
            use_buf ? xbuf64 : NULL
    #else
            NULL
    #endif
        );
        inv_std = rsqrtf(var + eps);

#if defined(USE_INT_LAYERNORM) && USE_INT_LAYERNORM \
 && defined(LN_INT_ENABLED) && LN_INT_ENABLED
        {
            int32_t K_q14  = (int32_t)lrintf(inv_std * _ln_sx_over_sy * KER_LN_INT_SCALEF);
            int32_t Km_q14 = (int32_t)lrintf(inv_std * mean * inv_scale_y * KER_LN_INT_SCALEF);

            if (use_buf) {
                kerln_int_affine_token(
                    xbuf64, 1, C, K_q14, Km_q14, _ln_gq14, _ln_bq14,
                    y + t, T);
            } else {
                kerln_int_affine_token(
                    x_col, T, C, K_q14, Km_q14, _ln_gq14, _ln_bq14,
                    y + t, T);
            }
        }
#else
        if (use_buf) {
            if (gamma && beta) {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)xbuf64[c];
                    norm = (xc - mean) * inv_std;
                    yf = norm * gamma[c] + beta[c];
                    q  = (int32_t)lrintf(yf * inv_scale_y);
                    y[(size_t)c * (size_t)T + (size_t)t] = ln__saturate_i8(q);
                }
            } else if (gamma) {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)xbuf64[c];
                    norm = (xc - mean) * inv_std;
                    yf = norm * gamma[c];
                    q  = (int32_t)lrintf(yf * inv_scale_y);
                    y[(size_t)c * (size_t)T + (size_t)t] = ln__saturate_i8(q);
                }
            } else if (beta) {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)xbuf64[c];
                    norm = (xc - mean) * inv_std;
                    yf = norm + beta[c];
                    q  = (int32_t)lrintf(yf * inv_scale_y);
                    y[(size_t)c * (size_t)T + (size_t)t] = ln__saturate_i8(q);
                }
            } else {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)xbuf64[c];
                    norm = (xc - mean) * inv_std;
                    q  = (int32_t)lrintf(norm * inv_scale_y);
                    y[(size_t)c * (size_t)T + (size_t)t] = ln__saturate_i8(q);
                }
            }
        } else {
            xr = x + t;
            yw = y + t;

            if (gamma && beta) {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)(*xr);
                    norm = (xc - mean) * inv_std;
                    yf = norm * gamma[c] + beta[c];
                    q  = (int32_t)lrintf(yf * inv_scale_y);
                    *yw = ln__saturate_i8(q);
                    xr += T;
                    yw += T;
                }
            } else if (gamma) {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)(*xr);
                    norm = (xc - mean) * inv_std;
                    yf = norm * gamma[c];
                    q  = (int32_t)lrintf(yf * inv_scale_y);
                    *yw = ln__saturate_i8(q);
                    xr += T;
                    yw += T;
                }
            } else if (beta) {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)(*xr);
                    norm = (xc - mean) * inv_std;
                    yf = norm + beta[c];
                    q  = (int32_t)lrintf(yf * inv_scale_y);
                    *yw = ln__saturate_i8(q);
                    xr += T;
                    yw += T;
                }
            } else {
                for (c = 0; c < C; ++c) {
                    xc   = scale_x * (float)(*xr);
                    norm = (xc - mean) * inv_std;
                    q  = (int32_t)lrintf(norm * inv_scale_y);
                    *yw = ln__saturate_i8(q);
                    xr += T;
                    yw += T;
                }
            }
        }
#endif
    }
}

static inline void layernorm_forward_f32_i8io_noaffine(
    const int8_t *__restrict x, int C, int T,
    float scale_x, float eps, float scale_y,
    int8_t *__restrict y)
{
    layernorm_forward_f32_i8io(x, C, T, scale_x, NULL, NULL, eps, scale_y, y);
}

#ifdef __cplusplus
} // extern "C"
#endif

#ifndef KER_LAYERNORM_KEEP_SQRTF
#undef sqrtf
#endif

#endif // LAYERNORM_H_
