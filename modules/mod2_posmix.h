#pragma once
#ifndef POSMIX_FLASH_H_
#define POSMIX_FLASH_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ker_conv1d.h"  // conv1d_dw_flash_stream, descriptors, conv1d_out_len

static inline int8_t sat_i8(int v) {
    return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
}

static inline void posmix_depthwise_residual_flash(
    const int8_t *__restrict x,                   // [C, T]
    int C, int T,
    const conv1d_dw_flash_desc_t *__restrict cd,  // C, K, stride, pad, dil + FLASH offsets
    flash_read_fn fread,                          // FLASH reader
    int8_t *__restrict y,                         // [C, T] (output)
    int32_t *__restrict acc_scalar,               // [1]
    int8_t *__restrict kbuf,                      // [K]
    int32_t *__restrict bmr_tmp                   // [3]
)
{
    if (!x || !cd || !fread || !y || !acc_scalar || !kbuf || !bmr_tmp) return;
    if (C <= 0 || T <= 0 || cd->K <= 0 || cd->stride <= 0 || cd->dil <= 0) return;

    const int Tout = conv1d_out_len(T, cd->K, cd->stride, cd->pad, cd->dil);

    if (Tout != T) {
        for (int c = 0; c < C; ++c) {
            const int base = c * T;
            for (int t = 0; t < T; ++t) y[base + t] = x[base + t];
        }
        return;
    }

    // 1) Depthwise conv output into y
    conv1d_dw_flash_stream(x, T, cd, fread, y, acc_scalar, kbuf, bmr_tmp);

    // 2) Residual add
    for (int c = 0; c < C; ++c) {
        const int base = c * T;
        for (int t = 0; t < T; ++t) {
            y[base + t] = sat_i8((int)x[base + t] + (int)y[base + t]);
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // POSMIX_FLASH_H_
