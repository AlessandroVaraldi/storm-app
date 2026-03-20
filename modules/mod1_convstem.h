#pragma once
#ifndef CONVSTEM_FLASH_H_
#define CONVSTEM_FLASH_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ker_conv1d.h"
#include "ker_silu.h"

static void __attribute__((noinline)) convstem_flash(
    const int8_t *__restrict x,                  // [Cin, Tin]
    int Tin,
    const conv1d_pw_flash_desc_t *__restrict cd, // Cin, Cout, K, stride, pad, dil + FLASH offsets
    flash_read_fn fread,                         // FLASH reader callback
    int32_t alpha, int32_t beta, int rshift,     // LUT index mapping params
    int8_t *__restrict y,                        // [Cout, Tout]
    int32_t *__restrict acc_line,                // [CONV1D_OUT_TILE]
    int8_t *__restrict wbuf0,                    // [CONV1D_OUT_TILE * Cin * K]
    int8_t *__restrict wbuf1,                    // [CONV1D_OUT_TILE * Cin * K]
    int32_t *__restrict bbuf,                    // [CONV1D_OUT_TILE]
    int32_t *__restrict mbuf,                    // [CONV1D_OUT_TILE]
    int32_t *__restrict rbuf                     // [CONV1D_OUT_TILE]
)
{
    if (!x || !cd || !fread || !y) return;
    if (!acc_line || !wbuf0 || !wbuf1 || !bbuf || !mbuf || !rbuf) return;

    const int Cin    = cd->Cin;
    const int Cout   = cd->Cout;
    const int K      = cd->K;
    const int stride = cd->stride;
    const int pad    = cd->pad;
    const int dil    = cd->dil;

    if (Tin <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || stride <= 0 || dil <= 0) return;

    const int Tout = conv1d_out_len(Tin, K, stride, pad, dil);
    if (Tout <= 0) return;

    // Stage A+B: Conv (FLASH-tiled) + per-channel requant to int8 (pre-activation)
    conv1d_pw_flash_tiled(
        x, Tin,
        cd,
        fread,
        y,                         // [Cout, Tout]
        acc_line,                  // [CONV1D_OUT_TILE]
        wbuf0, wbuf1,              // [OUT_TILE * Cin * K] each
        bbuf, mbuf, rbuf);         // [OUT_TILE] each

    // Stage C: SiLU via LUT (in-place over y)
    silu_lut_q15_model(
        y,
        (size_t)((size_t)Cout * (size_t)Tout),
        alpha, beta, rshift,
        y);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CONVSTEM_FLASH_H_
