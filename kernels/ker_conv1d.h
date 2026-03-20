#pragma once
#ifndef KER_CONV1D_H_
#define KER_CONV1D_H_

#include <stdint.h>
#include <stddef.h>

#include "ker_xpulp.h"
#include "ker_linear.h"  // for fc_requant_R1, REQUANT32_M_SHIFT

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CONV1D_OUT_TILE
#define CONV1D_OUT_TILE 32   // output-channel tile size for pointwise conv
#endif

#ifndef CONV1D_ALIGN
#define CONV1D_ALIGN 4       // FLASH reads aligned to 4 bytes
#endif

#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

static inline int conv1d_out_len(int Tin, int K, int stride, int pad, int dil)
{
    const int eff = (K - 1) * dil + 1;
    const int num = Tin + 2 * pad - eff;
    return (num >= 0) ? (num / stride + 1) : 0;
}

// FLASH I/O descriptor
#ifndef XHEEP_FLASH_READ_FN_TYPEDEF
#define XHEEP_FLASH_READ_FN_TYPEDEF
typedef int (*flash_read_fn)(uint32_t offset, void *dst, size_t nbytes);
#endif

// Pointwise weight layout in FLASH: [Cout, Cin, K] int8
typedef struct {
    // Byte offsets in FLASH
    uint32_t off_w;   // weights int8, order [Cout, Cin, K]
    uint32_t off_b;   // bias    int32 [Cout]
    uint32_t off_m;   // mul     int32 [Cout]
    uint32_t off_r;   // r       int32 [Cout]
    // Conv parameters
    int Cin, Cout;
    int K, stride, pad, dil;
} conv1d_pw_flash_desc_t;

// Depthwise weight layout in FLASH : w:[Cin, 1, K] int8; Cout = Cin
typedef struct {
    uint32_t off_w;   // weights int8, order [Cin, 1, K]
    uint32_t off_b;   // bias    int32 [Cin]
    uint32_t off_m;   // mul     int32 [Cin]
    uint32_t off_r;   // r       int32 [Cin]
    // Conv parameters
    int C;            // Cin == Cout
    int K, stride, pad, dil;
} conv1d_dw_flash_desc_t;

// Pointwise, OUT-tile: accumulate one temporal row (one t_out) for the tile
// x   : [Cin, Tin]          (int8, SRAM)
// wtl : [oN, Cin, K]        (int8, SRAM)  -- tile pesi già letta da FLASH
// btl : [oN]                (int32, SRAM) -- bias tile
// acc : [oN]                (int32, SRAM) -- out
static inline void conv1d_pw_core_line_tile(
    const int8_t *__restrict x, int Tin, int Cin,
    const int8_t *__restrict wtl, const int32_t *__restrict btl,
    int oN, int K, int stride, int pad, int dil,
    int t_out, int32_t *__restrict acc)
{
    const int ti0 = t_out * stride - pad;
    // init with bias
    for (int o = 0; o < oN; ++o) acc[o] = btl ? btl[o] : 0;

#if XHEEP_USE_XPULP_DOT4
    // Fast-path: dil=1 and fully in-bounds window -> use packed dot4 over K.
    if (dil == 1 && ti0 >= 0 && (ti0 + K) <= Tin) {
        for (int ci = 0; ci < Cin; ++ci) {
            const int8_t *xptr = x + (size_t)ci * (size_t)Tin + (size_t)ti0;
            const int w_ci_base = ci * K;

            int k = 0;
            for (; k + 4 <= K; k += 4) {
                const uint32_t xp = xheep_load_u32_unaligned((const void *)(xptr + k));
                const int w_k_off = w_ci_base + k;
                for (int o = 0; o < oN; ++o) {
                    const int w_base_o = o * (Cin * K);
                    const uint32_t wp = xheep_load_u32_unaligned((const void *)(wtl + w_base_o + w_k_off));
                    acc[o] += dot4_i8_xpulp_packed_l(xp, wp);
                }
            }
            for (; k < K; ++k) {
                const int32_t xv = (int32_t)xptr[k];
                const int w_k_off = w_ci_base + k;
                for (int o = 0; o < oN; ++o) {
                    const int w_base_o = o * (Cin * K);
                    acc[o] += xv * (int32_t)wtl[w_base_o + w_k_off];
                }
            }
        }
        return;
    }
#endif

    for (int ci = 0; ci < Cin; ++ci) {
        const int x_ch = ci * Tin;
        const int w_ci_base = ci * K; // within block [o, Cin, K]
        for (int k = 0; k < K; ++k) {
            const int ti = ti0 + k * dil;
            if ((unsigned)ti >= (unsigned)Tin) continue;
            const int8_t xv = x[x_ch + ti];
            const int w_k_off = w_ci_base + k;
            // vectorize over oN dim
            for (int o = 0; o < oN; ++o) {
                const int w_base_o = o * (Cin * K);
                acc[o] += (int32_t)xv * (int32_t)wtl[w_base_o + w_k_off];
            }
        }
    }
}

// Depthwise, one channel at a time: accumulate the full line for one channel
static inline void conv1d_dw_core_line_chan(
    const int8_t *__restrict x_c, int Tin,
    const int8_t *__restrict w_k, int K, int stride, int pad, int dil,
    int t_out, int32_t *__restrict acc /* size 1: singolo canale */)
{
    const int ti0 = t_out * stride - pad;
    int32_t s = acc[0];

#if XHEEP_USE_XPULP_DOT4
    // Fast-path: contiguous window
    if (dil == 1 && ti0 >= 0 && (ti0 + K) <= Tin) {
        const int8_t *xptr = x_c + ti0;
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            const uint32_t xp = xheep_load_u32_unaligned((const void *)(xptr + k));
            const uint32_t wp = xheep_load_u32_unaligned((const void *)(w_k + k));
            s += dot4_i8_xpulp_packed_l(xp, wp);
        }
        for (; k < K; ++k) {
            s += (int32_t)xptr[k] * (int32_t)w_k[k];
        }
        acc[0] = s;
        return;
    }
#endif

    for (int k = 0; k < K; ++k) {
        const int ti = ti0 + k * dil;
        if ((unsigned)ti < (unsigned)Tin) {
            s += (int32_t)x_c[ti] * (int32_t)w_k[k];
        }
    }
    acc[0] = s;
}

// 32-bit only requant for a single accumulator value.
static inline int8_t requant_scalar(
    int32_t acc, int32_t m, int32_t r, int Cin_eff)
{
    const int R1 = fc_requant_R1(Cin_eff);
    const int32_t rnd1 = R1 > 0 ? ((int32_t)1 << (R1 - 1)) : 0;
    const int32_t a = (acc + rnd1) >> R1;
    const int32_t prod = a * m;
    const int32_t rnd2 = (int32_t)1 << (r - 1);
    return SATURATE_INT8((prod + rnd2) >> r);
}

// WRAPPER: POINTWISE FLASH-TILED
// x : [Cin, Tin]           (int8, SRAM)
// y : [Cout, Tout]         (int8, SRAM) in channel-major
// wbuf0/1: [oTile * Cin * K] int8  (doppio buffer pesi)
// bbuf/mbuf/rbuf: [oTile] int32
// acc_line: [oTile] int32
static inline void conv1d_pw_flash_tiled(
    const int8_t *__restrict x, int Tin,
    const conv1d_pw_flash_desc_t *cd,
    flash_read_fn fread,
    int8_t *__restrict y /* [Cout, Tout] */,
    int32_t *__restrict acc_line,       /* [CONV1D_OUT_TILE] */
    int8_t *__restrict wbuf0, int8_t *__restrict wbuf1, /* [OUT_TILE*Cin*K] */
    int32_t *__restrict bbuf, int32_t *__restrict mbuf, int32_t *__restrict rbuf)
{
    if (!x || !cd || !fread || !y || !acc_line || !wbuf0 || !wbuf1 || !bbuf || !mbuf || !rbuf) return;

    const int Cin = cd->Cin, Cout = cd->Cout;
    const int K = cd->K, stride = cd->stride, pad = cd->pad, dil = cd->dil;
    const int Tout = conv1d_out_len(Tin, K, stride, pad, dil);
    if (Tin <= 0 || Cin <= 0 || Cout <= 0 || K <= 0 || stride <= 0 || dil <= 0 || Tout <= 0) return;

    const size_t wrow_bytes = (size_t)Cin * (size_t)K;            // per-outchannel
    const size_t wtile_bytes = wrow_bytes * (size_t)CONV1D_OUT_TILE;

    int8_t *wb[2] = { wbuf0, wbuf1 };
    int ping = 0;

    for (int o0 = 0; o0 < Cout; o0 += CONV1D_OUT_TILE)
    {
        const int oN = (o0 + CONV1D_OUT_TILE <= Cout) ? CONV1D_OUT_TILE : (Cout - o0);
        const size_t tile_bytes = (size_t)oN * wrow_bytes;

        // Load b/m/r tile
        fread(cd->off_b + (uint32_t)(o0 * 4), bbuf, (size_t)oN * 4);
        fread(cd->off_m + (uint32_t)(o0 * 4), mbuf, (size_t)oN * 4);
        fread(cd->off_r + (uint32_t)(o0 * 4), rbuf, (size_t)oN * 4);

        // Prefetch weight tile 0
        fread(cd->off_w + (uint32_t)((size_t)o0 * wrow_bytes), wb[ping], tile_bytes);

        // For each temporal position
        for (int to = 0; to < Tout; ++to)
        {
            // Accumulate
            conv1d_pw_core_line_tile(
                x, Tin, Cin,
                wb[ping], bbuf,
                oN, K, stride, pad, dil,
                to, acc_line);

            // Requant + linear store (channel-major)
            int8_t *y_line = y + (size_t)o0 * (size_t)Tout + (size_t)to;
            for (int o = 0; o < oN; ++o) {
                const int8_t v = requant_scalar(acc_line[o], mbuf[o], rbuf[o], Cin * K);
                y_line[o * Tout] = v;
            }
        }
    }
}

// WRAPPER: DEPTHWISE FLASH STREAMING
// x : [C, Tin], y : [C, Tout]
static inline void conv1d_dw_flash_stream(
    const int8_t *__restrict x, int Tin,
    const conv1d_dw_flash_desc_t *cd,
    flash_read_fn fread,
    int8_t *__restrict y,
    int32_t *__restrict acc_scalar, /* [1]  single-channel accumulator */
    int8_t *__restrict kbuf,        /* [K]  channel kernel */
    int32_t *__restrict bmr /* [3]  temporary b,m,r */)
{
    if (!x || !cd || !fread || !y || !acc_scalar || !kbuf || !bmr) return;
    const int C = cd->C, K = cd->K, stride = cd->stride, pad = cd->pad, dil = cd->dil;
    const int Tout = conv1d_out_len(Tin, K, stride, pad, dil);
    if (Tin <= 0 || C <= 0 || K <= 0 || stride <= 0 || dil <= 0 || Tout <= 0) return;

    for (int c = 0; c < C; ++c)
    {
        // load kernel K for channel c
        const uint32_t w_off = cd->off_w + (uint32_t)((size_t)c * (size_t)K);
        fread(w_off, kbuf, (size_t)K);

        // load b/m/r for channel c
        fread(cd->off_b + (uint32_t)(c * 4), &bmr[0], 4);
        fread(cd->off_m + (uint32_t)(c * 4), &bmr[1], 4);
        fread(cd->off_r + (uint32_t)(c * 4), &bmr[2], 4);

        const int8_t *x_c = x + (size_t)c * (size_t)Tin;
        int8_t *y_c = y + (size_t)c * (size_t)Tout;

        for (int to = 0; to < Tout; ++to)
        {
            acc_scalar[0] = bmr[0]; // bias
            conv1d_dw_core_line_chan(
                x_c, Tin,
                kbuf, K, stride, pad, dil,
                to, acc_scalar);

            y_c[to] = requant_scalar(acc_scalar[0], bmr[1], bmr[2], K);
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // KER_CONV1D_H_
