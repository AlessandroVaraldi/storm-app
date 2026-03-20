#pragma once
#ifndef MOD_ATTNPOOL_FLASH_H_
#define MOD_ATTNPOOL_FLASH_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ker_layernorm.h" 
#include "ker_linear.h" 
#include "ker_gelu.h"
#include "ker_softmax.h"
#include "ker_dotproduct.h"

#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif
#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

static inline int attnpool_forward_i8_streamed(
    // Input
    const int8_t*  __restrict Z, int C, int T,
    // LayerNorm
    float ln_scale_x, float ln_scale_y, float ln_eps,
    const float*    __restrict ln_gamma, // [C] (nullable, in SRAM)
    const float*    __restrict ln_beta,  // [C] (nullable, in SRAM)
    // FC0 (C -> H)
    const fc_flash_desc_t* __restrict fd_fc0, // Cin=C, Cout=H
    int H,
    // GELU LUT params (index mapping + Q15->int8 to FC1 input scale)
    int32_t gelu_alpha, int32_t gelu_beta, int gelu_rshift,
    int32_t gelu_M_out, int32_t gelu_R_out,
    // FC1 (H -> 1)
    const fc_flash_desc_t* __restrict fd_fc1, // Cin=H, Cout=1
    // Softmax: precomputed integer LUT bin scale (from export)
    int32_t softmax_mul_bins,
    // Optional per-channel requant for the final feature (nullable)
    const int32_t* __restrict M_feat, // [C] or NULL
    const int32_t* __restrict R_feat, // [C] or NULL
    // FLASH reader
    flash_read_fn fread,
    // Shared tiles / work buffers
    int8_t*  __restrict wbuf0,        // weights tile buffer (int8), size >= FC_OUT_TILE*max(C,H)
    int8_t*  __restrict wbuf1,        // secondary tile / temp vector buffer (int8)
    int32_t* __restrict acc_tile,     // [FC_OUT_TILE]
    int32_t* __restrict bbuf,         // [FC_OUT_TILE]
    int32_t* __restrict mbuf,         // [FC_OUT_TILE]
    int32_t* __restrict rbuf,         // [FC_OUT_TILE]
    int32_t* __restrict logits_T,     // [T] (int32)
    int16_t* __restrict probs_q15,    // [T] (int16)
    // Slabs
    int8_t*  __restrict ZLN,          // [C, T]  (LayerNorm(Z))
    int8_t*  __restrict H1,           // [H, T]  (FC0 output)
    int8_t*  __restrict H1A,          // [H, T]  (GELU(H1))
    // Optional debug outputs (nullable)
    int8_t*  __restrict s_i8_out,     // [T]
    int16_t* __restrict w_q15_out,    // [T]
    // Output
    int8_t*  __restrict feat_i8       // [C]
)
{
    if (UNLIKELY(C <= 0 || T <= 0 || H <= 0 || !Z || !feat_i8 || !fd_fc0 || !fd_fc1 || !fread))
        return -1;

    // 1) LayerNorm over time: Z -> ZLN  (operate on the full [C,T])
    layernorm_forward_f32_i8io(
        Z, C, T,
        ln_scale_x,
        ln_gamma, ln_beta,
        ln_eps,
        ln_scale_y,
        ZLN);

    // 2) FC0: ZLN [C,T]  →  H1 [H,T]   (tile over H; reuse each weight tile for all T)
    for (int o0 = 0; o0 < H; o0 += FC_OUT_TILE) {
        const int oN = (o0 + FC_OUT_TILE <= H) ? FC_OUT_TILE : (H - o0);
        const size_t w_bytes = (size_t)oN * (size_t)fd_fc0->Cin;

        // Load per-tile params
        if (fread(fd_fc0->off_b + (uint32_t)(o0 * 4), bbuf, (size_t)oN * 4)) return -2;
        if (fread(fd_fc0->off_m + (uint32_t)(o0 * 4), mbuf, (size_t)oN * 4)) return -3;
        if (fread(fd_fc0->off_r + (uint32_t)(o0 * 4), rbuf, (size_t)oN * 4)) return -4;
        // Load weight tile [oN, C]
        if (fread(fd_fc0->off_w + (uint32_t)((size_t)o0 * (size_t)fd_fc0->Cin), wbuf0, w_bytes)) return -5;

        // Process all tokens
        for (int t = 0; t < T; ++t) {
            // Gather ZLN[:,t] -> contiguous vector in wbuf1[0:C]
            for (int c = 0; c < C; ++c) wbuf1[c] = ZLN[(size_t)c * (size_t)T + (size_t)t];

            // FC core on this out-tile for token t
            fc_core_outtile_one_row(wbuf1, C, wbuf0, oN, bbuf, acc_tile);

            // Requantize (per-channel) to int8; write into H1[o0:o0+oN, t]
            fc_requant_outtile(acc_tile, mbuf, rbuf, oN, C, wbuf1);
            for (int k = 0; k < oN; ++k) {
                const int hidx = o0 + k;
                H1[(size_t)hidx * (size_t)T + (size_t)t] = wbuf1[k];
            }
        }
    }

    // 3) GELU on H1 per token: H1[:,t] → H1A[:,t]
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) wbuf1[h] = H1[(size_t)h * (size_t)T + (size_t)t];

        gelu_lut_q15_model(wbuf1, (size_t)H, gelu_alpha, gelu_beta, gelu_rshift, gelu_M_out, gelu_R_out, wbuf1);

        // Scatter back
        for (int h = 0; h < H; ++h) H1A[(size_t)h * (size_t)T + (size_t)t] = wbuf1[h];
    }

    // 4) FC1 (H -> 1) for all tokens:
    int32_t b1 = 0, m1 = 0, r1 = 0;
    // Read bias/m/r (each length 1)
    if (fd_fc1->off_b) { if (fread(fd_fc1->off_b, &b1, 4)) return -6; } else { b1 = 0; }
    if (fread(fd_fc1->off_m, &m1, 4)) return -7;
    if (fread(fd_fc1->off_r, &r1, 4)) return -8;
    // Read weight vector [1,H] into wbuf0[0:H]
    if (fread(fd_fc1->off_w, wbuf0, (size_t)H)) return -9;

    for (int t = 0; t < T; ++t) {
        // Compute dot: s = b1 + sum_h w[h]*H1A[h,t]
        // Use strided access across H1A (channel-major [H,T]): elements are at h*T + t
        int32_t acc = b1;
        // unrolled dot (portable)
        int h = 0;
        for (; h + 4 <= H; h += 4) {
            acc += (int32_t)wbuf0[h+0] * (int32_t)H1A[(size_t)(h+0) * (size_t)T + (size_t)t];
            acc += (int32_t)wbuf0[h+1] * (int32_t)H1A[(size_t)(h+1) * (size_t)T + (size_t)t];
            acc += (int32_t)wbuf0[h+2] * (int32_t)H1A[(size_t)(h+2) * (size_t)T + (size_t)t];
            acc += (int32_t)wbuf0[h+3] * (int32_t)H1A[(size_t)(h+3) * (size_t)T + (size_t)t];
        }
        for (; h < H; ++h) {
            acc += (int32_t)wbuf0[h] * (int32_t)H1A[(size_t)h * (size_t)T + (size_t)t];
        }

        // Requant to int8 (single-channel)
        int64_t prod = (int64_t)acc * (int64_t)m1;
        int32_t s_q;
        if (r1 > 0)      s_q = (int32_t)((prod + ((int64_t)1 << (r1 - 1))) >> r1);
        else if (r1 < 0) s_q = (int32_t)(prod << (-r1));
        else             s_q = (int32_t)prod;
        int8_t s_i8 = SATURATE_INT8(s_q);

        if (s_i8_out) s_i8_out[t] = s_i8;
        logits_T[t] = (int32_t)s_i8;
    }

    // 5) Softmax over T (Q15) — uses precomputed integer bin scale
    softmax_row_q15_bins(logits_T, (size_t)T, softmax_mul_bins, SM_SOFTMAX_INTBINS_RSHIFT, probs_q15);
    if (w_q15_out) {
        for (int t = 0; t < T; ++t) w_q15_out[t] = probs_q15[t];
    }

    // 6) Weighted sum over time to get feat_i8[C]
    if (M_feat && R_feat) {
        attnpool_weighted_sum_ct_q15_to_i8_requant(ZLN, C, T, probs_q15, M_feat, R_feat, feat_i8);
    } else {
        attnpool_weighted_sum_ct_q15_to_i8(ZLN, C, T, probs_q15, feat_i8);
    }

    return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MOD_ATTNPOOL_FLASH_H_
