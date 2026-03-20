#pragma once
#ifndef MOD_MLP_FLASH_H_
#define MOD_MLP_FLASH_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

#include "ker_layernorm.h"
#include "ker_linear.h" 
#include "ker_gelu.h"

#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif
#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

    // Gather column t from [C,T] (channel-major) into contiguous [C]
    static inline void mlp__gather_token_ct(
        const int8_t *__restrict X, int C, int T, int t,
        int8_t *__restrict row_out)
    {
        const int tt = (t < 0) ? 0 : (t >= T ? (T - 1) : t);
        // Unroll x4
        int c = 0;
        for (; c + 3 < C; c += 4) {
            row_out[c+0] = X[(c+0) * T + tt];
            row_out[c+1] = X[(c+1) * T + tt];
            row_out[c+2] = X[(c+2) * T + tt];
            row_out[c+3] = X[(c+3) * T + tt];
        }
        for (; c < C; ++c) {
            row_out[c] = X[c * T + tt];
        }
    }

    // Scatter contiguous [C] into column t of [C,T] (channel-major)
    static inline void mlp__scatter_token_ct(
        const int8_t *__restrict row_in, int C, int T, int t,
        int8_t *__restrict Y)
    {
        // Unroll x4
        int c = 0;
        for (; c + 3 < C; c += 4)
        {
            Y[(c + 0) * T + t] = row_in[c + 0];
            Y[(c + 1) * T + t] = row_in[c + 1];
            Y[(c + 2) * T + t] = row_in[c + 2];
            Y[(c + 3) * T + t] = row_in[c + 3];
        }
        for (; c < C; ++c)
        {
            Y[c * T + t] = row_in[c];
        }
    }

static inline void mlp_forward_i8_preln_flash(
    // Input / geometry
    const int8_t *__restrict x, int C, int T, int H,
    // LayerNorm params
    float ln_scale_x, float ln_scale_y, float ln_eps,
    const float *__restrict ln_gamma, // [C] (nullable)
    const float *__restrict ln_beta,  // [C] (nullable)
    // FC descriptors in FLASH
    const fc_flash_desc_t *__restrict fd_fc1, // Cin=C, Cout=H
    const fc_flash_desc_t *__restrict fd_fc2, // Cin=H, Cout=C
    flash_read_fn fread,
    // GELU LUT params
    int32_t gelu_alpha, int32_t gelu_beta, int gelu_rshift,
    int32_t gelu_M_out, int32_t gelu_R_out,
    // Work buffers
    int8_t *__restrict buf_tok_in,  // [max(C,H)]
    int8_t *__restrict buf_tok_mid, // [max(C,H)]
    int32_t *__restrict acc_tile,   // [FC_OUT_TILE]
    int8_t *__restrict wbuf0,       // [FC_OUT_TILE * max(C,H)]
    int8_t *__restrict wbuf1,       // [FC_OUT_TILE * max(C,H)]
    int32_t *__restrict bbuf,       // [FC_OUT_TILE]
    int32_t *__restrict mbuf,       // [FC_OUT_TILE]
    int32_t *__restrict rbuf,       // [FC_OUT_TILE]
    // Slabs
    int8_t *__restrict H1,  // [H, T]
    int8_t *__restrict H1A, // [H, T]
    // Output
    int8_t *__restrict y // [C, T]
)
{
    if (UNLIKELY(C <= 0 || T <= 0 || H <= 0 || !x || !y || !fd_fc1 || !fd_fc2 || !fread))
        return;

    // 1) Pre-LayerNorm into y (y holds y_ln afterwards)
    layernorm_forward_f32_i8io(
        x, C, T,
        ln_scale_x,
        ln_gamma, ln_beta,
        ln_eps,
        ln_scale_y,
        y);

    // 2) FC1: y_ln [C,T] → H1 [H,T] (tiled su H) with weights ping-pong
    for (int o0 = 0, tile_id = 0; o0 < H; o0 += FC_OUT_TILE, ++tile_id)
    {
        const int oN = (o0 + FC_OUT_TILE <= H) ? FC_OUT_TILE : (H - o0);
        const size_t w_tile_bytes = (size_t)oN * (size_t)fd_fc1->Cin;

        int8_t *wbuf_cur = (tile_id & 1) ? wbuf1 : wbuf0;

        // Load per-tile params
        fread(fd_fc1->off_b + (uint32_t)(o0 * 4), bbuf, (size_t)oN * 4);
        fread(fd_fc1->off_m + (uint32_t)(o0 * 4), mbuf, (size_t)oN * 4);
        fread(fd_fc1->off_r + (uint32_t)(o0 * 4), rbuf, (size_t)oN * 4);
        // Load weights tile
        fread(fd_fc1->off_w + (uint32_t)((size_t)o0 * (size_t)fd_fc1->Cin), wbuf_cur, w_tile_bytes);

        for (int t = 0; t < T; ++t)
        {
            // gather y_ln[:,t] -> buf_tok_in[C]
            mlp__gather_token_ct(y, C, T, t, buf_tok_in);

            // compute tile outputs for token t
            fc_core_outtile_one_row(buf_tok_in, C, wbuf_cur, oN, bbuf, acc_tile);
            fc_requant_outtile(acc_tile, mbuf, rbuf, oN, C, buf_tok_mid);

            // scatter into H1[o0:o0+oN, t]
            for (int k = 0; k < oN; ++k)
            {
                const int hidx = o0 + k;
                H1[(size_t)hidx * (size_t)T + (size_t)t] = buf_tok_mid[k];
            }
        }
    }

    // 3) GELU: apply per token on H1[:,t] → H1A[:,t]
    for (int t = 0; t < T; ++t)
    {
        // gather H contiguous vector of column t
        int h = 0;
        for (; h + 3 < H; h += 4) {
            buf_tok_mid[h+0] = H1[(size_t)(h+0) * (size_t)T + (size_t)t];
            buf_tok_mid[h+1] = H1[(size_t)(h+1) * (size_t)T + (size_t)t];
            buf_tok_mid[h+2] = H1[(size_t)(h+2) * (size_t)T + (size_t)t];
            buf_tok_mid[h+3] = H1[(size_t)(h+3) * (size_t)T + (size_t)t];
        }
        for (; h < H; ++h) {
            buf_tok_mid[h] = H1[(size_t)h * (size_t)T + (size_t)t];
        }

        gelu_lut_q15_model(buf_tok_mid, (size_t)H, gelu_alpha, gelu_beta, gelu_rshift, gelu_M_out, gelu_R_out, buf_tok_mid);

        // scatter back into H1A[:,t]
        h = 0;
        for (; h + 3 < H; h += 4) {
            H1A[(size_t)(h+0) * (size_t)T + (size_t)t] = buf_tok_mid[h+0];
            H1A[(size_t)(h+1) * (size_t)T + (size_t)t] = buf_tok_mid[h+1];
            H1A[(size_t)(h+2) * (size_t)T + (size_t)t] = buf_tok_mid[h+2];
            H1A[(size_t)(h+3) * (size_t)T + (size_t)t] = buf_tok_mid[h+3];
        }
        for (; h < H; ++h) {
            H1A[(size_t)h * (size_t)T + (size_t)t] = buf_tok_mid[h];
        }
    }

    // 4) FC2: H1A [H,T] → out [C,T] (tiled over output C, reuse tile across all tokens)
    for (int o0 = 0, tile_id = 0; o0 < C; o0 += FC_OUT_TILE, ++tile_id)
    {
        const int oN = (o0 + FC_OUT_TILE <= C) ? FC_OUT_TILE : (C - o0);
        const size_t w_tile_bytes = (size_t)oN * (size_t)fd_fc2->Cin;
        int8_t *wbuf_cur = (tile_id & 1) ? wbuf1 : wbuf0;

        // Load per-tile params
        fread(fd_fc2->off_b + (uint32_t)(o0 * 4), bbuf, (size_t)oN * 4);
        fread(fd_fc2->off_m + (uint32_t)(o0 * 4), mbuf, (size_t)oN * 4);
        fread(fd_fc2->off_r + (uint32_t)(o0 * 4), rbuf, (size_t)oN * 4);
        // Load weights tile
        fread(fd_fc2->off_w + (uint32_t)((size_t)o0 * (size_t)fd_fc2->Cin), wbuf_cur, w_tile_bytes);

        for (int t = 0; t < T; ++t)
        {
            // gather H1A[:,t] -> buf_tok_in[H]
            int h = 0;
            for (; h + 3 < H; h += 4) {
                buf_tok_in[h+0] = H1A[(size_t)(h+0) * (size_t)T + (size_t)t];
                buf_tok_in[h+1] = H1A[(size_t)(h+1) * (size_t)T + (size_t)t];
                buf_tok_in[h+2] = H1A[(size_t)(h+2) * (size_t)T + (size_t)t];
                buf_tok_in[h+3] = H1A[(size_t)(h+3) * (size_t)T + (size_t)t];
            }
            for (; h < H; ++h) {
                buf_tok_in[h] = H1A[(size_t)h * (size_t)T + (size_t)t];
            }

            // compute tile outputs for token t
            fc_core_outtile_one_row(buf_tok_in, H, wbuf_cur, oN, bbuf, acc_tile);
            fc_requant_outtile(acc_tile, mbuf, rbuf, oN, H, buf_tok_mid);

            // residual add with original x[:,t] and store into y[:,t]
            for (int k = 0; k < oN; ++k)
            {
                const int cidx = o0 + k;
                const size_t pos = (size_t)cidx * (size_t)T + (size_t)t;
                const int16_t s = (int16_t)buf_tok_mid[k] + (int16_t)x[pos];
                y[pos] = SATURATE_INT8(s);
            }
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MOD_MLP_FLASH_H_
