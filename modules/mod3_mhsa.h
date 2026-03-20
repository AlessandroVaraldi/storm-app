#pragma once
#ifndef MHSA_FLASH_H_
#define MHSA_FLASH_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "ker_linear.h"
#include "ker_softmax.h"
#include "ker_layernorm.h"
#include "ker_xpulp.h"

#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#ifndef SATURATE_INT8
#define SATURATE_INT8(x) ((int8_t)((x) < -128 ? -128 : ((x) > 127 ? 127 : (x))))
#endif

// Gather token t from [C,T] (channel-major) into a dense row [C]
static inline void mhsa__gather_token_ct(const int8_t *__restrict X, int C, int T, int t, int8_t *__restrict row_out)
{
    const int tt = (t < 0) ? 0 : (t >= T ? (T - 1) : t);
    for (int c = 0; c < C; ++c) row_out[c] = X[c * T + tt];
}

// Dot between two int8 vectors (length L) → int32
static inline int32_t mhsa__dot_i8_i8(const int8_t *__restrict a, const int8_t *__restrict b, int L)
{
    return xheep_dot_i8_i8_any(a, b, L);
}

// Apply softmax on a vector of int32 logits with given float scale, output Q15
static inline void mhsa__softmax_row_q15f(const int32_t *__restrict logits, int N, float scale, int16_t *__restrict out_q15)
{
    #if SOFTMAX_USE_INTBINS
    softmax_row_q15f_intbins(logits, (size_t)N, scale, out_q15);
    #else
    softmax_row_q15f(logits, (size_t)N, scale, out_q15);
    #endif
}

// Q/K/V are stored as [token][head][dh], contiguous in dh (TOKEN-MAJOR layout for cache efficiency)
// Old: [h][T][dh] - caused strided access when iterating tokens
// New: [t][h][dh] - makes Q[t] and K[t] contiguous, eliminating cache misses
static inline int8_t* mhsa__ptr_q(const int8_t* Qbase, int nhead, int T, int dh, int t, int h)
{ return (int8_t*)(Qbase + ((size_t)t * (size_t)nhead + (size_t)h) * (size_t)dh); }
static inline int8_t* mhsa__ptr_k(const int8_t* Kbase, int nhead, int T, int dh, int t, int h)
{ return (int8_t*)(Kbase + ((size_t)t * (size_t)nhead + (size_t)h) * (size_t)dh); }
static inline int8_t* mhsa__ptr_v(const int8_t* Vbase, int nhead, int T, int dh, int t, int h)
{ return (int8_t*)(Vbase + ((size_t)t * (size_t)nhead + (size_t)h) * (size_t)dh); }

// Dot of two contiguous int8 vectors (length L) -> int32
static inline int32_t mhsa__dot_i8_i8_contig(const int8_t* __restrict a, const int8_t* __restrict b, int L)
{
    return xheep_dot_i8_i8_any(a, b, L);
}

static inline void mhsa_forward_i8_preln_flash(
    const int8_t *__restrict x, int C, int T,
    int nhead, int window_radius,
    float ln_scale_x, float ln_scale_y, float ln_eps,
    const float *__restrict ln_gamma, // nullable
    const float *__restrict ln_beta,  // nullable
    // Q, K, V and OUT descriptors in FLASH (now separate)
    const fc_flash_desc_t *__restrict fd_q,
    const fc_flash_desc_t *__restrict fd_k,
    const fc_flash_desc_t *__restrict fd_v,
    const fc_flash_desc_t *__restrict fd_out,
    flash_read_fn fread,
    // Softmax: precomputed integer LUT bin scale (from export)
    int32_t softmax_mul_bins,
    // Work buffers
    int32_t *__restrict scores_row, // [win_len]
    int8_t *__restrict buf_qkv,     // [3*C*T]
    int8_t *__restrict buf_tok_in,  // [C]
    int8_t *__restrict buf_tok_out, // [C] (optional use)
    int16_t *__restrict probs_q15,  // [win_len]
    int32_t *__restrict acc_tile,   // [FC_OUT_TILE]
    int8_t *__restrict wbuf0,       // [FC_OUT_TILE * C]
    int8_t *__restrict wbuf1,       // [FC_OUT_TILE * C]
    int32_t *__restrict bbuf,       // [FC_OUT_TILE]
    int32_t *__restrict mbuf,       // [FC_OUT_TILE]
    int32_t *__restrict rbuf,       // [FC_OUT_TILE]
    // Output
    int8_t *__restrict y            // [C, T]
)
{
    (void)buf_tok_in;
    (void)buf_tok_out;
    (void)wbuf1;

    if (UNLIKELY(C <= 0 || T <= 0 || nhead <= 0 || (C % nhead) != 0)) return;
    if (UNLIKELY(!fd_q || !fd_k || !fd_v || !fd_out)) return;
    if (UNLIKELY(fd_q->Cin != C || fd_q->Cout != C)) return;
    if (UNLIKELY(fd_k->Cin != C || fd_k->Cout != C)) return;
    if (UNLIKELY(fd_v->Cin != C || fd_v->Cout != C)) return;
    if (UNLIKELY(fd_out->Cin != C || fd_out->Cout != C)) return;

    const int dh = C / nhead;
    const int w  = (window_radius <= 0 || window_radius >= T) ? T : (2*window_radius + 1);
    const int win_len = w;
    if (UNLIKELY(win_len <= 0)) return;

    // 1) Pre-LayerNorm: write LN(x) into y (reuse y as y_ln).
    layernorm_forward_f32_i8io(
        x, C, T,
        ln_scale_x,
        ln_gamma, ln_beta,
        ln_eps,
        ln_scale_y,
        y);

    // Transpose y[C,T] → y_tc[T,C] once, replacing all per-token gathers.
    int8_t *y_tc = (int8_t *)alloca((size_t)C * (size_t)T);
    for (int c = 0; c < C; ++c) {
        const int8_t *src = y + (size_t)c * (size_t)T;
        for (int t = 0; t < T; ++t)
            y_tc[(size_t)t * (size_t)C + c] = src[t];
    }

    // 2) QKV into buf_qkv as three [T][nhead][dh] slices (TOKEN-MAJOR).
    int8_t *Qbase = buf_qkv;
    int8_t *Kbase = buf_qkv + (size_t)C * (size_t)T;
    int8_t *Vbase = buf_qkv + (size_t)2 * (size_t)C * (size_t)T;

    {
        const fc_flash_desc_t *fd_list[3] = {fd_q, fd_k, fd_v};
        int8_t *base_list[3] = {Qbase, Kbase, Vbase};

        int32_t acc_batch[FC_TOKEN_BATCH * FC_OUT_TILE];
        int8_t req_buf[FC_OUT_TILE];

        for (int qi = 0; qi < 3; ++qi) {
            const fc_flash_desc_t *fd = fd_list[qi];
            int8_t *qkv_base = base_list[qi];

            for (int o0 = 0; o0 < C; o0 += FC_OUT_TILE) {
                const int oN = (o0 + FC_OUT_TILE <= C) ? FC_OUT_TILE : (C - o0);
                const size_t w_bytes = (size_t)oN * (size_t)fd->Cin;

                fread(fd->off_b + (uint32_t)(o0 * 4), bbuf, (size_t)oN * 4);
                fread(fd->off_m + (uint32_t)(o0 * 4), mbuf, (size_t)oN * 4);
                fread(fd->off_r + (uint32_t)(o0 * 4), rbuf, (size_t)oN * 4);
                fread(fd->off_w + (uint32_t)((size_t)o0 * (size_t)fd->Cin), wbuf0, w_bytes);

                // Batched: FC_TOKEN_BATCH tokens at a time
                int t = 0;
                for (; t + FC_TOKEN_BATCH <= T; t += FC_TOKEN_BATCH) {
                    fc_core_outtile_batch(
                        &y_tc[(size_t)t * (size_t)C], C, C, FC_TOKEN_BATCH,
                        wbuf0, oN, bbuf, acc_batch);
                    for (int n = 0; n < FC_TOKEN_BATCH; ++n) {
                        fc_requant_outtile(acc_batch + n * FC_OUT_TILE, mbuf, rbuf, oN, C, req_buf);
                        __builtin_memcpy(&qkv_base[(size_t)(t + n) * (size_t)C + o0], req_buf, (size_t)oN);
                    }
                }
                // Remainder tokens
                for (; t < T; ++t) {
                    fc_core_outtile_one_row(&y_tc[(size_t)t * (size_t)C], C, wbuf0, oN, bbuf, acc_tile);
                    fc_requant_outtile(acc_tile, mbuf, rbuf, oN, C, req_buf);
                    __builtin_memcpy(&qkv_base[(size_t)t * (size_t)C + o0], req_buf, (size_t)oN);
                }
            }
        }
    }

    // 3–4) Attention per head.
    int32_t *acc_vec = (int32_t *)alloca((size_t)dh * sizeof(int32_t));

    for (int h = 0; h < nhead; ++h) {
        if (win_len == T) {
            // ---- Global attention ----
            for (int t = 0; t < T; ++t) {
                const int8_t *q_t = mhsa__ptr_q(Qbase, nhead, T, dh, t, h);

                for (int j = 0; j < T; ++j) {
                    const int8_t *k_j = mhsa__ptr_k(Kbase, nhead, T, dh, j, h);
                    scores_row[j] = mhsa__dot_i8_i8_contig(q_t, k_j, dh);
                }
                softmax_row_q15_bins(scores_row, (size_t)T, softmax_mul_bins, SM_SOFTMAX_INTBINS_RSHIFT, probs_q15);

                // [C] V accumulation with 4-way unrolled packed loads
                for (int d = 0; d < dh; ++d) acc_vec[d] = 0;

                for (int j = 0; j < T; ++j) {
                    const int8_t *v_j = mhsa__ptr_v(Vbase, nhead, T, dh, j, h);
                    const int32_t p = (int32_t)probs_q15[j];
                    int d = 0;
                    for (; d + 3 < dh; d += 4) {
                        const uint32_t vp = xheep_load_u32_unaligned(v_j + d);
                        acc_vec[d + 0] += p * (int32_t)(int8_t)(vp);
                        acc_vec[d + 1] += p * (int32_t)(int8_t)(vp >> 8);
                        acc_vec[d + 2] += p * (int32_t)(int8_t)(vp >> 16);
                        acc_vec[d + 3] += p * (int32_t)(int8_t)(vp >> 24);
                    }
                    for (; d < dh; ++d)
                        acc_vec[d] += p * (int32_t)v_j[d];
                }

                // Write context to y_tc[T,C] — sequential writes (no stride-T scatter)
                int8_t *ctx_row = &y_tc[(size_t)t * (size_t)C + (size_t)h * (size_t)dh];
                for (int d = 0; d < dh; ++d) {
                    int32_t a = acc_vec[d] >> 15;
                    if (a > 127) a = 127; else if (a < -128) a = -128;
                    ctx_row[d] = (int8_t)a;
                }
            }
        } else {
            // ---- Windowed attention ----
            const int wrad = window_radius;
            for (int t = 0; t < T; ++t) {
                const int8_t *q_t = mhsa__ptr_q(Qbase, nhead, T, dh, t, h);

                for (int jj = 0; jj < win_len; ++jj) {
                    const int j = t - wrad + jj;
                    if (j < 0 || j >= T) { scores_row[jj] = 0; }
                    else {
                        const int8_t *k_j = mhsa__ptr_k(Kbase, nhead, T, dh, j, h);
                        scores_row[jj] = mhsa__dot_i8_i8_contig(q_t, k_j, dh);
                    }
                }
                softmax_row_q15_bins(scores_row, (size_t)win_len, softmax_mul_bins, SM_SOFTMAX_INTBINS_RSHIFT, probs_q15);

                for (int d = 0; d < dh; ++d) acc_vec[d] = 0;

                for (int jj = 0; jj < win_len; ++jj) {
                    const int j = t - wrad + jj;
                    if ((unsigned)j < (unsigned)T) {
                        const int8_t *v_j = mhsa__ptr_v(Vbase, nhead, T, dh, j, h);
                        const int32_t p = (int32_t)probs_q15[jj];
                        int d = 0;
                        for (; d + 3 < dh; d += 4) {
                            const uint32_t vp = xheep_load_u32_unaligned(v_j + d);
                            acc_vec[d + 0] += p * (int32_t)(int8_t)(vp);
                            acc_vec[d + 1] += p * (int32_t)(int8_t)(vp >> 8);
                            acc_vec[d + 2] += p * (int32_t)(int8_t)(vp >> 16);
                            acc_vec[d + 3] += p * (int32_t)(int8_t)(vp >> 24);
                        }
                        for (; d < dh; ++d)
                            acc_vec[d] += p * (int32_t)v_j[d];
                    }
                }

                int8_t *ctx_row = &y_tc[(size_t)t * (size_t)C + (size_t)h * (size_t)dh];
                for (int d = 0; d < dh; ++d) {
                    int32_t a = acc_vec[d] >> 15;
                    if (a > 127) a = 127; else if (a < -128) a = -128;
                    ctx_row[d] = (int8_t)a;
                }
            }
        }
    }

    // 5) Output projection: y_tc[T,C] context → y[C,T] + residual x.
    {
        int32_t acc_batch[FC_TOKEN_BATCH * FC_OUT_TILE];
        int8_t req_buf[FC_OUT_TILE];

        for (int o0 = 0; o0 < C; o0 += FC_OUT_TILE) {
            const int oN = (o0 + FC_OUT_TILE <= C) ? FC_OUT_TILE : (C - o0);
            const size_t w_bytes = (size_t)oN * (size_t)fd_out->Cin;

            fread(fd_out->off_b + (uint32_t)(o0 * 4), bbuf, (size_t)oN * 4);
            fread(fd_out->off_m + (uint32_t)(o0 * 4), mbuf, (size_t)oN * 4);
            fread(fd_out->off_r + (uint32_t)(o0 * 4), rbuf, (size_t)oN * 4);
            fread(fd_out->off_w + (uint32_t)((size_t)o0 * (size_t)fd_out->Cin), wbuf0, w_bytes);

            int t = 0;
            for (; t + FC_TOKEN_BATCH <= T; t += FC_TOKEN_BATCH) {
                fc_core_outtile_batch(
                    &y_tc[(size_t)t * (size_t)C], C, C, FC_TOKEN_BATCH,
                    wbuf0, oN, bbuf, acc_batch);
                for (int n = 0; n < FC_TOKEN_BATCH; ++n) {
                    fc_requant_outtile(acc_batch + n * FC_OUT_TILE, mbuf, rbuf, oN, C, req_buf);
                    for (int k = 0; k < oN; ++k) {
                        const int cidx = o0 + k;
                        const size_t pos = (size_t)cidx * (size_t)T + (size_t)(t + n);
                        const int16_t s = (int16_t)req_buf[k] + (int16_t)x[pos];
                        y[pos] = SATURATE_INT8(s);
                    }
                }
            }
            // Remainder tokens
            for (; t < T; ++t) {
                fc_core_outtile_one_row(&y_tc[(size_t)t * (size_t)C], C, wbuf0, oN, bbuf, acc_tile);
                fc_requant_outtile(acc_tile, mbuf, rbuf, oN, C, req_buf);
                for (int k = 0; k < oN; ++k) {
                    const int cidx = o0 + k;
                    const size_t pos = (size_t)cidx * (size_t)T + (size_t)t;
                    const int16_t s = (int16_t)req_buf[k] + (int16_t)x[pos];
                    y[pos] = SATURATE_INT8(s);
                }
            }
        }
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MHSA_FLASH_H_
