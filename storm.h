#pragma once
#ifndef TINYTRANSFORMER_WRAPPER_H_
#define TINYTRANSFORMER_WRAPPER_H_

#ifdef __cplusplus
extern "C"
{
#endif

// ---- External dependencies ----
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "build_config.h"

#if USE_FLASH
#include "model_flash.h"
#include "flash_reader.h"
#else
#include "model.h"
#endif

#include "profiler.h"

#include "mod1_convstem.h"
#include "mod2_posmix.h"
#include "mod3_mhsa.h"
#include "mod4_mlp.h"
#include "mod5_attnpool.h"
#include "mod6_classifier.h"

#include "ker_layernorm.h"
#include "ker_linear.h"

#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

    // -----------------------------------------------------------------------------
    // Unified status codes (negative = failure, 0 = success).
    // Each distinct failure path has a unique value so main.c can diagnose precisely.
    // -----------------------------------------------------------------------------
    typedef enum
    {
        OK = 0,
        // Args / geometry
        ERR_BAD_ARGS = -100,
        ERR_ACT_SCALES_COUNT_MISMATCH = -101,
        ERR_CONV_TOUT_MISMATCH = -102,

        // FLASH preload: LayerNorm params (granular per symbol for precise diagnosis)
        ERR_RD_LN_BLK0_MHA_GAMMA = -200,
        ERR_RD_LN_BLK0_MHA_BETA = -201,
        ERR_RD_LN_BLK0_MLP_GAMMA = -202,
        ERR_RD_LN_BLK0_MLP_BETA = -203,
        ERR_RD_LN_BLK1_MHA_GAMMA = -204,
        ERR_RD_LN_BLK1_MHA_BETA = -205,
        ERR_RD_LN_BLK1_MLP_GAMMA = -206,
        ERR_RD_LN_BLK1_MLP_BETA = -207,
        ERR_RD_LN_BLK2_MHA_GAMMA = -208,
        ERR_RD_LN_BLK2_MHA_BETA = -209,
        ERR_RD_LN_BLK2_MLP_GAMMA = -210,
        ERR_RD_LN_BLK2_MLP_BETA = -211,
        ERR_RD_LN_BLK3_MHA_GAMMA = -212,
        ERR_RD_LN_BLK3_MHA_BETA = -213,
        ERR_RD_LN_BLK3_MLP_GAMMA = -214,
        ERR_RD_LN_BLK3_MLP_BETA = -215,
        ERR_RD_LN_FINAL_GAMMA = -216,
        ERR_RD_LN_FINAL_BETA = -217,
        ERR_RD_LN_AP_GAMMA = -218,
        ERR_RD_LN_AP_BETA = -219,
        ERR_RD_LN_CLS_GAMMA = -220,
        ERR_RD_LN_CLS_BETA = -221,

        // FLASH preload: activation scales
        ERR_RD_ACT_SCALES = -230,

        // FLASH preload: integer LayerNorm rsqrt LUT
        ERR_RD_LN_RSQRT_LUT = -240,

        // AttnPool streamed (map of its internal rc)
        ERR_AP_BAD_ARGS = -300,
        ERR_AP_RD_FC0_B = -301,
        ERR_AP_RD_FC0_M = -302,
        ERR_AP_RD_FC0_R = -303,
        ERR_AP_RD_FC0_W = -304,
        ERR_AP_RD_FC1_B = -305,
        ERR_AP_RD_FC1_M = -306,
        ERR_AP_RD_FC1_R = -307,
        ERR_AP_RD_FC1_W = -308,
        ERR_AP_UNKNOWN = -309,

        // Classifier streamed FC
        ERR_CLS_BAD_ARGS = -320,
        ERR_CLS_RD_BIAS = -321,
        ERR_CLS_RD_M = -322,
        ERR_CLS_RD_R = -323,
        ERR_CLS_RD_WEIGHTS = -324
    } Status;

    // Helper to map attnpool_forward_i8_streamed rc -> Status
    static inline Status map_attnpool_rc(int rc)
    {
        switch (rc)
        {
        case 0:
            return OK;
        case -1:
            return ERR_AP_BAD_ARGS;
        case -2:
            return ERR_AP_RD_FC0_B;
        case -3:
            return ERR_AP_RD_FC0_M;
        case -4:
            return ERR_AP_RD_FC0_R;
        case -5:
            return ERR_AP_RD_FC0_W;
        case -6:
            return ERR_AP_RD_FC1_B;
        case -7:
            return ERR_AP_RD_FC1_M;
        case -8:
            return ERR_AP_RD_FC1_R;
        case -9:
            return ERR_AP_RD_FC1_W;
        default:
            return ERR_AP_UNKNOWN;
        }
    }

    // -----------------------------------------------------------------------------
    // Flash reader adapter
    // -----------------------------------------------------------------------------
#if USE_FLASH
    static inline int flash_read_adapter(uint32_t off, void *dst, size_t nbytes)
    {
        // flash_read_bytes returns 0 on success
        return flash_read_bytes((const void *)off, dst, nbytes);
    }

    static inline int flash_reader_trampoline(void *user, uint32_t off, void *dst, size_t nbytes)
    {
        (void)user;
        return flash_read_adapter(off, dst, nbytes);
    }

    static inline WeightReader make_flash_reader(void)
    {
        WeightReader r;
        r.user = NULL;
        r.read = flash_reader_trampoline;
        return r;
    }
#else
    // RAM reader: when USE_FLASH=0, data is static in RAM.
    // The "offsets" are actually direct pointers to RAM data from model.h.
    static inline int ram_read_adapter(uint32_t off, void *dst, size_t nbytes)
    {
        // Simple memcpy: the offset IS the source address in static RAM
        const void *src = (const void *)(uintptr_t)off;
        memcpy(dst, src, nbytes);
        return 0; // success
    }

    static inline int ram_reader_trampoline(void *user, uint32_t off, void *dst, size_t nbytes)
    {
        (void)user;
        return ram_read_adapter(off, dst, nbytes);
    }

    // When USE_FLASH=0, provide flash_read_bytes as an inline wrapper
    // This function is called by preload functions and must work with RAM pointers
    static inline int flash_read_bytes(const void *src_addr, void *dst, size_t nbytes)
    {
        // src_addr is a pointer to static const data in model.h
        memcpy(dst, src_addr, nbytes);
        return 0; // success
    }

    static inline WeightReader make_flash_reader(void)
    {
        // Return a RAM reader when USE_FLASH=0
        WeightReader r;
        r.user = NULL;
        r.read = ram_reader_trampoline;
        return r;
    }
#endif

    // -----------------------------------------------------------------------------
    // Activation scales (Q31) — SRAM cache to avoid XIP on each access
    // -----------------------------------------------------------------------------
    enum
    {
        ACT_SCALES_COUNT = (int)(sizeof(g_act_scales_q31) / sizeof(g_act_scales_q31[0]))
    };
    static int32_t g_act_scales_q31_sram[ACT_SCALES_COUNT];

    // Access to exported activation scales (Q31 -> float) from SRAM cache
    static inline float act_scale_from_q31_idx(int idx)
    {
        if (idx < 0 || idx >= ACT_SCALES_COUNT)
        {
            // Defensive: return 0 if out-of-range
            return 0.0f;
        }
        const uint32_t q = (uint32_t)g_act_scales_q31_sram[idx];
        const float inv = 1.0f / 2147483648.0f; // 2^31
        return (float)q * inv;
    }

// Indexing pattern from exporter (deterministic):
#define S_STEM_OUT_IDX (0)
#define S_POSMIX_OUT_IDX (1)
#define S_BLK_MHA_OUT_IDX(i) (2 + 2 * (i))
#define S_BLK_MLP_OUT_IDX(i) (3 + 2 * (i))
#define S_FINAL_NORM_OUT_IDX (2 + 2 * MODEL_DEPTH)
#define S_ATTNSCORE_IDX (3 + 2 * MODEL_DEPTH)
#define S_CLS_IN_IDX (4 + 2 * MODEL_DEPTH)
#define S_CLS_OUT_IDX (5 + 2 * MODEL_DEPTH)

// Check presence of exported activation/LUT params and softmax scales
#ifndef SILU_ALPHA
#error "SILU_* macros missing in model.h. Regenerate exporter with activation params."
#endif
#ifndef GELU_ALPHA
#error "GELU_* macros missing in model.h. Regenerate exporter with activation params."
#endif
#ifndef AP_GELU_ALPHA
#error "AP_GELU_* macros missing in model.h. Regenerate exporter with activation params."
#endif
#ifndef MHSA_SOFTMAX_SCALE
#error "MHSA_SOFTMAX_SCALE missing in model.h. Regenerate exporter with softmax scale."
#endif
#ifndef AP_SOFTMAX_SCALE
#error "AP_SOFTMAX_SCALE missing in model.h. Regenerate exporter with AP softmax scale."
#endif

    // -----------------------------------------------------------------------------
    // LayerNorm params — one-time SRAM cache (to avoid XIP/flash reads inside LN)
    // -----------------------------------------------------------------------------
    static float g_ln_blk_mha_gamma[MODEL_DEPTH][MODEL_D_MODEL];
    static float g_ln_blk_mha_beta[MODEL_DEPTH][MODEL_D_MODEL];
    static float g_ln_blk_mlp_gamma[MODEL_DEPTH][MODEL_D_MODEL];
    static float g_ln_blk_mlp_beta[MODEL_DEPTH][MODEL_D_MODEL];
    static float g_ln_final_gamma[MODEL_D_MODEL];
    static float g_ln_final_beta[MODEL_D_MODEL];
    static float g_ln_ap_gamma[MODEL_D_MODEL];
    static float g_ln_ap_beta[MODEL_D_MODEL];
    static float g_ln_cls_gamma[MODEL_D_MODEL];
    static float g_ln_cls_beta[MODEL_D_MODEL];

    // Preload LN params
    static inline Status preload_ln_params_to_sram(void)
    {
        // This function works for both USE_FLASH=1 and USE_FLASH=0:
        // - When USE_FLASH=1: flash_read_bytes reads from external flash
        // - When USE_FLASH=0: flash_read_bytes (ram_read_adapter) reads from static RAM arrays in model.h
        // Block LNs
#if MODEL_DEPTH >= 1
        if (flash_read_bytes(ln_blk0_mha_ln_gamma, g_ln_blk_mha_gamma[0], sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_BLK0_MHA_GAMMA;
        if (flash_read_bytes(ln_blk0_mha_ln_beta,  g_ln_blk_mha_beta[0],  sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_BLK0_MHA_BETA;
        if (flash_read_bytes(ln_blk0_mlp_ln_gamma, g_ln_blk_mlp_gamma[0], sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_BLK0_MLP_GAMMA;
        if (flash_read_bytes(ln_blk0_mlp_ln_beta,  g_ln_blk_mlp_beta[0],  sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_BLK0_MLP_BETA;
#endif
#if MODEL_DEPTH >= 2
        if (flash_read_bytes(ln_blk1_mha_ln_gamma, g_ln_blk_mha_gamma[1], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK1_MHA_GAMMA;
        if (flash_read_bytes(ln_blk1_mha_ln_beta, g_ln_blk_mha_beta[1], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK1_MHA_BETA;
        if (flash_read_bytes(ln_blk1_mlp_ln_gamma, g_ln_blk_mlp_gamma[1], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK1_MLP_GAMMA;
        if (flash_read_bytes(ln_blk1_mlp_ln_beta, g_ln_blk_mlp_beta[1], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK1_MLP_BETA;
#endif
#if MODEL_DEPTH >= 3
        if (flash_read_bytes(ln_blk2_mha_ln_gamma, g_ln_blk_mha_gamma[2], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK2_MHA_GAMMA;
        if (flash_read_bytes(ln_blk2_mha_ln_beta, g_ln_blk_mha_beta[2], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK2_MHA_BETA;
        if (flash_read_bytes(ln_blk2_mlp_ln_gamma, g_ln_blk_mlp_gamma[2], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK2_MLP_GAMMA;
        if (flash_read_bytes(ln_blk2_mlp_ln_beta, g_ln_blk_mlp_beta[2], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK2_MLP_BETA;
#endif
#if MODEL_DEPTH >= 4
        if (flash_read_bytes(ln_blk3_mha_ln_gamma, g_ln_blk_mha_gamma[3], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK3_MHA_GAMMA;
        if (flash_read_bytes(ln_blk3_mha_ln_beta, g_ln_blk_mha_beta[3], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK3_MHA_BETA;
        if (flash_read_bytes(ln_blk3_mlp_ln_gamma, g_ln_blk_mlp_gamma[3], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK3_MLP_GAMMA;
        if (flash_read_bytes(ln_blk3_mlp_ln_beta, g_ln_blk_mlp_beta[3], sizeof(float) * MODEL_D_MODEL)) return ERR_RD_LN_BLK3_MLP_BETA;
#endif
        // Final / AttnPool / Classifier
        if (flash_read_bytes(ln_final_norm_gamma,   g_ln_final_gamma, sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_FINAL_GAMMA;
        if (flash_read_bytes(ln_final_norm_beta,    g_ln_final_beta,  sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_FINAL_BETA;
        if (flash_read_bytes(ln_attn_pool_ln_gamma, g_ln_ap_gamma,    sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_AP_GAMMA;
        if (flash_read_bytes(ln_attn_pool_ln_beta,  g_ln_ap_beta,     sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_AP_BETA;
        if (flash_read_bytes(ln_classifier_ln_gamma,g_ln_cls_gamma,   sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_CLS_GAMMA;
        if (flash_read_bytes(ln_classifier_ln_beta, g_ln_cls_beta,    sizeof(float)*MODEL_D_MODEL)) return ERR_RD_LN_CLS_BETA;
        return OK;
    }

    // ---- Integer LayerNorm: rsqrt LUT SRAM cache + init ----
#if USE_INT_LAYERNORM && defined(LN_INT_ENABLED) && LN_INT_ENABLED
    static float g_ln_rsqrt_lut_sram[LN_INT_RSQRT_LUT_SIZE];

    static inline Status preload_rsqrt_lut_to_sram(void)
    {
        if (flash_read_bytes(ln_rsqrt_lut, g_ln_rsqrt_lut_sram, sizeof(float) * LN_INT_RSQRT_LUT_SIZE)) {
            return ERR_RD_LN_RSQRT_LUT;
        }
        kerln_int_lut_init(g_ln_rsqrt_lut_sram, LN_INT_RSQRT_LUT_SIZE, LN_INT_RSQRT_LOG_MIN, LN_INT_RSQRT_LOG_RANGE);
        return OK;
    }
#endif

    static inline Status preload_act_scales_to_sram(void)
    {
        /* Copy all activation scales from source (FLASH or RAM) once */
        if (flash_read_bytes(g_act_scales_q31, g_act_scales_q31_sram, sizeof(g_act_scales_q31)))
            return ERR_RD_ACT_SCALES;
        return OK;
    }

    // Optional helper: preload all constants we want in SRAM up-front
    static inline Status transformer_preload_constants_to_sram(void)
    {
        Status s;
        s = preload_ln_params_to_sram();      if (s != OK) return s;
        s = preload_act_scales_to_sram();     if (s != OK) return s;
#if USE_INT_LAYERNORM && defined(LN_INT_ENABLED) && LN_INT_ENABLED
        s = preload_rsqrt_lut_to_sram();      if (s != OK) return s;
#endif
        return OK;
    }

    // -----------------------------------------------------------------------------
    // FC / Conv descriptors (offset = absolute symbol address)
    // -----------------------------------------------------------------------------
    static inline void fc_desc_fill(fc_flash_desc_t *d, int Cin, int Cout, const int8_t *W, const int32_t *B, const int32_t *M, const int32_t *R)
    {
        d->Cin = Cin;
        d->Cout = Cout;
        d->off_w = (uint32_t)(uintptr_t)W;
        d->off_b = (uint32_t)(uintptr_t)B;
        d->off_m = (uint32_t)(uintptr_t)M;
        d->off_r = (uint32_t)(uintptr_t)R;
    }

    static inline void conv_pw_desc_fill(conv1d_pw_flash_desc_t *d, int Cin, int Cout, int K, int stride, int pad, int dil, const int8_t *W, const int32_t *B, const int32_t *M, const int32_t *R)
    {
        d->Cin = Cin;
        d->Cout = Cout;
        d->K = K;
        d->stride = stride;
        d->pad = pad;
        d->dil = dil;
        d->off_w = (uint32_t)(uintptr_t)W;
        d->off_b = (uint32_t)(uintptr_t)B;
        d->off_m = (uint32_t)(uintptr_t)M;
        d->off_r = (uint32_t)(uintptr_t)R;
    }

    static inline void conv_dw_desc_fill(conv1d_dw_flash_desc_t *d, int C, int K, int stride, int pad, int dil, const int8_t *W, const int32_t *B, const int32_t *M_to_in, const int32_t *R_to_in)
    {
        d->C = C;
        d->K = K;
        d->stride = stride;
        d->pad = pad;
        d->dil = dil;
        d->off_w = (uint32_t)(uintptr_t)W;
        d->off_b = (uint32_t)(uintptr_t)B;
        d->off_m = (uint32_t)(uintptr_t)M_to_in;
        d->off_r = (uint32_t)(uintptr_t)R_to_in;
    }

    // -----------------------------------------------------------------------------
    // LN parameter getters
    // -----------------------------------------------------------------------------
    static inline void get_ln_mha_params(int i, const float **gamma, const float **beta, float *eps)
    {
        switch (i)
        {
#if MODEL_DEPTH >= 1
        case 0:
            *gamma = g_ln_blk_mha_gamma[0];
            *beta = g_ln_blk_mha_beta[0];
            *eps = LN_BLK0_MHA_LN_EPS;
            return;
#endif
#if MODEL_DEPTH >= 2
        case 1:
            *gamma = g_ln_blk_mha_gamma[1];
            *beta = g_ln_blk_mha_beta[1];
            *eps = LN_BLK1_MHA_LN_EPS;
            return;
#endif
#if MODEL_DEPTH >= 3
        case 2:
            *gamma = g_ln_blk_mha_gamma[2];
            *beta = g_ln_blk_mha_beta[2];
            *eps = LN_BLK2_MHA_LN_EPS;
            return;
#endif
#if MODEL_DEPTH >= 4
        case 3:
            *gamma = g_ln_blk_mha_gamma[3];
            *beta = g_ln_blk_mha_beta[3];
            *eps = LN_BLK3_MHA_LN_EPS;
            return;
#endif
        default:
            break;
        }
        *gamma = NULL;
        *beta = NULL;
        *eps = 1e-5f;
    }

    static inline void get_ln_mlp_params(int i, const float **gamma, const float **beta, float *eps)
    {
        switch (i)
        {
#if MODEL_DEPTH >= 1
        case 0:
            *gamma = g_ln_blk_mlp_gamma[0];
            *beta = g_ln_blk_mlp_beta[0];
            *eps = LN_BLK0_MLP_LN_EPS;
            return;
#endif
#if MODEL_DEPTH >= 2
        case 1:
            *gamma = g_ln_blk_mlp_gamma[1];
            *beta = g_ln_blk_mlp_beta[1];
            *eps = LN_BLK1_MLP_LN_EPS;
            return;
#endif
#if MODEL_DEPTH >= 3
        case 2:
            *gamma = g_ln_blk_mlp_gamma[2];
            *beta = g_ln_blk_mlp_beta[2];
            *eps = LN_BLK2_MLP_LN_EPS;
            return;
#endif
#if MODEL_DEPTH >= 4
        case 3:
            *gamma = g_ln_blk_mlp_gamma[3];
            *beta = g_ln_blk_mlp_beta[3];
            *eps = LN_BLK3_MLP_LN_EPS;
            return;
#endif
        default:
            break;
        }
        *gamma = NULL;
        *beta = NULL;
        *eps = 1e-5f;
    }

    static inline void get_ln_final_params(const float **gamma, const float **beta, float *eps)
    {
        *gamma = g_ln_final_gamma;
        *beta = g_ln_final_beta;
        *eps = LN_FINAL_NORM_EPS;
    }
    static inline void get_ln_attnpool_params(const float **g, const float **b, float *e)
    {
        *g = g_ln_ap_gamma;
        *b = g_ln_ap_beta;
        *e = LN_ATTN_POOL_LN_EPS;
    }
    static inline void get_ln_classifier_params(const float **g, const float **b, float *e)
    {
        *g = g_ln_cls_gamma;
        *b = g_ln_cls_beta;
        *e = LN_CLASSIFIER_LN_EPS;
    }

    // -----------------------------------------------------------------------------
    // FC index helpers
    // -----------------------------------------------------------------------------
    static inline void fc_by_index(fc_flash_desc_t *d, int idx)
    {
        // Layout: 6 FCs per block [Q, K, V, proj, mlp.fc1, mlp.fc2]
        //   block i: indices 6*i .. 6*i+5
        //   then: attnpool.fc0, attnpool.fc1, classifier
        switch (idx)
        {
#if MODEL_DEPTH >= 1
        // Block 0: Q=0, K=1, V=2, proj=3, mlp.fc1=4, mlp.fc2=5
        case 0:
            fc_desc_fill(d, FC0_IN, FC0_OUT, fc0_W, fc0_B, fc0_M, fc0_R);
            break;
        case 1:
            fc_desc_fill(d, FC1_IN, FC1_OUT, fc1_W, fc1_B, fc1_M, fc1_R);
            break;
        case 2:
            fc_desc_fill(d, FC2_IN, FC2_OUT, fc2_W, fc2_B, fc2_M, fc2_R);
            break;
        case 3:
            fc_desc_fill(d, FC3_IN, FC3_OUT, fc3_W, fc3_B, fc3_M, fc3_R);
            break;
        case 4:
            fc_desc_fill(d, FC4_IN, FC4_OUT, fc4_W, fc4_B, fc4_M, fc4_R);
            break;
        case 5:
            fc_desc_fill(d, FC5_IN, FC5_OUT, fc5_W, fc5_B, fc5_M, fc5_R);
            break;
#endif
#if MODEL_DEPTH >= 2
        // Block 1: Q=6, K=7, V=8, proj=9, mlp.fc1=10, mlp.fc2=11
        case 6:
            fc_desc_fill(d, FC6_IN, FC6_OUT, fc6_W, fc6_B, fc6_M, fc6_R);
            break;
        case 7:
            fc_desc_fill(d, FC7_IN, FC7_OUT, fc7_W, fc7_B, fc7_M, fc7_R);
            break;
        case 8:
            fc_desc_fill(d, FC8_IN, FC8_OUT, fc8_W, fc8_B, fc8_M, fc8_R);
            break;
        case 9:
            fc_desc_fill(d, FC9_IN, FC9_OUT, fc9_W, fc9_B, fc9_M, fc9_R);
            break;
        case 10:
            fc_desc_fill(d, FC10_IN, FC10_OUT, fc10_W, fc10_B, fc10_M, fc10_R);
            break;
        case 11:
            fc_desc_fill(d, FC11_IN, FC11_OUT, fc11_W, fc11_B, fc11_M, fc11_R);
            break;
#endif
#if MODEL_DEPTH >= 3
        // Block 2: Q=12, K=13, V=14, proj=15, mlp.fc1=16, mlp.fc2=17
        case 12:
            fc_desc_fill(d, FC12_IN, FC12_OUT, fc12_W, fc12_B, fc12_M, fc12_R);
            break;
        case 13:
            fc_desc_fill(d, FC13_IN, FC13_OUT, fc13_W, fc13_B, fc13_M, fc13_R);
            break;
        case 14:
            fc_desc_fill(d, FC14_IN, FC14_OUT, fc14_W, fc14_B, fc14_M, fc14_R);
            break;
        case 15:
            fc_desc_fill(d, FC15_IN, FC15_OUT, fc15_W, fc15_B, fc15_M, fc15_R);
            break;
        case 16:
            fc_desc_fill(d, FC16_IN, FC16_OUT, fc16_W, fc16_B, fc16_M, fc16_R);
            break;
        case 17:
            fc_desc_fill(d, FC17_IN, FC17_OUT, fc17_W, fc17_B, fc17_M, fc17_R);
            break;
#endif
#if MODEL_DEPTH >= 4
        // Block 3: Q=18, K=19, V=20, proj=21, mlp.fc1=22, mlp.fc2=23
        case 18:
            fc_desc_fill(d, FC18_IN, FC18_OUT, fc18_W, fc18_B, fc18_M, fc18_R);
            break;
        case 19:
            fc_desc_fill(d, FC19_IN, FC19_OUT, fc19_W, fc19_B, fc19_M, fc19_R);
            break;
        case 20:
            fc_desc_fill(d, FC20_IN, FC20_OUT, fc20_W, fc20_B, fc20_M, fc20_R);
            break;
        case 21:
            fc_desc_fill(d, FC21_IN, FC21_OUT, fc21_W, fc21_B, fc21_M, fc21_R);
            break;
        case 22:
            fc_desc_fill(d, FC22_IN, FC22_OUT, fc22_W, fc22_B, fc22_M, fc22_R);
            break;
        case 23:
            fc_desc_fill(d, FC23_IN, FC23_OUT, fc23_W, fc23_B, fc23_M, fc23_R);
            break;
#endif
        default: /* unreachable */
            break;
        }
    }

    // --- Tail FC helpers (attnpool + classifier) ---
    // These sit at indices 6*MODEL_DEPTH+{0,1,2} which vary with depth,
    // so we resolve them directly via preprocessor token pasting.
#define _FC_PASTE3(a,b,c)  a##b##c
#define _FC_PASTE(a,b,c)   _FC_PASTE3(a,b,c)

#define _FC_TAIL_BASE  (6 * MODEL_DEPTH)
#define _FC_TAIL_ID(off)  (_FC_TAIL_BASE + (off))

    // Expand fcN_W / fcN_IN / etc. for the tail indices
#define _FC_TAIL_SYM(prefix, off, suffix)  _FC_PASTE(prefix, _FC_TAIL_ID(off), suffix)

    // Unfortunately C token-pasting cannot paste computed integers.
    // Use explicit #if chains instead.

#if MODEL_DEPTH == 1
#define _FC_AP0_IDX  6
#define _FC_AP1_IDX  7
#define _FC_CLS_IDX  8
#elif MODEL_DEPTH == 2
#define _FC_AP0_IDX  12
#define _FC_AP1_IDX  13
#define _FC_CLS_IDX  14
#elif MODEL_DEPTH == 3
#define _FC_AP0_IDX  18
#define _FC_AP1_IDX  19
#define _FC_CLS_IDX  20
#elif MODEL_DEPTH == 4
#define _FC_AP0_IDX  24
#define _FC_AP1_IDX  25
#define _FC_CLS_IDX  26
#endif

    // Token-paste helpers to build symbol names from an index literal
#define _FC_CAT2(a, b) a ## b
#define _FC_CAT(a, b)  _FC_CAT2(a, b)
#define _FC_W(n)   _FC_CAT(fc, _FC_CAT(n, _W))
#define _FC_B(n)   _FC_CAT(fc, _FC_CAT(n, _B))
#define _FC_M(n)   _FC_CAT(fc, _FC_CAT(n, _M))
#define _FC_R(n)   _FC_CAT(fc, _FC_CAT(n, _R))
#define _FC_IN(n)  _FC_CAT(FC, _FC_CAT(n, _IN))
#define _FC_OUT(n) _FC_CAT(FC, _FC_CAT(n, _OUT))

    static inline void fc_attn_fc0_desc(fc_flash_desc_t *d) {
        fc_desc_fill(d, _FC_IN(_FC_AP0_IDX), _FC_OUT(_FC_AP0_IDX),
                     _FC_W(_FC_AP0_IDX), _FC_B(_FC_AP0_IDX),
                     _FC_M(_FC_AP0_IDX), _FC_R(_FC_AP0_IDX));
    }
    static inline void fc_attn_fc1_desc(fc_flash_desc_t *d) {
        fc_desc_fill(d, _FC_IN(_FC_AP1_IDX), _FC_OUT(_FC_AP1_IDX),
                     _FC_W(_FC_AP1_IDX), _FC_B(_FC_AP1_IDX),
                     _FC_M(_FC_AP1_IDX), _FC_R(_FC_AP1_IDX));
    }
    static inline void fc_classifier_desc(fc_flash_desc_t *d) {
        fc_desc_fill(d, _FC_IN(_FC_CLS_IDX), _FC_OUT(_FC_CLS_IDX),
                     _FC_W(_FC_CLS_IDX), _FC_B(_FC_CLS_IDX),
                     _FC_M(_FC_CLS_IDX), _FC_R(_FC_CLS_IDX));
    }

    // --- Block FC helpers: 6 FCs per block [Q, K, V, proj, fc1, fc2] ---
    static inline void fc_q_desc(fc_flash_desc_t *d, int i) { fc_by_index(d, 6 * i + 0); }
    static inline void fc_k_desc(fc_flash_desc_t *d, int i) { fc_by_index(d, 6 * i + 1); }
    static inline void fc_v_desc(fc_flash_desc_t *d, int i) { fc_by_index(d, 6 * i + 2); }
    static inline void fc_proj_desc(fc_flash_desc_t *d, int i) { fc_by_index(d, 6 * i + 3); }
    static inline void fc_mlp_fc1_desc(fc_flash_desc_t *d, int i) { fc_by_index(d, 6 * i + 4); }
    static inline void fc_mlp_fc2_desc(fc_flash_desc_t *d, int i) { fc_by_index(d, 6 * i + 5); }

    // -----------------------------------------------------------------------------
    // User-supplied runtime params
    // -----------------------------------------------------------------------------
    typedef struct
    {
        // SiLU LUT mapping for ConvStem (affine index mapping into ker_silu)
        int32_t silu_alpha, silu_beta;
        int silu_rshift;

        // MHSA softmax: precomputed integer LUT bin scale (eliminates float from hot path)
        int32_t mhsa_softmax_mul_bins;

        // AttnPool softmax: precomputed integer LUT bin scale
        int32_t ap_softmax_mul_bins;

        // MLP GELU: LUT index mapping + Q15 -> int8 requant to FC2 input scale
        int32_t gelu_alpha, gelu_beta;
        int gelu_rshift;
        int32_t gelu_M_out, gelu_R_out;

        // AttnPool GELU (usually same mapping/requant as above if quantized likewise)
        int32_t ap_gelu_alpha, ap_gelu_beta;
        int ap_gelu_rshift;
        int32_t ap_gelu_M_out, ap_gelu_R_out;

        // Classifier LN scales (per-tensor): s_in_feat and s_after_ln (float)
        float cls_ln_s_in, cls_ln_s_out;
    } Params;

    // Convenience: fill Params directly from exporter macros / activation scales
    static inline void params_from_export_defaults(Params *p)
    {
        if (!p)
            return;
        // SiLU (stem)
        p->silu_alpha = SILU_ALPHA;
        p->silu_beta = SILU_BETA;
        p->silu_rshift = SILU_RSHIFT;
        // MLP GELU
        p->gelu_alpha = GELU_ALPHA;
        p->gelu_beta = GELU_BETA;
        p->gelu_rshift = GELU_RSHIFT;
        p->gelu_M_out = GELU_M_OUT;
        p->gelu_R_out = GELU_R_OUT;
        // AttnPool GELU
        p->ap_gelu_alpha = AP_GELU_ALPHA;
        p->ap_gelu_beta = AP_GELU_BETA;
        p->ap_gelu_rshift = AP_GELU_RSHIFT;
        p->ap_gelu_M_out = AP_GELU_M_OUT;
        p->ap_gelu_R_out = AP_GELU_R_OUT;
        // Softmax precomputed integer bin scales
        p->mhsa_softmax_mul_bins = MHSA_SOFTMAX_MUL_BINS;
        p->ap_softmax_mul_bins = AP_SOFTMAX_MUL_BINS;
        // Classifier LN per-tensor scales (float)
        p->cls_ln_s_in = act_scale_from_q31_idx(S_CLS_IN_IDX);
        p->cls_ln_s_out = p->cls_ln_s_in;
    }

    // -----------------------------------------------------------------------------
    // Workspace (caller allocates; we keep RAM small by reusing scratch)
    // -----------------------------------------------------------------------------
    typedef struct
    {
        // Ping-pong slabs for [C,T] tensors (int8). Size: MODEL_D_MODEL * T each.
        int8_t *slab_a_ct;
        int8_t *slab_b_ct;

        // Stem scratch
        int32_t *stem_acc_line; // [CONV1D_OUT_TILE]
        int8_t *stem_wbuf0;     // [CONV1D_OUT_TILE * CONV0_INC * CONV0_K]
        int8_t *stem_wbuf1;     // same as above (ping/pong)
        int32_t *stem_bbuf;     // [CONV1D_OUT_TILE]
        int32_t *stem_mbuf;     // [CONV1D_OUT_TILE]
        int32_t *stem_rbuf;     // [CONV1D_OUT_TILE]

        // PosMix scratch
        int32_t *pos_acc_scalar; // [1]
        int8_t *pos_kbuf;        // [CONV1_K]
        int32_t *pos_bmr;        // [3]

        // Shared FC tiles (MHSA/MLP/AttnPool)
        int32_t *acc_tile; // [FC_OUT_TILE]
        int8_t *wbuf0_fc;  // [FC_OUT_TILE * MODEL_D_MODEL]
        int8_t *wbuf1_fc;  // [FC_OUT_TILE * MODEL_D_MODEL]
        int32_t *bbuf_fc;  // [FC_OUT_TILE]
        int32_t *mbuf_fc;  // [FC_OUT_TILE]
        int32_t *rbuf_fc;  // [FC_OUT_TILE]

        // MHSA work
        int32_t *mhsa_scores_row; // [win_len]  (win_len = 2*w+1 or T)
        int16_t *mhsa_probs_q15;  // [win_len]
        int8_t *mhsa_buf_qkv;     // [3 * MODEL_D_MODEL * T]
        int8_t *mhsa_buf_tok;     // [MODEL_D_MODEL]   (used for gather/scatter)
        // For ctx we reuse slab_b_ct as [C,T].

        // MLP slabs
        int8_t *mlp_H1;  // [H * T] (you can alias H1A to H1 to save RAM)
        int8_t *mlp_H1A; // [H * T]
        int8_t *mlp_tok; // [max(C,H)]

        // AttnPool work
        int32_t *ap_logits_T;  // [T]
        int16_t *ap_probs_q15; // [T]
        // Reuse: ZLN -> slab_b_ct, H1/H1A -> mlp_H1/mlp_H1A

        // Classifier tiles
        int8_t *cls_w_tile;  // [OC_BLK * MODEL_D_MODEL]
        int32_t *cls_b_tile; // [OC_BLK]
        int8_t *cls_tmp_i8;  // [MODEL_D_MODEL]

        // Optional debug: pooled feature right after AttnPool (before classifier LN)
        // If NULL, no copy is performed.
        int8_t *cls_feat_preln_i8; // [MODEL_D_MODEL] or NULL
    } Workspace;

    // -----------------------------------------------------------------------------
    // Main forward
    //   x_in:   [MODEL_IN_CHANNELS, T] int8 (channel-major)
    //   T:      sequence length
    //   params: runtime params (LUT mappings and softmax scale)
    //   ws:     workspace (preallocated)
    //   logits: [MODEL_NUM_CLASSES] int32 output (requantized to classifier_out int8 domain)
    // -----------------------------------------------------------------------------
    static inline int transformer_forward_i8_flash(
        const int8_t *__restrict x_in, int T,
        const Params *__restrict params,
        Workspace *__restrict ws,
        int32_t *__restrict logits_out_i32)
    {
        if (UNLIKELY(!x_in || T <= 0 || !params || !ws || !logits_out_i32))
        {
            printf("transformer_forward_i8_flash: invalid NULL pointer or T<=0\n");
            return ERR_BAD_ARGS;
        }

        if (5 + 2 * MODEL_DEPTH + 1 != ACT_SCALES_COUNT)
        {
            printf("Activation scales count mismatch! Expected %d, got %d\n", 5 + 2 * MODEL_DEPTH + 1, ACT_SCALES_COUNT);
            return ERR_ACT_SCALES_COUNT_MISMATCH;
        }

#if USE_FLASH
        flash_read_fn fread = flash_read_adapter;
#else
        flash_read_fn fread = ram_read_adapter;
#endif

        // Fetch constants to SRAM (LN params + activation scales)
        // Default behavior preserves the current flow: load each inference.
#if PRELOAD_CONSTANTS_EACH_INFERENCE
        {
            Status s = transformer_preload_constants_to_sram();
            if (s != OK)
            {
                return s;
            }
        }
#else
        {
            static int preloaded = 0;
            if (!preloaded)
            {
                Status s = transformer_preload_constants_to_sram();
                if (s != OK)
                {
                    return s;
                }
                preloaded = 1;
            }
        }
#endif

        // ========= 0) Build descriptors from model.h =========
        // ConvStem
        conv1d_pw_flash_desc_t d_stem;
        conv_pw_desc_fill(&d_stem, CONV0_INC, CONV0_OUTC, CONV0_K, CONV0_STRIDE, CONV0_PAD, CONV0_DIL, conv0_W, conv0_B, conv0_M, conv0_R);

        // Positional mixing (depthwise)
        conv1d_dw_flash_desc_t d_pos;
        conv_dw_desc_fill(&d_pos, CONV1_OUTC /*==C*/, CONV1_K, CONV1_STRIDE, CONV1_PAD, CONV1_DIL, conv1_W, conv1_B, conv1_M, conv1_R);

        // FC descriptors are constructed per use via helpers (see below).

        // ========= 1) ConvStem =========
#ifdef ENABLE_PROFILER
        PROF_BEGIN_SECTION(PROF_CONVSTEM);
#endif
        // Output length from conv parameters:
        const int Tout = conv1d_out_len(T, CONV0_K, CONV0_STRIDE, CONV0_PAD, CONV0_DIL);
        if (UNLIKELY(Tout != T))
        {
            // The trained graph uses stride=1 and “same” padding; enforce equality.
            printf("ConvStem output length mismatch! Expected %d, got %d\n", T, Tout);
            return ERR_CONV_TOUT_MISMATCH;
        }

        // stem scales: SiLU LUT mapping provided by params; BN already folded.
        convstem_flash(
            /*x*/ x_in, /*Tin*/ T,
            /*desc*/ &d_stem, /*fread*/ fread,
            /*SiLU LUT map*/ params->silu_alpha, params->silu_beta, params->silu_rshift,
            /*y*/ ws->slab_a_ct,
            /*scratch*/ ws->stem_acc_line,
            ws->stem_wbuf0, ws->stem_wbuf1,
            ws->stem_bbuf, ws->stem_mbuf, ws->stem_rbuf);

#ifdef ENABLE_PROFILER
        PROF_END_SECTION(PROF_CONVSTEM);
#endif

        // ========= 2) Positional Mixing (depthwise conv) =========
#ifdef ENABLE_PROFILER
        PROF_BEGIN_SECTION(PROF_POSMIX);
#endif
        posmix_depthwise_residual_flash(
            /*x*/ ws->slab_a_ct, /*C*/ MODEL_D_MODEL, /*T*/ T,
            /*desc*/ &d_pos, /*fread*/ fread,
            /*y*/ ws->slab_b_ct,
            /*acc_scalar*/ ws->pos_acc_scalar,
            /*kbuf*/ ws->pos_kbuf,
            /*bmr*/ ws->pos_bmr);

#ifdef ENABLE_PROFILER
       PROF_END_SECTION(PROF_POSMIX);
#endif

        // Current tensor lives in slab_b_ct
        int8_t *cur = ws->slab_b_ct;
        int8_t *nxt = ws->slab_a_ct;

        // ========= 3) Transformer Blocks =========
        for (int i = 0; i < MODEL_DEPTH; ++i)
        {

#ifdef ENABLE_PROFILER
            ProfSectionId sec_mhsa = (ProfSectionId)(PROF_MHSA_0 + i);
            ProfSectionId sec_mlp  = (ProfSectionId)(PROF_MLP_0  + i);
#endif

            // ----- MHSA (pre-LN) -----
            // LN scales: in = previous activation scale; out = same (qkv exported with s_out = s_in)
            const float ln_mha_sx = (i == 0) ? act_scale_from_q31_idx(S_POSMIX_OUT_IDX) : act_scale_from_q31_idx(S_BLK_MLP_OUT_IDX(i - 1));
            const float ln_mha_sy = ln_mha_sx;
            const float *ln_mha_gamma = NULL;
            const float *ln_mha_beta = NULL;
            float ln_mha_eps = 1e-5f;
            get_ln_mha_params(i, &ln_mha_gamma, &ln_mha_beta, &ln_mha_eps);

#ifdef ENABLE_PROFILER
            PROF_BEGIN_SECTION(sec_mhsa);
#endif

            fc_flash_desc_t fd_q, fd_k, fd_v, fd_proj;
            fc_q_desc(&fd_q, i);
            fc_k_desc(&fd_k, i);
            fc_v_desc(&fd_v, i);
            fc_proj_desc(&fd_proj, i);

            // Use slab_b_ct for ctx (buf_ctx), and write the final MHSA+res into nxt
            int window_radius = MODEL_ATTENTION_WINDOW; // radius; if 0 or >=T => global
            mhsa_forward_i8_preln_flash(
                /*x*/ cur, /*C*/ MODEL_D_MODEL, /*T*/ T,
                /*nhead*/ MODEL_NHEAD, /*window_radius*/ window_radius,
                /*LN scales*/ ln_mha_sx, ln_mha_sy, ln_mha_eps,
                /*LN affine*/ ln_mha_gamma, ln_mha_beta,
                /*Q/K/V/OUT*/ &fd_q, &fd_k, &fd_v, &fd_proj, /*fread*/ fread,
                /*softmax*/ params->mhsa_softmax_mul_bins,
                /*work*/ ws->mhsa_scores_row,
                ws->mhsa_buf_qkv,
                ws->mhsa_buf_tok,
                ws->mhsa_buf_tok, // reuse as buf_tok_out
                ws->mhsa_probs_q15,
                ws->acc_tile,
                ws->wbuf0_fc, ws->wbuf1_fc,
                ws->bbuf_fc, ws->mbuf_fc, ws->rbuf_fc,
                /*output*/ nxt);

#ifdef ENABLE_PROFILER
            PROF_END_SECTION(sec_mhsa);
#endif

            // Swap so that “cur” holds MHSA output
            int8_t *tmp = cur;
            cur = nxt;
            nxt = tmp;

            // ----- MLP (pre-LN) -----
            const float ln_mlp_sx = act_scale_from_q31_idx(S_BLK_MHA_OUT_IDX(i));
            const float ln_mlp_sy = ln_mlp_sx; // FC1 exported with s_out = s_in
            const float *ln_mlp_gamma = NULL;
            const float *ln_mlp_beta = NULL;
            float ln_mlp_eps = 1e-5f;

#ifdef ENABLE_PROFILER
            PROF_BEGIN_SECTION(sec_mlp);
#endif

            get_ln_mlp_params(i, &ln_mlp_gamma, &ln_mlp_beta, &ln_mlp_eps);

            fc_flash_desc_t fd_fc1, fd_fc2;
            fc_mlp_fc1_desc(&fd_fc1, i);
            fc_mlp_fc2_desc(&fd_fc2, i);

            mlp_forward_i8_preln_flash(
                /*x*/ cur, /*C*/ MODEL_D_MODEL, /*T*/ T,
                /*H*/ fd_fc1.Cout,
                /*LN*/ ln_mlp_sx, ln_mlp_sy, ln_mlp_eps,
                /*gamma/beta*/ ln_mlp_gamma, ln_mlp_beta,
                /*FCs*/ &fd_fc1, &fd_fc2, /*fread*/ fread,
                /*GELU LUT*/ params->gelu_alpha, params->gelu_beta, params->gelu_rshift,
                /*GELU->FC2 in scale requant*/ params->gelu_M_out, params->gelu_R_out,
                /*work*/ ws->mlp_tok, ws->mlp_tok, // buf_tok_in/mid share the same max(C,H) vector
                ws->acc_tile,
                ws->wbuf0_fc, ws->wbuf1_fc,
                ws->bbuf_fc, ws->mbuf_fc, ws->rbuf_fc,
                /*slabs*/ ws->mlp_H1, ws->mlp_H1A,
                /*y*/ nxt);

#ifdef ENABLE_PROFILER
            PROF_END_SECTION(sec_mlp);
#endif

            // Residual is inside module; swap to make cur=MLP output
            tmp = cur;
            cur = nxt;
            nxt = tmp;
        }

        // ========= 4) Final LayerNorm =========
        const float *ln_fn_gamma = NULL, *ln_fn_beta = NULL;
        float ln_fn_eps = 1e-5f;
        get_ln_final_params(&ln_fn_gamma, &ln_fn_beta, &ln_fn_eps);
        const float ln_fn_sx = act_scale_from_q31_idx(S_BLK_MLP_OUT_IDX(MODEL_DEPTH - 1));
        const float ln_fn_sy = act_scale_from_q31_idx(S_FINAL_NORM_OUT_IDX);
        // Write ZLN into nxt

#ifdef ENABLE_PROFILER
        PROF_BEGIN_SECTION(PROF_FINAL_LN);
#endif

        layernorm_forward_f32_i8io(
            /*x*/ cur, /*C*/ MODEL_D_MODEL, /*T*/ T,
            /*scale_x*/ ln_fn_sx, ln_fn_gamma, ln_fn_beta, ln_fn_eps,
            /*scale_y*/ ln_fn_sy,
            /*y*/ nxt);

#ifdef ENABLE_PROFILER
        PROF_END_SECTION(PROF_FINAL_LN);
#endif

        // ========= 5) Attention Pooling =========
        fc_flash_desc_t fd_ap0, fd_ap1;
        fc_attn_fc0_desc(&fd_ap0);
        fc_attn_fc1_desc(&fd_ap1);
        const float *ln_ap_gamma = NULL, *ln_ap_beta = NULL;
        float ln_ap_eps = 1e-5f;
        get_ln_attnpool_params(&ln_ap_gamma, &ln_ap_beta, &ln_ap_eps);

#ifdef ENABLE_PROFILER
        PROF_BEGIN_SECTION(PROF_ATTENPOOL);
#endif

        int rc_ap = attnpool_forward_i8_streamed(
            /*Z*/ nxt, /*C*/ MODEL_D_MODEL, /*T*/ T,
            /*LN*/ ln_fn_sy, ln_fn_sy, ln_ap_eps, ln_ap_gamma, ln_ap_beta,
            /*FC0*/ &fd_ap0, /*H*/ fd_ap0.Cout,
            /*GELU LUT*/ params->ap_gelu_alpha, params->ap_gelu_beta, params->ap_gelu_rshift,
            /*Q15->int8 to FC1 input*/ params->ap_gelu_M_out, params->ap_gelu_R_out,
            /*FC1*/ &fd_ap1,
            /*softmax scale (T)*/ params->ap_softmax_mul_bins,
            /*optional per-channel feat requant*/ NULL, NULL,
            /*FLASH*/ fread,
            /*shared tiles*/ ws->wbuf0_fc, ws->wbuf1_fc,
            ws->acc_tile, ws->bbuf_fc, ws->mbuf_fc, ws->rbuf_fc,
            /*logits_T*/ ws->ap_logits_T, /*probs_q15*/ ws->ap_probs_q15,
            /*slabs*/ cur, ws->mlp_H1, ws->mlp_H1A,
            /*dbg*/ NULL, NULL,
            /*out feat*/ ws->cls_tmp_i8 /*reuse as feat[d]*/);
        if (rc_ap)
            return map_attnpool_rc(rc_ap);

        // Optional debug snapshot: preserve pooled feature before classifier LN overwrites it.
        if (ws->cls_feat_preln_i8) {
            memcpy(ws->cls_feat_preln_i8, ws->cls_tmp_i8, (size_t)MODEL_D_MODEL);
        }

#ifdef ENABLE_PROFILER
        PROF_END_SECTION(PROF_ATTENPOOL);
#endif

        // ======== 6) Classifier ========
        {

#ifdef ENABLE_PROFILER
            PROF_BEGIN_SECTION(PROF_CLASSIFIER);
#endif

            // 1) LayerNorm on the pooled feature (in-place over ws->cls_tmp_i8)
            float cls_sx = params->cls_ln_s_in;
            float cls_sy = params->cls_ln_s_out;
            if (cls_sx == 0.0f) cls_sx = act_scale_from_q31_idx(S_CLS_IN_IDX);
            if (cls_sy == 0.0f) cls_sy = cls_sx;
            layernorm_forward_f32_i8io(
                /*x*/ ws->cls_tmp_i8, /*C*/ MODEL_D_MODEL, /*T*/ 1,
                /*scale_x*/ cls_sx,
                /*gamma*/ g_ln_cls_gamma,
                /*beta*/ g_ln_cls_beta,
                /*eps*/ LN_CLASSIFIER_LN_EPS,
                /*scale_y*/ cls_sy,
                /*y*/ ws->cls_tmp_i8);

            // 2) Get the classifier FC offsets
            fc_flash_desc_t fd_cls;
            fc_classifier_desc(&fd_cls); // fills Cin, Cout, off_w, off_b, off_m, off_r

            // 3) Streamed FC (weights from flash) + per-class requant -> int8 logits
            // IMPORTANT: classifier weights are quantized per-out-channel, so raw int32 accumulators
            // are not directly comparable across classes. We requant to a common int8 domain
            // (exported as classifier_out scale) before returning/logging/argmax.
            //
            // This replaces the previous ker_gemm streamed GEMM with ker_linear kernels.
            // Output tile is fixed to 32 to match ws->cls_* buffers.
            const int Cin = MODEL_D_MODEL;
            const int Cout = MODEL_NUM_CLASSES;
            const int OC_BLK = 32;
            int8_t* logits_i8 = ws->mhsa_buf_tok; // must be >= Cout

            int rc_cls = 0;
            for (int o0 = 0; o0 < Cout; o0 += OC_BLK) {
                const int oN = (o0 + OC_BLK <= Cout) ? OC_BLK : (Cout - o0);

                // Bias tile (optional)
                if (fd_cls.off_b) {
                    if (fread(fd_cls.off_b + (uint32_t)(o0 * 4), ws->cls_b_tile, (size_t)oN * 4)) { rc_cls = -3; break; }
                } else {
                    for (int j = 0; j < oN; ++j) ws->cls_b_tile[j] = 0;
                }

                // Per-channel requant params
                if (fread(fd_cls.off_m + (uint32_t)(o0 * 4), ws->mbuf_fc, (size_t)oN * 4)) { rc_cls = -4; break; }
                if (fread(fd_cls.off_r + (uint32_t)(o0 * 4), ws->rbuf_fc, (size_t)oN * 4)) { rc_cls = -5; break; }

                // Weights tile: [oN, Cin] row-major
                if (fread(fd_cls.off_w + (uint32_t)((size_t)o0 * (size_t)Cin), ws->cls_w_tile, (size_t)oN * (size_t)Cin)) { rc_cls = -6; break; }

                // Accumulate + requant for this segment
                fc_core_outtile_one_row(
                    /*x*/ ws->cls_tmp_i8, /*Cin*/ Cin,
                    /*w_tile*/ ws->cls_w_tile, /*oN*/ oN,
                    /*b_tile*/ ws->cls_b_tile,
                    /*acc*/ ws->acc_tile);
                fc_requant_outtile(
                    /*acc*/ ws->acc_tile,
                    /*m_tile*/ ws->mbuf_fc,
                    /*r_tile*/ ws->rbuf_fc,
                    /*oN*/ oN,
                    /*Cin*/ Cin,
                    /*y_row_seg*/ logits_i8 + o0);
            }

            if (rc_cls == 0) {
                for (int n = 0; n < MODEL_NUM_CLASSES; ++n)
                    logits_out_i32[n] = (int32_t)logits_i8[n];
            }

#ifdef ENABLE_PROFILER
            PROF_END_SECTION(PROF_CLASSIFIER);
#endif

            if (rc_cls != 0) {
                // Map streamed-classifier rc to unified codes
                if (rc_cls == -3) return ERR_CLS_RD_BIAS;
                if (rc_cls == -4) return ERR_CLS_RD_M;
                if (rc_cls == -5) return ERR_CLS_RD_R;
                if (rc_cls == -6) return ERR_CLS_RD_WEIGHTS;
                return ERR_CLS_BAD_ARGS;
            }
            return OK; // success
        }
    }

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TINYTRANSFORMER_WRAPPER_H_
