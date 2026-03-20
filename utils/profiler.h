#pragma once
#ifndef TINYTRANSFORMER_PROFILER_H_
#define TINYTRANSFORMER_PROFILER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*
 * TinyTransformer generic profiler (header-only, no printf here)
 *
 * - Types (ProfSectionId, ProfEntry, Profiler) are always defined.
 * - Actual timing code is enabled only if ENABLE_PROFILER is defined.
 * - Global helper macros use PROFILER_USE_GLOBAL + extern Profiler g_profiler.
 */

/* Max number of blocks we expose individually */
#ifndef PROF_MAX_BLOCKS
#define PROF_MAX_BLOCKS 4
#endif

/* Section IDs — must match what transformer.h uses */
typedef enum
{
    PROF_CONVSTEM = 0,
    PROF_POSMIX,

    /* Per-block MHSA (0..3) */
    PROF_MHSA_0,
    PROF_MHSA_1,

    /* Per-block MLP (0..3) */
    PROF_MLP_0,
    PROF_MLP_1,

    PROF_FINAL_LN,        /* Final transformer LN before AttnPool   */
    PROF_ATTENPOOL,       /* AttnPool module                        */
    PROF_CLASSIFIER,      /* Classifier (LN + FC)                   */

    PROF_SECTION_COUNT    /* must be last */
} ProfSectionId;

/* Single section statistics */
typedef struct
{
    uint64_t total_ticks; /* accumulated ticks over all calls        */
    uint32_t calls;       /* number of times this section was called */

    uint64_t last_start;  /* internal: last start tick                */
    uint8_t  active;      /* internal: 1 if currently running         */
} ProfEntry;

/* Profiler container */
typedef struct
{
    ProfEntry sections[PROF_SECTION_COUNT];
} Profiler;

/* Time source: you must implement this in some .c file:
 *     uint64_t prof_now(void);
 * or define PROF_NOW() yourself before including this header.
 */
#ifndef PROF_NOW
uint64_t prof_now(void);
#define PROF_NOW() prof_now()
#endif

/* Reset all statistics */
static inline void profiler_reset(Profiler *p)
{
#ifdef ENABLE_PROFILER
    if (!p) return;
    for (int i = 0; i < PROF_SECTION_COUNT; ++i)
    {
        p->sections[i].total_ticks = 0;
        p->sections[i].calls       = 0;
        p->sections[i].last_start  = 0;
        p->sections[i].active      = 0;
    }
#else
    (void)p;
#endif
}

/* Begin measurement for a given section */
static inline void profiler_begin(Profiler *p, ProfSectionId id)
{
#ifdef ENABLE_PROFILER
    if (!p) return;
    if (id < 0 || id >= PROF_SECTION_COUNT) return;

    ProfEntry *e = &p->sections[id];
    if (e->active) return; /* ignore nested starts */
    e->active     = 1;
    e->last_start = PROF_NOW();
#else
    (void)p;
    (void)id;
#endif
}

/* End measurement for a given section */
static inline void profiler_end(Profiler *p, ProfSectionId id)
{
#ifdef ENABLE_PROFILER
    if (!p) return;
    if (id < 0 || id >= PROF_SECTION_COUNT) return;

    ProfEntry *e = &p->sections[id];
    if (!e->active) return;

    uint64_t now = PROF_NOW();
    e->active = 0;
    e->calls += 1;
    e->total_ticks += (now - e->last_start);
#else
    (void)p;
    (void)id;
#endif
}

/* Optional global-style helpers:
 *
 *  - Define PROFILER_USE_GLOBAL and declare in exactly one C file:
 *        Profiler g_profiler;
 *
 *  - Then use:
 *        PROF_RESET_GLOBAL();
 *        PROF_BEGIN_SECTION(PROF_CONVSTEM);
 *        PROF_END_SECTION(PROF_CONVSTEM);
 */
#ifdef PROFILER_USE_GLOBAL
extern Profiler g_profiler;

#define PROF_RESET_GLOBAL()            profiler_reset(&g_profiler)
#define PROF_BEGIN_SECTION(sec_id)     profiler_begin(&g_profiler, (sec_id))
#define PROF_END_SECTION(sec_id)       profiler_end(&g_profiler, (sec_id))

#else /* !PROFILER_USE_GLOBAL */

#define PROF_RESET_GLOBAL()            ((void)0)
#define PROF_BEGIN_SECTION(sec_id)     ((void)(sec_id))
#define PROF_END_SECTION(sec_id)       ((void)(sec_id))

#endif /* PROFILER_USE_GLOBAL */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TINYTRANSFORMER_PROFILER_H_ */
