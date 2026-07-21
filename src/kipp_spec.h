#ifndef KIPP_SPEC_H
#define KIPP_SPEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Prompt-lookup draft generation for self-speculative decoding. Finds the
 * most recent earlier occurrence of the tail n-gram of `history` and proposes
 * the tokens that followed it. Tries n-gram lengths from max_ngram down to
 * min_ngram and uses the longest that matches (longer context = a more
 * reliable guess). Writes up to max_draft proposed tokens to `drafts` and
 * returns the count (0 when nothing matches). No model, no allocation.
 */
size_t kipp_spec_prompt_lookup(const uint32_t *history, size_t history_len,
                               uint32_t max_ngram, uint32_t min_ngram,
                               size_t max_draft, uint32_t *drafts);

/*
 * Adaptive drafting gate. On a bandwidth-bound engine a k-token verify costs
 * close to k+1 plain decode steps, so drafting pays only when a good fraction
 * of each draft is accepted; sustained low acceptance makes speculation a net
 * loss. The gate tracks an EMA of per-verify acceptance and suspends drafting
 * when it falls below a threshold, then probes periodically so a later
 * repetitive stretch can re-enable it. Purely local state; the caller decides
 * what a "committed token" is (one emitted token).
 *
 * The EMA decay keeps roughly two to three verify steps of memory: two zero-
 * acceptance drafts push a fresh gate below the off threshold, which bounds
 * the wasted work on hopeless text to about two verify steps per probe
 * window. A probe that accepts at least half its draft switches drafting
 * back on and the EMA restarts from that probe's acceptance (the old regime's
 * history is stale by definition).
 */
#define KIPP_SPEC_GATE_EMA_DECAY 0.5f
#define KIPP_SPEC_GATE_OFF 0.20f
#define KIPP_SPEC_GATE_PROBE_ON 0.50f
#define KIPP_SPEC_GATE_PROBE_INTERVAL 32u

typedef struct {
    float ema;                   /* EMA of accepted/drafted per verify step */
    int active;                  /* drafting currently enabled */
    uint32_t tokens_since_probe; /* committed tokens while suspended */
} kipp_spec_gate;

/* Start enabled with a neutral EMA. */
void kipp_spec_gate_init(kipp_spec_gate *gate);

/* Should the caller draft this step (either active, or due for a probe)? */
int kipp_spec_gate_should_draft(const kipp_spec_gate *gate);

/* Record a verify step's outcome; drafted must be nonzero. */
void kipp_spec_gate_record(kipp_spec_gate *gate, uint32_t drafted,
                           uint32_t accepted);

/* Record one committed token on a step that did not draft. */
void kipp_spec_gate_tick(kipp_spec_gate *gate);

#ifdef __cplusplus
}
#endif

#endif
