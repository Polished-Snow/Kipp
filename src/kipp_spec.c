#include "kipp_spec.h"

#include <string.h>

size_t kipp_spec_prompt_lookup(const uint32_t *history, size_t history_len,
                               uint32_t max_ngram, uint32_t min_ngram,
                               size_t max_draft, uint32_t *drafts) {
    if (history == NULL || drafts == NULL || max_draft == 0 ||
        history_len == 0 || min_ngram == 0 || max_ngram < min_ngram) {
        return 0;
    }
    if ((size_t)max_ngram >= history_len) {
        max_ngram = (uint32_t)(history_len - 1);
    }
    for (uint32_t ngram = max_ngram; ngram >= min_ngram; --ngram) {
        if ((size_t)ngram >= history_len) {
            continue;
        }
        const uint32_t *needle = history + history_len - ngram;
        /* Most recent earlier occurrence: scan right-to-left over the
         * positions where the n-gram could start before the tail. */
        for (size_t start = history_len - ngram; start-- > 0;) {
            if (memcmp(history + start, needle,
                       (size_t)ngram * sizeof(*history)) != 0) {
                continue;
            }
            size_t after = start + ngram;
            size_t available = history_len - after;
            size_t count = available < max_draft ? available : max_draft;
            if (count == 0) {
                break; /* match sits at the very end; try a shorter n-gram */
            }
            memcpy(drafts, history + after, count * sizeof(*drafts));
            return count;
        }
    }
    return 0;
}

void kipp_spec_gate_init(kipp_spec_gate *gate) {
    gate->ema = 0.5f;
    gate->active = 1;
    gate->tokens_since_probe = 0;
}

int kipp_spec_gate_should_draft(const kipp_spec_gate *gate) {
    return gate->active ||
           gate->tokens_since_probe >= KIPP_SPEC_GATE_PROBE_INTERVAL;
}

void kipp_spec_gate_record(kipp_spec_gate *gate, uint32_t drafted,
                           uint32_t accepted) {
    if (drafted == 0) {
        return;
    }
    float fraction = (float)accepted / (float)drafted;
    if (gate->active) {
        gate->ema = KIPP_SPEC_GATE_EMA_DECAY * gate->ema +
                    (1.0f - KIPP_SPEC_GATE_EMA_DECAY) * fraction;
        if (gate->ema < KIPP_SPEC_GATE_OFF) {
            gate->active = 0;
        }
    } else {
        /* Probe step: re-enable on a strong result and restart the EMA from
         * it — the suspended regime's history is stale either way. */
        if (fraction >= KIPP_SPEC_GATE_PROBE_ON) {
            gate->active = 1;
            gate->ema = fraction;
        }
    }
    gate->tokens_since_probe = 0;
}

void kipp_spec_gate_tick(kipp_spec_gate *gate) {
    if (!gate->active) {
        ++gate->tokens_since_probe;
    }
}
