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
