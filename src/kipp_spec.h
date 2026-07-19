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

#ifdef __cplusplus
}
#endif

#endif
