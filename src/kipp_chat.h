#ifndef KIPP_CHAT_H
#define KIPP_CHAT_H

#include "kipp.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Native Qwen3 ChatML rendering. The whole dense family shares one ChatML
 * template; the only per-variant difference is the assistant generation
 * prompt's thinking suffix, so this is hand-coded rather than driven by a
 * general template engine. Rendering is gated byte-for-byte against Hugging
 * Face's apply_chat_template output (tools/generate_chat_vectors.py).
 *
 * Tool calling and think-block history stripping are out of scope for this
 * first cut: content is treated as plain text.
 */

typedef enum {
    KIPP_ROLE_SYSTEM = 0,
    KIPP_ROLE_USER = 1,
    KIPP_ROLE_ASSISTANT = 2
} kipp_chat_role;

typedef struct {
    kipp_chat_role role;
    const char *content;
} kipp_chat_message;

typedef struct {
    kipp_variant variant;
    bool add_generation_prompt;
    /* Honored only by the hybrid instruct variant (KIPP_VARIANT_INSTRUCT):
     * false injects an empty <think></think> block after the assistant
     * prompt to suppress reasoning. Ignored by the 2507 variants and base. */
    bool enable_thinking;
} kipp_chat_options;

/*
 * Render messages into a Qwen3 ChatML prompt. On success writes a malloc'd
 * NUL-terminated string to *out_text (caller frees with kipp_text_free) and
 * returns 0. Base checkpoints have no chat template and return
 * KIPP_ERROR_UNSUPPORTED. A NULL/invalid argument returns KIPP_ERROR_ARGUMENT.
 */
int kipp_chat_render(const kipp_chat_message *messages, size_t message_count,
                     const kipp_chat_options *options, char **out_text,
                     kipp_error *error);

#ifdef __cplusplus
}
#endif

#endif
