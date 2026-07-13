#ifndef KIPP_BACKEND_H
#define KIPP_BACKEND_H

#include "kipp.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    KIPP_TENSOR_BF16 = 30
} kipp_tensor_type;

typedef struct {
    const char *name;
    const uint8_t *data;
    uint64_t dimensions[2];
    uint32_t dimension_count;
    kipp_tensor_type type;
    uint64_t byte_count;
} kipp_tensor_view;

typedef struct {
    kipp_tensor_view attention_norm;
    kipp_tensor_view attention_q;
    kipp_tensor_view attention_k;
    kipp_tensor_view attention_v;
    kipp_tensor_view attention_output;
    kipp_tensor_view attention_q_norm;
    kipp_tensor_view attention_k_norm;
    kipp_tensor_view feed_forward_norm;
    kipp_tensor_view feed_forward_gate;
    kipp_tensor_view feed_forward_up;
    kipp_tensor_view feed_forward_down;
} kipp_qwen3_layer_weights;

typedef struct {
    kipp_tensor_view token_embedding;
    kipp_qwen3_layer_weights layers[KIPP_BLOCK_COUNT];
    kipp_tensor_view output_norm;
} kipp_qwen3_weights;

typedef struct {
    const uint8_t *mapping;
    uint64_t mapping_size;
    kipp_qwen3_weights weights;
} kipp_model_view;

typedef struct {
    void *session;
    const uint32_t *tokens;
    uint32_t token_count;
    uint32_t start_position;
    float *logits;
} kipp_eval_item;

typedef struct {
    int (*model_create)(const kipp_model_view *model, void **backend_model,
                        kipp_error *error);
    void (*model_destroy)(void *backend_model);
    int (*session_create)(void *backend_model, uint32_t capacity,
                          void **backend_session, kipp_error *error);
    void (*session_destroy)(void *backend_session);
    int (*session_reset)(void *backend_session, kipp_error *error);
    int (*eval)(void *backend_model, kipp_eval_item *items, size_t item_count,
                kipp_error *error);
    /*
     * Optional: keep only the first `length` cached tokens so evaluation can
     * resume from that position. Backends that leave this NULL report
     * truncation as unsupported.
     */
    int (*session_truncate)(void *backend_session, uint32_t length,
                            kipp_error *error);
} kipp_backend_ops;

#endif
