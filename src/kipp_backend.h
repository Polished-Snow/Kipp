#ifndef KIPP_BACKEND_H
#define KIPP_BACKEND_H

#include "kipp.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Qwen3-32B has 64 layers; the layer-view array is sized for the family. */
#define KIPP_MAX_BLOCK_COUNT 64u

/*
 * One supported, pinned checkpoint. The compiled-in registry
 * (src/kipp_checkpoints.h) is the only growth point for model support: a
 * GGUF whose (repository, revision) pair is not in the registry, or whose
 * metadata differs from its registry entry in any field, is rejected.
 */
typedef struct {
    const char *id;         /* stable identifier, e.g. "qwen3-4b-base" */
    const char *repository; /* pinned Hugging Face repository */
    const char *revision;   /* pinned immutable revision */
    kipp_variant variant;
    uint32_t block_count;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t context_length;
    float rope_theta;
    bool tied_embeddings; /* false => a separate lm_head.weight is required */
    uint32_t eos_token_id;
    uint32_t stop_tokens[KIPP_MAX_STOP_TOKENS];
    uint32_t stop_token_count;
} kipp_checkpoint_spec;

/*
 * Weight quantization scheme for a model artifact. Applies only to the seven
 * per-layer projection tensors; token embedding, lm_head, and every norm
 * stay BF16. The scheme is recorded in GGUF metadata (kipp.quant.scheme) and
 * validated against the per-tensor types in the tensor directory.
 */
typedef enum {
    KIPP_QUANT_BF16 = 0,
    KIPP_QUANT_Q8_0 = 1,
    KIPP_QUANT_AFFINE4_GS32 = 2
} kipp_quant_scheme;

/*
 * Validated per-model dimensions, derived from the matched registry entry.
 * attention_width is head_count * head_dim and is NOT embedding_length:
 * they differ at 0.6B (2048 vs 1024) and 32B (8192 vs 5120).
 */
typedef struct {
    const kipp_checkpoint_spec *spec;
    uint32_t block_count;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t attention_width;
    uint32_t context_length;
    float rope_theta;
    bool tied_embeddings;
    kipp_quant_scheme quant_scheme;
    /*
     * Pooled KV: when nonzero, the backend model owns one shared slab of
     * this many 32-position blocks and pooled sessions map their block
     * tables onto it. 0 keeps today's per-session slabs.
     */
    uint32_t kv_pool_blocks;
} kipp_model_config;

/*
 * ggml type ids for BF16 (30) and Q8_0 (8); AFFINE4_GS32 is a Kipp-private
 * id. Q8_0 packs 32 weights as {fp16 scale; int8 qs[32]} = 34 bytes;
 * AFFINE4_GS32 packs 32 as {uint8 nibbles[16]; fp16 scale; fp16 bias} = 20
 * bytes with w = scale*q + bias.
 */
typedef enum {
    KIPP_TENSOR_Q8_0 = 8,
    KIPP_TENSOR_BF16 = 30,
    KIPP_TENSOR_AFFINE4_GS32 = 1000
} kipp_tensor_type;

#define KIPP_QUANT_BLOCK 32u
#define KIPP_Q8_0_BLOCK_BYTES 34u
#define KIPP_AFFINE4_GROUP_BYTES 20u

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
    kipp_qwen3_layer_weights layers[KIPP_MAX_BLOCK_COUNT];
    kipp_tensor_view output_norm;
    /* Output projection: aliases token_embedding when embeddings are tied,
     * otherwise binds the checkpoint's separate lm_head.weight tensor. */
    kipp_tensor_view lm_head;
} kipp_qwen3_weights;

typedef struct {
    const uint8_t *mapping;
    uint64_t mapping_size;
    kipp_model_config config;
    kipp_qwen3_weights weights;
} kipp_model_view;

typedef struct {
    void *session;
    const uint32_t *tokens;
    uint32_t token_count;
    uint32_t start_position;
    float *logits;
    /*
     * Number of trailing tokens to write logits for (1..token_count). Row r
     * (0-based) holds the logits for token index token_count-logits_count+r,
     * at logits + r*KIPP_VOCAB_SIZE. 1 = only the final token (the common
     * case). Enables speculative verify and prompt logprobs.
     */
    uint32_t logits_count;
    /*
     * Pooled sessions only: logical-to-physical block map into the model's
     * pooled slab, valid for this call, covering every block up to
     * ceil((start_position + token_count) / 32) entries. NULL for sessions
     * with private slabs (their identity table lives in the backend).
     */
    const uint32_t *block_table;
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
