#ifndef KIPP_H
#define KIPP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KIPP_VERSION "0.0.2"

/*
 * Family-invariant Qwen3 dense dimensions. Everything that varies between
 * checkpoints (layer count, hidden width, feed-forward width, query heads,
 * context length, RoPE theta, embedding tying, stop tokens) lives in the
 * compiled-in checkpoint registry and is reported via kipp_model_info.
 */
#define KIPP_ATTENTION_HEAD_COUNT_KV 8u
#define KIPP_ATTENTION_HEAD_DIM 128u
#define KIPP_VOCAB_SIZE 151936u
#define KIPP_ENDOFTEXT_TOKEN_ID 151643u
#define KIPP_IM_END_TOKEN_ID 151645u
#define KIPP_MAX_STOP_TOKENS 2u

typedef struct kipp_model kipp_model;
typedef struct kipp_session kipp_session;

typedef enum {
    KIPP_VARIANT_BASE = 0,          /* pretrained; stops on <|endoftext|> */
    KIPP_VARIANT_INSTRUCT = 1,      /* hybrid-thinking instruct (Qwen3-X) */
    KIPP_VARIANT_INSTRUCT_2507 = 2, /* non-thinking 2507 instruct */
    KIPP_VARIANT_THINKING_2507 = 3  /* thinking-only 2507 */
} kipp_variant;

typedef enum {
    KIPP_BACKEND_CPU = 0,
    KIPP_BACKEND_METAL = 1,
    KIPP_BACKEND_CUDA = 2
} kipp_backend_kind;

typedef enum {
    KIPP_OK = 0,
    KIPP_ERROR_ARGUMENT,
    KIPP_ERROR_IO,
    KIPP_ERROR_FORMAT,
    KIPP_ERROR_UNSUPPORTED,
    KIPP_ERROR_MEMORY,
    KIPP_ERROR_RANGE,
    KIPP_ERROR_INTERNAL
} kipp_error_code;

typedef struct {
    kipp_error_code code;
    char message[256];
} kipp_error;

typedef struct {
    uint32_t *data;
    size_t count;
} kipp_tokens;

typedef struct {
    const char *checkpoint_id; /* registry id, e.g. "qwen3-4b-base" */
    const char *repository;
    const char *revision;
    kipp_variant variant;
    kipp_backend_kind backend;
    uint32_t block_count;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t attention_head_dim;
    uint32_t vocab_size;
    uint32_t context_length;
    float rope_theta;
    int tied_embeddings;
    const char *quant_scheme; /* "bf16", "q8_0", or "affine4_gs32" */
    /* Generation should stop when it samples any of these tokens. */
    uint32_t stop_tokens[KIPP_MAX_STOP_TOKENS];
    uint32_t stop_token_count;
    uint64_t mapped_bytes;
} kipp_model_info;

typedef struct {
    uint32_t capacity;
    uint32_t length;
    uint64_t cache_bytes;
} kipp_session_info;

/*
 * Open a strict Kipp GGUF artifact for a registered Qwen3 dense checkpoint.
 * The returned model owns a read-only mmap until kipp_model_close().
 */
int kipp_model_open(const char *path, kipp_model **out_model, kipp_error *error);
int kipp_model_open_backend(const char *path, kipp_backend_kind backend,
                            kipp_model **out_model, kipp_error *error);
/*
 * Like kipp_model_open_backend, but backs every session with one shared KV
 * slab of kv_pool_blocks 32-token blocks and enables cross-session prefix
 * sharing: a finished session's full blocks are published to a
 * content-addressed pool, and a later session can adopt a matching prefix
 * with kipp_session_match_prefix instead of re-evaluating it. CPU and Metal;
 * CUDA returns KIPP_ERROR_UNSUPPORTED. kv_pool_blocks must be nonzero.
 */
int kipp_model_open_pooled(const char *path, kipp_backend_kind backend,
                           uint32_t kv_pool_blocks, kipp_model **out_model,
                           kipp_error *error);
int kipp_model_close(kipp_model *model, kipp_error *error);
int kipp_model_get_info(const kipp_model *model, kipp_model_info *out_info);
/* Nonzero when generation should stop after sampling this token. */
int kipp_model_is_stop_token(const kipp_model *model, uint32_t token);
const char *kipp_backend_name(kipp_backend_kind backend);

/*
 * Tokenization is model-owned and Qwen-specific. The output allocation belongs
 * to the caller and must be released with kipp_tokens_free().
 */
int kipp_tokenize(const kipp_model *model, const char *text,
                  kipp_tokens *out_tokens, kipp_error *error);
int kipp_detokenize(const kipp_model *model, const uint32_t *tokens,
                    size_t token_count, char **out_text, size_t *out_length,
                    kipp_error *error);
void kipp_tokens_free(kipp_tokens *tokens);
void kipp_text_free(char *text);

/*
 * Stateless evaluation of at most 256 tokens on the model's backend. It
 * computes full FP32 logits for the final token; logits_count must be at
 * least KIPP_VOCAB_SIZE. Session evaluation below is the efficient path.
 */
int kipp_model_eval(const kipp_model *model, const uint32_t *tokens,
                    size_t token_count, float *logits, size_t logits_count,
                    kipp_error *error);

/*
 * A Phase 2 session owns one bounded, independent BF16 KV timeline. Eval
 * appends all supplied tokens and writes logits for the final token.
 */
int kipp_session_create(kipp_model *model, uint32_t capacity,
                        kipp_session **out_session, kipp_error *error);
int kipp_session_reset(kipp_session *session, kipp_error *error);
/*
 * Keep only the first `length` tokens of the session timeline; subsequent
 * evaluation resumes from that position. Enables prefix reuse when a new
 * prompt shares a prefix with the session's history. Backends without
 * truncation support return KIPP_ERROR_UNSUPPORTED.
 */
int kipp_session_truncate(kipp_session *session, uint32_t length,
                          kipp_error *error);
void kipp_session_destroy(kipp_session *session);
int kipp_session_get_info(const kipp_session *session,
                          kipp_session_info *out_info);
int kipp_session_eval(kipp_session *session, const uint32_t *tokens,
                      size_t token_count, float *logits, size_t logits_count,
                      kipp_error *error);

/*
 * Like kipp_session_eval, but writes full FP32 logits for the last `rows`
 * supplied tokens (1 <= rows <= token_count) instead of just the final one.
 * `logits` must hold rows * KIPP_VOCAB_SIZE floats; row r holds the logits
 * for token index token_count-rows+r. Backends without multi-row support
 * return KIPP_ERROR_UNSUPPORTED for rows > 1. This is the primitive behind
 * speculative-decode verification and prompt logprobs.
 */
int kipp_session_eval_n(kipp_session *session, const uint32_t *tokens,
                        size_t token_count, float *logits, uint32_t rows,
                        kipp_error *error);

/*
 * On a fresh (length 0) session of a pooled model: adopt the longest
 * published prefix of `tokens` from the pool — whole 32-token blocks only,
 * capped one token short of token_count so evaluation always has a token
 * left to produce logits. Evaluation then resumes at *matched_tokens.
 * Sessions of non-pooled models return KIPP_ERROR_UNSUPPORTED.
 */
int kipp_session_match_prefix(kipp_session *session, const uint32_t *tokens,
                              size_t token_count, uint32_t *matched_tokens,
                              kipp_error *error);

/* Pool occupancy and reuse counters for a pooled model. */
typedef struct kipp_kv_pool_stats_public {
    uint32_t total_blocks;
    uint32_t free_blocks;
    uint64_t reused_blocks_total;
    uint64_t evicted_blocks_total;
} kipp_kv_pool_stats_public;

int kipp_model_kv_pool_stats(const kipp_model *model,
                             kipp_kv_pool_stats_public *out_stats);

/*
 * Evaluate several sessions in one backend call. Each item appends its
 * tokens to its own session and receives full FP32 logits for its final
 * token. Sessions must be distinct and belong to the same model. Backends
 * may interleave the work so weights are read once per step across items;
 * results must match isolated evaluation within the backend's documented
 * tolerance. At most KIPP_EVAL_BATCH_LIMIT items per call.
 */
#define KIPP_EVAL_BATCH_LIMIT 32u

typedef struct {
    kipp_session *session;
    const uint32_t *tokens;
    size_t token_count;
    float *logits; /* at least KIPP_VOCAB_SIZE values */
} kipp_batch_item;

int kipp_eval_batch(kipp_model *model, kipp_batch_item *items,
                    size_t item_count, kipp_error *error);

/*
 * Sample one token from full FP32 logits. A temperature at or below zero
 * selects the argmax and never touches rng_state. Otherwise logits are
 * scaled by 1/temperature, converted to probabilities, truncated to the
 * smallest set whose cumulative probability reaches top_p, and sampled with
 * the caller-owned deterministic RNG state, which must be nonzero.
 */
int kipp_sample(const float *logits, size_t logits_count, float temperature,
                float top_p, uint64_t *rng_state, uint32_t *out_token,
                kipp_error *error);

/*
 * Full sampling parameters. Defaults that disable a stage: top_k = 0,
 * min_p = 0, all penalties = 0 (repetition_penalty may also be 1 to disable),
 * recent_count = 0, logit_bias_count = 0. temperature <= 0 forces argmax
 * (over penalty/bias-adjusted logits) and ignores every randomness stage.
 *
 * Ordering matches the common local-inference chain: logit_bias, then the
 * OpenAI/CTRL-style penalties over the recent-token window, then temperature
 * scaling, then top_k, min_p, and top_p filtering, then the draw.
 */
typedef struct {
    float temperature;
    float top_p;               /* nucleus mass in (0, 1]; 1 disables */
    uint32_t top_k;            /* keep k highest; 0 disables */
    float min_p;               /* keep prob >= min_p * max_prob; 0 disables */
    float frequency_penalty;   /* OpenAI: logit -= count * penalty */
    float presence_penalty;    /* OpenAI: logit -= penalty if present */
    float repetition_penalty;  /* CTRL/HF: divide/mul by penalty; 1 disables */
    const uint32_t *recent_tokens; /* window the penalties look back over */
    size_t recent_count;
    const uint32_t *logit_bias_tokens; /* optional per-token additive bias */
    const float *logit_bias_values;
    size_t logit_bias_count;
} kipp_sample_params;

int kipp_sample_ex(const float *logits, size_t logits_count,
                   const kipp_sample_params *params, uint64_t *rng_state,
                   uint32_t *out_token, kipp_error *error);

const char *kipp_error_code_name(kipp_error_code code);

#ifdef KIPP_TESTING
float kipp_test_bf16_to_float(uint16_t value);
uint16_t kipp_test_float_to_bf16(float value);
float kipp_test_fp16_to_float(uint16_t value);
void kipp_test_matvec_q8_0(const uint8_t *weight, const float *input,
                           float *output, size_t rows, size_t columns);
void kipp_test_matvec_affine4_gs32(const uint8_t *weight, const float *input,
                                   float *output, size_t rows, size_t columns);
int kipp_test_checked_add_size(size_t left, size_t right, size_t *result);
int kipp_test_checked_multiply_size(size_t left, size_t right, size_t *result);
uint64_t kipp_test_kv_cache_bytes(uint32_t block_count, uint32_t capacity);
size_t kipp_test_kv_cache_offset(uint32_t capacity, uint32_t layer,
                                 uint32_t position, uint32_t head,
                                 uint32_t dimension);
void kipp_test_rms_norm(const float *input, const uint16_t *weight,
                        float *output, size_t length, float epsilon);
void kipp_test_matvec_bf16(const uint16_t *weight, const float *input,
                           float *output, size_t rows, size_t columns);
void kipp_test_rope(float *head, size_t head_dim, uint32_t position,
                    float theta);
void kipp_test_softmax(float *values, size_t count);
float kipp_test_silu(float value);
void kipp_test_causal_gqa(const float *query, const float *key,
                          const float *value, float *output, float *scores,
                          size_t token_count, size_t query_head_count);
int kipp_test_pretokenize(const char *text, size_t **offsets,
                          size_t **lengths, size_t *count);
int kipp_test_normalize_nfc(const char *text, char **normalized);
/* Reverse a CPU session's paged block table (before any eval) so the paged
 * gate can prove reads/writes are correct under a non-identity mapping. */
int kipp_test_scramble_session_kv(kipp_session *session);
#endif

#ifdef __cplusplus
}
#endif

#endif
