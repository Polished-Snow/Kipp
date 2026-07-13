#ifndef KIPP_H
#define KIPP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KIPP_VERSION "0.0.1"

#define KIPP_MODEL_REPOSITORY "Qwen/Qwen3-4B-Base"
#define KIPP_MODEL_REVISION "906bfd4b4dc7f14ee4320094d8b41684abff8539"

#define KIPP_BLOCK_COUNT 36u
#define KIPP_EMBEDDING_LENGTH 2560u
#define KIPP_FEED_FORWARD_LENGTH 9728u
#define KIPP_ATTENTION_HEAD_COUNT 32u
#define KIPP_ATTENTION_HEAD_COUNT_KV 8u
#define KIPP_ATTENTION_HEAD_DIM 128u
#define KIPP_VOCAB_SIZE 151936u
#define KIPP_CONTEXT_LENGTH 32768u
#define KIPP_EOS_TOKEN_ID 151643u

typedef struct kipp_model kipp_model;
typedef struct kipp_session kipp_session;

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
    const char *repository;
    const char *revision;
    kipp_backend_kind backend;
    uint32_t block_count;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t attention_head_dim;
    uint32_t vocab_size;
    uint32_t context_length;
    uint64_t mapped_bytes;
} kipp_model_info;

typedef struct {
    uint32_t capacity;
    uint32_t length;
    uint64_t cache_bytes;
} kipp_session_info;

/*
 * Open the strict Kipp Qwen3-4B GGUF artifact. The returned model owns a
 * read-only mmap until kipp_model_close().
 */
int kipp_model_open(const char *path, kipp_model **out_model, kipp_error *error);
int kipp_model_open_backend(const char *path, kipp_backend_kind backend,
                            kipp_model **out_model, kipp_error *error);
int kipp_model_close(kipp_model *model, kipp_error *error);
int kipp_model_get_info(const kipp_model *model, kipp_model_info *out_info);
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

const char *kipp_error_code_name(kipp_error_code code);

#ifdef KIPP_TESTING
float kipp_test_bf16_to_float(uint16_t value);
uint16_t kipp_test_float_to_bf16(float value);
int kipp_test_checked_add_size(size_t left, size_t right, size_t *result);
int kipp_test_checked_multiply_size(size_t left, size_t right, size_t *result);
uint64_t kipp_test_kv_cache_bytes(uint32_t capacity);
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
                          size_t token_count);
int kipp_test_pretokenize(const char *text, size_t **offsets,
                          size_t **lengths, size_t *count);
int kipp_test_normalize_nfc(const char *text, char **normalized);
#endif

#ifdef __cplusplus
}
#endif

#endif
