#include "kipp.h"
#include "kipp_backend.h"
#include "kipp_checkpoints.h"
#include "kipp_kv_pool.h"
#ifdef KIPP_ENABLE_METAL
#include "metal/kipp_metal.h"
#endif
#ifdef KIPP_ENABLE_CUDA
#include "cuda/kipp_cuda.h"
#endif

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define KIPP_GGUF_VERSION 3u
#define KIPP_GGUF_ALIGNMENT 32u
#define KIPP_MAX_TENSOR_COUNT 4096u
#define KIPP_PHASE1_MAX_TOKENS 256u
#define KIPP_RMS_EPSILON 1.0e-6f

enum gguf_value_type {
    GGUF_VALUE_UINT8 = 0,
    GGUF_VALUE_INT8 = 1,
    GGUF_VALUE_UINT16 = 2,
    GGUF_VALUE_INT16 = 3,
    GGUF_VALUE_UINT32 = 4,
    GGUF_VALUE_INT32 = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL = 7,
    GGUF_VALUE_STRING = 8,
    GGUF_VALUE_ARRAY = 9,
    GGUF_VALUE_UINT64 = 10,
    GGUF_VALUE_INT64 = 11,
    GGUF_VALUE_FLOAT64 = 12
};

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t offset;
} byte_cursor;

typedef struct {
    uint8_t *bytes;
    uint32_t length;
    bool special;
} tokenizer_token;

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t result;
    uint32_t rank;
} tokenizer_merge;

typedef struct {
    uint64_t hash;
    uint32_t token_plus_one;
} vocab_hash_entry;

typedef struct {
    uint64_t key_plus_one;
    uint32_t merge_plus_one;
} merge_hash_entry;

typedef struct {
    tokenizer_token *tokens;
    size_t token_count;
    tokenizer_merge *merges;
    size_t merge_count;
    vocab_hash_entry *vocab_hash;
    size_t vocab_hash_capacity;
    merge_hash_entry *merge_hash;
    size_t merge_hash_capacity;
    uint32_t *special_tokens;
    size_t special_token_count;
    bool add_bos;
    uint32_t eos_token;
} kipp_tokenizer;

typedef struct {
    char *name;
    uint64_t dimensions[2];
    uint32_t dimension_count;
    uint32_t type;
    uint64_t offset;
    bool bound;
} gguf_tensor_info;

typedef struct {
    char architecture[32];
    char source_repository[96];
    char source_revision[64];
    char quant_scheme[16];
    char tokenizer_model[16];
    char tokenizer_pre[16];
    uint32_t alignment;
    uint32_t block_count;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t attention_head_dim;
    uint32_t context_length;
    uint32_t vocab_size;
    float rope_theta;
    float rms_epsilon;
    bool tied_embeddings;
    bool saw_add_bos;
    kipp_tokenizer tokenizer;
    uint32_t *merge_left;
    size_t merge_left_count;
    uint32_t *merge_right;
    size_t merge_right_count;
    uint32_t *merge_result;
    size_t merge_result_count;
    uint8_t *token_special;
    size_t token_special_count;
} gguf_metadata;

struct kipp_model {
    int file_descriptor;
    uint8_t *mapping;
    size_t mapping_size;
    kipp_model_view view;
    kipp_tokenizer tokenizer;
    const kipp_backend_ops *backend_ops;
    void *backend_model;
    size_t active_session_count;
    kipp_backend_kind backend_kind;
    kipp_kv_pool *kv_pool; /* NULL unless opened with kipp_model_open_pooled */
};

/*
 * Pooled sessions (model->kv_pool != NULL) carry the sharing bookkeeping in
 * the core: `timeline` records every evaluated token id, `block_ids` maps
 * each logical block onto a pool block (this array is passed to the backend
 * as the eval-item block table), and `adopted_blocks` counts the leading
 * blocks adopted from the pool by kipp_session_match_prefix. Blocks a
 * session writes stay private until reset/destroy publishes the full ones
 * ("publish-at-finish"), which is what keeps truncation — speculative
 * rollback — from ever landing inside a shared block.
 */
struct kipp_session {
    kipp_model *model;
    void *backend_session;
    uint32_t capacity;
    uint32_t length;
    uint64_t cache_bytes;
    uint32_t *timeline;
    uint32_t *block_ids;
    uint32_t mapped_blocks;
    uint32_t adopted_blocks;
};

static const kipp_backend_ops *cpu_backend_operations(void);
static void destroy_model(kipp_model *model);

typedef struct {
    uint32_t first;
    uint32_t last;
} unicode_range;

typedef struct {
    uint32_t codepoint;
    uint32_t first;
    uint32_t second;
    uint8_t count;
} unicode_decomposition;

typedef struct {
    uint32_t codepoint;
    uint8_t combining_class;
} unicode_combining;

typedef struct {
    uint32_t first;
    uint32_t second;
    uint32_t result;
} unicode_composition;

#include "kipp_unicode.inc"

static void clear_error(kipp_error *error) {
    if (error != NULL) {
        error->code = KIPP_OK;
        error->message[0] = '\0';
    }
}

static int fail(kipp_error *error, kipp_error_code code, const char *format, ...) {
    if (error != NULL) {
        va_list arguments;
        error->code = code;
        va_start(arguments, format);
        (void)vsnprintf(error->message, sizeof(error->message), format, arguments);
        va_end(arguments);
    }
    return -1;
}

const char *kipp_error_code_name(kipp_error_code code) {
    switch (code) {
    case KIPP_OK:
        return "ok";
    case KIPP_ERROR_ARGUMENT:
        return "argument";
    case KIPP_ERROR_IO:
        return "io";
    case KIPP_ERROR_FORMAT:
        return "format";
    case KIPP_ERROR_UNSUPPORTED:
        return "unsupported";
    case KIPP_ERROR_MEMORY:
        return "memory";
    case KIPP_ERROR_RANGE:
        return "range";
    case KIPP_ERROR_INTERNAL:
        return "internal";
    }
    return "unknown";
}

static bool checked_add_size(size_t left, size_t right, size_t *result) {
    if (left > SIZE_MAX - right) {
        return false;
    }
    *result = left + right;
    return true;
}

static bool checked_multiply_size(size_t left, size_t right, size_t *result) {
    if (left != 0 && right > SIZE_MAX / left) {
        return false;
    }
    *result = left * right;
    return true;
}

static bool cursor_take(byte_cursor *cursor, size_t count, const uint8_t **output) {
    size_t end;
    if (!checked_add_size(cursor->offset, count, &end) || end > cursor->size) {
        return false;
    }
    *output = cursor->data + cursor->offset;
    cursor->offset = end;
    return true;
}

static bool cursor_u8(byte_cursor *cursor, uint8_t *value) {
    const uint8_t *bytes;
    if (!cursor_take(cursor, 1, &bytes)) {
        return false;
    }
    *value = bytes[0];
    return true;
}

static bool cursor_u32(byte_cursor *cursor, uint32_t *value) {
    const uint8_t *bytes;
    if (!cursor_take(cursor, 4, &bytes)) {
        return false;
    }
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return true;
}

static bool cursor_u64(byte_cursor *cursor, uint64_t *value) {
    const uint8_t *bytes;
    if (!cursor_take(cursor, 8, &bytes)) {
        return false;
    }
    *value = (uint64_t)bytes[0] | ((uint64_t)bytes[1] << 8) |
             ((uint64_t)bytes[2] << 16) | ((uint64_t)bytes[3] << 24) |
             ((uint64_t)bytes[4] << 32) | ((uint64_t)bytes[5] << 40) |
             ((uint64_t)bytes[6] << 48) | ((uint64_t)bytes[7] << 56);
    return true;
}

static bool cursor_f32(byte_cursor *cursor, float *value) {
    uint32_t bits;
    if (!cursor_u32(cursor, &bits)) {
        return false;
    }
    memcpy(value, &bits, sizeof(bits));
    return true;
}

static bool cursor_string_bytes(byte_cursor *cursor, const uint8_t **bytes,
                                size_t *length) {
    uint64_t wire_length;
    if (!cursor_u64(cursor, &wire_length) || wire_length > SIZE_MAX) {
        return false;
    }
    *length = (size_t)wire_length;
    return cursor_take(cursor, *length, bytes);
}

static bool cursor_c_string(byte_cursor *cursor, char **output) {
    const uint8_t *bytes;
    size_t length;
    char *string;
    if (!cursor_string_bytes(cursor, &bytes, &length) || length > SIZE_MAX - 1) {
        return false;
    }
    string = malloc(length + 1);
    if (string == NULL) {
        return false;
    }
    memcpy(string, bytes, length);
    string[length] = '\0';
    *output = string;
    return true;
}

static bool skip_gguf_value(byte_cursor *cursor, uint32_t type);

static bool skip_gguf_scalar(byte_cursor *cursor, uint32_t type) {
    const uint8_t *ignored;
    size_t size;
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        size = 1;
        break;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        size = 2;
        break;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        size = 4;
        break;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        size = 8;
        break;
    case GGUF_VALUE_STRING: {
        const uint8_t *string_bytes;
        size_t string_length;
        return cursor_string_bytes(cursor, &string_bytes, &string_length);
    }
    default:
        return false;
    }
    return cursor_take(cursor, size, &ignored);
}

static bool skip_gguf_value(byte_cursor *cursor, uint32_t type) {
    if (type != GGUF_VALUE_ARRAY) {
        return skip_gguf_scalar(cursor, type);
    }

    uint32_t element_type;
    uint64_t count;
    if (!cursor_u32(cursor, &element_type) || !cursor_u64(cursor, &count)) {
        return false;
    }
    for (uint64_t index = 0; index < count; ++index) {
        if (!skip_gguf_scalar(cursor, element_type)) {
            return false;
        }
    }
    return true;
}

static bool key_is(const uint8_t *key, size_t key_length, const char *expected) {
    size_t expected_length = strlen(expected);
    return key_length == expected_length &&
           memcmp(key, expected, expected_length) == 0;
}

static bool copy_metadata_string(byte_cursor *cursor, char *destination,
                                 size_t destination_size) {
    const uint8_t *bytes;
    size_t length;
    if (!cursor_string_bytes(cursor, &bytes, &length) ||
        length >= destination_size) {
        return false;
    }
    memcpy(destination, bytes, length);
    destination[length] = '\0';
    return true;
}

static bool read_u32_array(byte_cursor *cursor, uint32_t **values,
                           size_t *value_count) {
    uint32_t element_type;
    uint64_t count;
    size_t bytes;
    uint32_t *result;
    if (!cursor_u32(cursor, &element_type) || element_type != GGUF_VALUE_UINT32 ||
        !cursor_u64(cursor, &count) || count > SIZE_MAX ||
        !checked_multiply_size((size_t)count, sizeof(*result), &bytes)) {
        return false;
    }
    result = malloc(bytes == 0 ? 1 : bytes);
    if (result == NULL) {
        return false;
    }
    for (size_t index = 0; index < (size_t)count; ++index) {
        if (!cursor_u32(cursor, &result[index])) {
            free(result);
            return false;
        }
    }
    *values = result;
    *value_count = (size_t)count;
    return true;
}

static bool read_u8_array(byte_cursor *cursor, uint8_t **values,
                          size_t *value_count) {
    uint32_t element_type;
    uint64_t count;
    const uint8_t *source;
    uint8_t *result;
    if (!cursor_u32(cursor, &element_type) || element_type != GGUF_VALUE_UINT8 ||
        !cursor_u64(cursor, &count) || count > SIZE_MAX ||
        !cursor_take(cursor, (size_t)count, &source)) {
        return false;
    }
    result = malloc(count == 0 ? 1 : (size_t)count);
    if (result == NULL) {
        return false;
    }
    memcpy(result, source, (size_t)count);
    *values = result;
    *value_count = (size_t)count;
    return true;
}

static bool read_token_array(byte_cursor *cursor, tokenizer_token **tokens,
                             size_t *token_count) {
    uint32_t element_type;
    uint64_t count;
    tokenizer_token *result;
    if (!cursor_u32(cursor, &element_type) ||
        element_type != GGUF_VALUE_STRING || !cursor_u64(cursor, &count) ||
        count > SIZE_MAX / sizeof(*result)) {
        return false;
    }
    result = calloc((size_t)count, sizeof(*result));
    if (result == NULL) {
        return false;
    }
    for (size_t index = 0; index < (size_t)count; ++index) {
        const uint8_t *bytes;
        size_t length;
        if (!cursor_string_bytes(cursor, &bytes, &length) ||
            length > UINT32_MAX) {
            for (size_t previous = 0; previous < index; ++previous) {
                free(result[previous].bytes);
            }
            free(result);
            return false;
        }
        result[index].bytes = malloc(length == 0 ? 1 : length);
        if (result[index].bytes == NULL) {
            for (size_t previous = 0; previous < index; ++previous) {
                free(result[previous].bytes);
            }
            free(result);
            return false;
        }
        memcpy(result[index].bytes, bytes, length);
        result[index].length = (uint32_t)length;
    }
    *tokens = result;
    *token_count = (size_t)count;
    return true;
}

static bool read_metadata_entry(byte_cursor *cursor, gguf_metadata *metadata) {
    const uint8_t *key;
    size_t key_length;
    uint32_t type;
    if (!cursor_string_bytes(cursor, &key, &key_length) ||
        !cursor_u32(cursor, &type)) {
        return false;
    }

#define READ_U32_KEY(name, field)                                                \
    if (key_is(key, key_length, name)) {                                         \
        return type == GGUF_VALUE_UINT32 && cursor_u32(cursor, &metadata->field); \
    }
#define READ_F32_KEY(name, field)                                                \
    if (key_is(key, key_length, name)) {                                         \
        return type == GGUF_VALUE_FLOAT32 && cursor_f32(cursor, &metadata->field);\
    }

    if (key_is(key, key_length, "general.architecture")) {
        return type == GGUF_VALUE_STRING &&
               copy_metadata_string(cursor, metadata->architecture,
                                    sizeof(metadata->architecture));
    }
    if (key_is(key, key_length, "kipp.source.repository")) {
        return type == GGUF_VALUE_STRING &&
               copy_metadata_string(cursor, metadata->source_repository,
                                    sizeof(metadata->source_repository));
    }
    if (key_is(key, key_length, "kipp.source.revision")) {
        return type == GGUF_VALUE_STRING &&
               copy_metadata_string(cursor, metadata->source_revision,
                                    sizeof(metadata->source_revision));
    }
    if (key_is(key, key_length, "kipp.quant.scheme")) {
        return type == GGUF_VALUE_STRING &&
               copy_metadata_string(cursor, metadata->quant_scheme,
                                    sizeof(metadata->quant_scheme));
    }
    if (key_is(key, key_length, "tokenizer.ggml.model")) {
        return type == GGUF_VALUE_STRING &&
               copy_metadata_string(cursor, metadata->tokenizer_model,
                                    sizeof(metadata->tokenizer_model));
    }
    if (key_is(key, key_length, "tokenizer.ggml.pre")) {
        return type == GGUF_VALUE_STRING &&
               copy_metadata_string(cursor, metadata->tokenizer_pre,
                                    sizeof(metadata->tokenizer_pre));
    }
    READ_U32_KEY("general.alignment", alignment)
    READ_U32_KEY("qwen3.block_count", block_count)
    READ_U32_KEY("qwen3.embedding_length", embedding_length)
    READ_U32_KEY("qwen3.feed_forward_length", feed_forward_length)
    READ_U32_KEY("qwen3.attention.head_count", attention_head_count)
    READ_U32_KEY("qwen3.attention.head_count_kv", attention_head_count_kv)
    READ_U32_KEY("qwen3.attention.key_length", attention_head_dim)
    READ_U32_KEY("qwen3.context_length", context_length)
    READ_U32_KEY("qwen3.vocab_size", vocab_size)
    READ_F32_KEY("qwen3.rope.freq_base", rope_theta)
    READ_F32_KEY("qwen3.attention.layer_norm_rms_epsilon", rms_epsilon)

    if (key_is(key, key_length, "qwen3.tie_word_embeddings")) {
        uint8_t value;
        if (type != GGUF_VALUE_BOOL || !cursor_u8(cursor, &value)) {
            return false;
        }
        metadata->tied_embeddings = value != 0;
        return true;
    }
    if (key_is(key, key_length, "tokenizer.ggml.add_bos_token")) {
        uint8_t value;
        if (type != GGUF_VALUE_BOOL || !cursor_u8(cursor, &value)) {
            return false;
        }
        metadata->tokenizer.add_bos = value != 0;
        metadata->saw_add_bos = true;
        return true;
    }
    if (key_is(key, key_length, "tokenizer.ggml.eos_token_id")) {
        return type == GGUF_VALUE_UINT32 &&
               cursor_u32(cursor, &metadata->tokenizer.eos_token);
    }
    if (key_is(key, key_length, "kipp.tokenizer.tokens")) {
        return type == GGUF_VALUE_ARRAY &&
               read_token_array(cursor, &metadata->tokenizer.tokens,
                                &metadata->tokenizer.token_count);
    }
    if (key_is(key, key_length, "kipp.tokenizer.special")) {
        return type == GGUF_VALUE_ARRAY &&
               read_u8_array(cursor, &metadata->token_special,
                             &metadata->token_special_count);
    }
    if (key_is(key, key_length, "kipp.tokenizer.merge_left")) {
        return type == GGUF_VALUE_ARRAY &&
               read_u32_array(cursor, &metadata->merge_left,
                              &metadata->merge_left_count);
    }
    if (key_is(key, key_length, "kipp.tokenizer.merge_right")) {
        return type == GGUF_VALUE_ARRAY &&
               read_u32_array(cursor, &metadata->merge_right,
                              &metadata->merge_right_count);
    }
    if (key_is(key, key_length, "kipp.tokenizer.merge_result")) {
        return type == GGUF_VALUE_ARRAY &&
               read_u32_array(cursor, &metadata->merge_result,
                              &metadata->merge_result_count);
    }

#undef READ_U32_KEY
#undef READ_F32_KEY
    return skip_gguf_value(cursor, type);
}

static void free_tokenizer(kipp_tokenizer *tokenizer) {
    if (tokenizer == NULL) {
        return;
    }
    for (size_t index = 0; index < tokenizer->token_count; ++index) {
        free(tokenizer->tokens[index].bytes);
    }
    free(tokenizer->tokens);
    free(tokenizer->merges);
    free(tokenizer->vocab_hash);
    free(tokenizer->merge_hash);
    free(tokenizer->special_tokens);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

static void free_metadata(gguf_metadata *metadata) {
    free(metadata->merge_left);
    free(metadata->merge_right);
    free(metadata->merge_result);
    free(metadata->token_special);
    free_tokenizer(&metadata->tokenizer);
    memset(metadata, 0, sizeof(*metadata));
}

static uint64_t hash_bytes(const uint8_t *bytes, size_t length) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < length; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash == 0 ? 1 : hash;
}

static size_t next_power_of_two(size_t value) {
    size_t result = 1;
    while (result < value && result <= SIZE_MAX / 2) {
        result *= 2;
    }
    return result;
}

static int build_tokenizer_indexes(kipp_tokenizer *tokenizer,
                                   kipp_error *error) {
    size_t vocab_capacity = next_power_of_two(tokenizer->token_count * 2);
    size_t merge_capacity = next_power_of_two(tokenizer->merge_count * 2);
    if (vocab_capacity < tokenizer->token_count ||
        merge_capacity < tokenizer->merge_count) {
        return fail(error, KIPP_ERROR_MEMORY, "tokenizer index size overflow");
    }
    tokenizer->vocab_hash = calloc(vocab_capacity, sizeof(*tokenizer->vocab_hash));
    tokenizer->merge_hash = calloc(merge_capacity, sizeof(*tokenizer->merge_hash));
    if (tokenizer->vocab_hash == NULL || tokenizer->merge_hash == NULL) {
        return fail(error, KIPP_ERROR_MEMORY, "unable to allocate tokenizer indexes");
    }
    tokenizer->vocab_hash_capacity = vocab_capacity;
    tokenizer->merge_hash_capacity = merge_capacity;

    for (size_t token_id = 0; token_id < tokenizer->token_count; ++token_id) {
        tokenizer_token *token = &tokenizer->tokens[token_id];
        if (token->length == 0) {
            continue;
        }
        uint64_t hash = hash_bytes(token->bytes, token->length);
        size_t slot = (size_t)hash & (vocab_capacity - 1);
        while (tokenizer->vocab_hash[slot].token_plus_one != 0) {
            slot = (slot + 1) & (vocab_capacity - 1);
        }
        tokenizer->vocab_hash[slot].hash = hash;
        tokenizer->vocab_hash[slot].token_plus_one = (uint32_t)token_id + 1;
    }

    for (size_t merge_index = 0; merge_index < tokenizer->merge_count;
         ++merge_index) {
        tokenizer_merge *merge = &tokenizer->merges[merge_index];
        uint64_t key = ((uint64_t)merge->left << 32) | merge->right;
        uint64_t stored_key = key + 1;
        size_t slot = (size_t)(key ^ (key >> 33)) & (merge_capacity - 1);
        while (tokenizer->merge_hash[slot].merge_plus_one != 0) {
            slot = (slot + 1) & (merge_capacity - 1);
        }
        tokenizer->merge_hash[slot].key_plus_one = stored_key;
        tokenizer->merge_hash[slot].merge_plus_one = (uint32_t)merge_index + 1;
    }
    return 0;
}

static int finish_tokenizer_metadata(gguf_metadata *metadata,
                                     kipp_error *error) {
    size_t merge_count = metadata->merge_left_count;
    if (metadata->tokenizer.token_count != KIPP_VOCAB_SIZE ||
        metadata->token_special_count != KIPP_VOCAB_SIZE ||
        merge_count != metadata->merge_right_count ||
        merge_count != metadata->merge_result_count) {
        return fail(error, KIPP_ERROR_FORMAT,
                    "tokenizer arrays do not match the fixed vocabulary");
    }
    metadata->tokenizer.merges = calloc(merge_count,
                                        sizeof(*metadata->tokenizer.merges));
    if (metadata->tokenizer.merges == NULL) {
        return fail(error, KIPP_ERROR_MEMORY, "unable to allocate BPE merges");
    }
    metadata->tokenizer.merge_count = merge_count;
    size_t special_count = 0;
    for (size_t index = 0; index < KIPP_VOCAB_SIZE; ++index) {
        metadata->tokenizer.tokens[index].special =
            metadata->token_special[index] != 0;
        special_count += metadata->tokenizer.tokens[index].special;
    }
    metadata->tokenizer.special_tokens =
        malloc((special_count == 0 ? 1 : special_count) *
               sizeof(*metadata->tokenizer.special_tokens));
    if (metadata->tokenizer.special_tokens == NULL) {
        return fail(error, KIPP_ERROR_MEMORY,
                    "unable to allocate special-token index");
    }
    metadata->tokenizer.special_token_count = 0;
    for (size_t index = 0; index < KIPP_VOCAB_SIZE; ++index) {
        if (metadata->tokenizer.tokens[index].special) {
            metadata->tokenizer
                .special_tokens[metadata->tokenizer.special_token_count++] =
                (uint32_t)index;
        }
    }
    for (size_t index = 0; index < merge_count; ++index) {
        uint32_t left = metadata->merge_left[index];
        uint32_t right = metadata->merge_right[index];
        uint32_t result = metadata->merge_result[index];
        if (left >= KIPP_VOCAB_SIZE || right >= KIPP_VOCAB_SIZE ||
            result >= KIPP_VOCAB_SIZE) {
            return fail(error, KIPP_ERROR_FORMAT,
                        "BPE merge %zu contains an invalid token ID", index);
        }
        metadata->tokenizer.merges[index] =
            (tokenizer_merge){left, right, result, (uint32_t)index};
    }
    return build_tokenizer_indexes(&metadata->tokenizer, error);
}

static const kipp_checkpoint_spec *
find_checkpoint(const char *repository, const char *revision) {
    for (size_t index = 0; index < KIPP_SUPPORTED_CHECKPOINT_COUNT; ++index) {
        const kipp_checkpoint_spec *spec = &kipp_supported_checkpoints[index];
        if (strcmp(spec->repository, repository) == 0 &&
            strcmp(spec->revision, revision) == 0) {
            return spec;
        }
    }
    return NULL;
}

/*
 * Match the GGUF against the compiled-in checkpoint registry and require
 * every metadata field to equal the registry entry exactly. Unknown
 * checkpoints and mismatched dimensions are rejections, not extension
 * points.
 */
static const kipp_checkpoint_spec *
validate_metadata(const gguf_metadata *metadata, kipp_error *error) {
    if (strcmp(metadata->architecture, "qwen3") != 0) {
        fail(error, KIPP_ERROR_UNSUPPORTED,
             "expected qwen3 architecture, found '%s'",
             metadata->architecture);
        return NULL;
    }
    const kipp_checkpoint_spec *spec =
        find_checkpoint(metadata->source_repository,
                        metadata->source_revision);
    if (spec == NULL) {
        fail(error, KIPP_ERROR_UNSUPPORTED,
             "checkpoint '%s'@'%.16s' is not in the supported registry",
             metadata->source_repository, metadata->source_revision);
        return NULL;
    }
    if (metadata->alignment != KIPP_GGUF_ALIGNMENT ||
        metadata->block_count != spec->block_count ||
        metadata->block_count > KIPP_MAX_BLOCK_COUNT ||
        metadata->embedding_length != spec->embedding_length ||
        metadata->feed_forward_length != spec->feed_forward_length ||
        metadata->attention_head_count != spec->attention_head_count ||
        metadata->attention_head_count_kv != KIPP_ATTENTION_HEAD_COUNT_KV ||
        metadata->attention_head_dim != KIPP_ATTENTION_HEAD_DIM ||
        metadata->context_length != spec->context_length ||
        metadata->vocab_size != KIPP_VOCAB_SIZE ||
        fabsf(metadata->rope_theta - spec->rope_theta) > 0.5f ||
        fabsf(metadata->rms_epsilon - KIPP_RMS_EPSILON) > 1.0e-9f ||
        metadata->tied_embeddings != spec->tied_embeddings ||
        metadata->tokenizer.add_bos || !metadata->saw_add_bos ||
        strcmp(metadata->tokenizer_model, "gpt2") != 0 ||
        strcmp(metadata->tokenizer_pre, "qwen2") != 0 ||
        metadata->tokenizer.eos_token != spec->eos_token_id) {
        fail(error, KIPP_ERROR_UNSUPPORTED,
             "GGUF metadata does not match registry entry '%s'", spec->id);
        return NULL;
    }
    return spec;
}

static void config_from_spec(kipp_model_config *config,
                             const kipp_checkpoint_spec *spec) {
    config->spec = spec;
    config->block_count = spec->block_count;
    config->embedding_length = spec->embedding_length;
    config->feed_forward_length = spec->feed_forward_length;
    config->attention_head_count = spec->attention_head_count;
    config->attention_width =
        spec->attention_head_count * KIPP_ATTENTION_HEAD_DIM;
    config->context_length = spec->context_length;
    config->rope_theta = spec->rope_theta;
    config->tied_embeddings = spec->tied_embeddings;
}

static gguf_tensor_info *find_tensor(gguf_tensor_info *tensors,
                                     size_t tensor_count, const char *name) {
    for (size_t index = 0; index < tensor_count; ++index) {
        if (strcmp(tensors[index].name, name) == 0) {
            return &tensors[index];
        }
    }
    return NULL;
}

/* Storage bytes for element_count weights at the given tensor type. */
static bool tensor_type_byte_count(kipp_tensor_type type,
                                   uint64_t element_count, uint64_t *bytes) {
    switch (type) {
    case KIPP_TENSOR_BF16:
        if (element_count > UINT64_MAX / 2) {
            return false;
        }
        *bytes = element_count * 2;
        return true;
    case KIPP_TENSOR_Q8_0:
        if (element_count % KIPP_QUANT_BLOCK != 0 ||
            element_count / KIPP_QUANT_BLOCK > UINT64_MAX /
                                                   KIPP_Q8_0_BLOCK_BYTES) {
            return false;
        }
        *bytes = element_count / KIPP_QUANT_BLOCK * KIPP_Q8_0_BLOCK_BYTES;
        return true;
    case KIPP_TENSOR_AFFINE4_GS32:
        if (element_count % KIPP_QUANT_BLOCK != 0 ||
            element_count / KIPP_QUANT_BLOCK > UINT64_MAX /
                                                   KIPP_AFFINE4_GROUP_BYTES) {
            return false;
        }
        *bytes = element_count / KIPP_QUANT_BLOCK * KIPP_AFFINE4_GROUP_BYTES;
        return true;
    }
    return false;
}

static int bind_tensor(kipp_tensor_view *destination, gguf_tensor_info *tensors,
                       size_t tensor_count, const char *name,
                       kipp_tensor_type expected_type,
                       uint32_t dimension_count, uint64_t dimension_zero,
                       uint64_t dimension_one, const uint8_t *data_base,
                       size_t data_size, kipp_error *error) {
    gguf_tensor_info *source = find_tensor(tensors, tensor_count, name);
    uint64_t element_count;
    uint64_t byte_count;
    if (source == NULL) {
        return fail(error, KIPP_ERROR_FORMAT, "missing tensor '%s'", name);
    }
    if (source->bound) {
        return fail(error, KIPP_ERROR_FORMAT, "tensor '%s' was bound twice", name);
    }
    if (source->type != (uint32_t)expected_type ||
        source->dimension_count != dimension_count ||
        source->dimensions[0] != dimension_zero ||
        (dimension_count == 2 && source->dimensions[1] != dimension_one)) {
        return fail(error, KIPP_ERROR_FORMAT,
                    "tensor '%s' has an unexpected type or shape", name);
    }
    if (dimension_count == 1) {
        element_count = dimension_zero;
    } else if (dimension_zero != 0 && dimension_one > UINT64_MAX / dimension_zero) {
        return fail(error, KIPP_ERROR_FORMAT, "tensor '%s' size overflows", name);
    } else {
        element_count = dimension_zero * dimension_one;
    }
    if (!tensor_type_byte_count(expected_type, element_count, &byte_count)) {
        return fail(error, KIPP_ERROR_FORMAT,
                    "tensor '%s' byte size overflows or is misaligned", name);
    }
    if (source->offset % KIPP_GGUF_ALIGNMENT != 0 ||
        source->offset > data_size || byte_count > data_size - source->offset) {
        return fail(error, KIPP_ERROR_FORMAT,
                    "tensor '%s' points outside aligned model data", name);
    }
    destination->name = source->name;
    destination->data = data_base + source->offset;
    destination->dimensions[0] = dimension_zero;
    destination->dimensions[1] = dimension_one;
    destination->dimension_count = dimension_count;
    destination->type = expected_type;
    destination->byte_count = byte_count;
    source->bound = true;
    return 0;
}

static int bind_model_weights(kipp_model_view *view, gguf_tensor_info *tensors,
                              size_t tensor_count, const uint8_t *data_base,
                              size_t data_size, kipp_error *error) {
    const kipp_model_config *config = &view->config;
    /* Projections carry the model's quant scheme; norms, the token
     * embedding, and lm_head always stay BF16. */
    kipp_tensor_type projection_type = KIPP_TENSOR_BF16;
    if (config->quant_scheme == KIPP_QUANT_Q8_0) {
        projection_type = KIPP_TENSOR_Q8_0;
    } else if (config->quant_scheme == KIPP_QUANT_AFFINE4_GS32) {
        projection_type = KIPP_TENSOR_AFFINE4_GS32;
    }
#define BIND_VECTOR(target, tensor_name, length)                                  \
    do {                                                                          \
        if (bind_tensor(&(target), tensors, tensor_count, tensor_name,           \
                        KIPP_TENSOR_BF16, 1, length, 0, data_base, data_size,     \
                        error) != 0) {                                            \
            return -1;                                                            \
        }                                                                         \
    } while (0)
#define BIND_MATRIX(target, tensor_name, columns, rows)                           \
    do {                                                                          \
        if (bind_tensor(&(target), tensors, tensor_count, tensor_name,           \
                        KIPP_TENSOR_BF16, 2, columns, rows, data_base,            \
                        data_size, error) != 0) {                                 \
            return -1;                                                            \
        }                                                                         \
    } while (0)
#define BIND_PROJECTION(target, tensor_name, columns, rows)                       \
    do {                                                                          \
        if (bind_tensor(&(target), tensors, tensor_count, tensor_name,           \
                        projection_type, 2, columns, rows, data_base,             \
                        data_size, error) != 0) {                                 \
            return -1;                                                            \
        }                                                                         \
    } while (0)

    BIND_MATRIX(view->weights.token_embedding, "model.embed_tokens.weight",
                config->embedding_length, KIPP_VOCAB_SIZE);
    BIND_VECTOR(view->weights.output_norm, "model.norm.weight",
                config->embedding_length);
    if (config->tied_embeddings) {
        view->weights.lm_head = view->weights.token_embedding;
    } else {
        BIND_MATRIX(view->weights.lm_head, "lm_head.weight",
                    config->embedding_length, KIPP_VOCAB_SIZE);
    }

    for (uint32_t layer_index = 0; layer_index < config->block_count;
         ++layer_index) {
        char name[128];
        kipp_qwen3_layer_weights *layer = &view->weights.layers[layer_index];

#define LAYER_NAME(suffix)                                                        \
        (void)snprintf(name, sizeof(name), "model.layers.%u.%s", layer_index, suffix)

        LAYER_NAME("input_layernorm.weight");
        BIND_VECTOR(layer->attention_norm, name, config->embedding_length);
        LAYER_NAME("self_attn.q_proj.weight");
        BIND_PROJECTION(layer->attention_q, name, config->embedding_length,
                    config->attention_width);
        LAYER_NAME("self_attn.k_proj.weight");
        BIND_PROJECTION(layer->attention_k, name, config->embedding_length,
                    KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM);
        LAYER_NAME("self_attn.v_proj.weight");
        BIND_PROJECTION(layer->attention_v, name, config->embedding_length,
                    KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM);
        LAYER_NAME("self_attn.o_proj.weight");
        BIND_PROJECTION(layer->attention_output, name, config->attention_width,
                    config->embedding_length);
        LAYER_NAME("self_attn.q_norm.weight");
        BIND_VECTOR(layer->attention_q_norm, name, KIPP_ATTENTION_HEAD_DIM);
        LAYER_NAME("self_attn.k_norm.weight");
        BIND_VECTOR(layer->attention_k_norm, name, KIPP_ATTENTION_HEAD_DIM);
        LAYER_NAME("post_attention_layernorm.weight");
        BIND_VECTOR(layer->feed_forward_norm, name, config->embedding_length);
        LAYER_NAME("mlp.gate_proj.weight");
        BIND_PROJECTION(layer->feed_forward_gate, name, config->embedding_length,
                    config->feed_forward_length);
        LAYER_NAME("mlp.up_proj.weight");
        BIND_PROJECTION(layer->feed_forward_up, name, config->embedding_length,
                    config->feed_forward_length);
        LAYER_NAME("mlp.down_proj.weight");
        BIND_PROJECTION(layer->feed_forward_down, name,
                    config->feed_forward_length, config->embedding_length);
#undef LAYER_NAME
    }
    for (size_t index = 0; index < tensor_count; ++index) {
        if (!tensors[index].bound) {
            return fail(error, KIPP_ERROR_FORMAT,
                        "unexpected tensor '%s' in strict model", tensors[index].name);
        }
    }
#undef BIND_VECTOR
#undef BIND_MATRIX
#undef BIND_PROJECTION
    return 0;
}

static void free_tensor_infos(gguf_tensor_info *tensors, size_t count) {
    if (tensors == NULL) {
        return;
    }
    for (size_t index = 0; index < count; ++index) {
        free(tensors[index].name);
    }
    free(tensors);
}

static bool info_byte_count(const gguf_tensor_info *info, uint64_t *bytes) {
    uint64_t elements = info->dimensions[0];
    if (info->dimension_count == 2) {
        if (info->dimensions[0] != 0 &&
            info->dimensions[1] > UINT64_MAX / info->dimensions[0]) {
            return false;
        }
        elements *= info->dimensions[1];
    }
    return tensor_type_byte_count((kipp_tensor_type)info->type, elements,
                                  bytes);
}

static int validate_tensor_ranges(const gguf_tensor_info *tensors, size_t count,
                                  size_t data_size, kipp_error *error) {
    for (size_t first = 0; first < count; ++first) {
        uint64_t first_bytes;
        if (!info_byte_count(&tensors[first], &first_bytes)) {
            return fail(error, KIPP_ERROR_FORMAT,
                        "tensor '%s' has an invalid type or byte count",
                        tensors[first].name);
        }
        uint64_t first_end = tensors[first].offset + first_bytes;
        if (first_end < tensors[first].offset || first_end > data_size) {
            return fail(error, KIPP_ERROR_FORMAT,
                        "tensor '%s' range is invalid", tensors[first].name);
        }
        for (size_t second = 0; second < first; ++second) {
            uint64_t second_bytes;
            if (!info_byte_count(&tensors[second], &second_bytes)) {
                continue;
            }
            uint64_t second_end = tensors[second].offset + second_bytes;
            if (tensors[first].offset < second_end &&
                tensors[second].offset < first_end) {
                return fail(error, KIPP_ERROR_FORMAT,
                            "tensors '%s' and '%s' overlap",
                            tensors[first].name, tensors[second].name);
            }
        }
    }
    return 0;
}

static int parse_model_mapping(kipp_model *model, kipp_error *error) {
    byte_cursor cursor = {model->mapping, model->mapping_size, 0};
    const uint8_t *magic;
    uint32_t version;
    uint64_t tensor_count_wire;
    uint64_t metadata_count;
    gguf_metadata metadata = {0};
    gguf_tensor_info *tensors = NULL;
    size_t tensor_count = 0;
    int result = -1;

    if (!cursor_take(&cursor, 4, &magic) || memcmp(magic, "GGUF", 4) != 0 ||
        !cursor_u32(&cursor, &version) || version != KIPP_GGUF_VERSION ||
        !cursor_u64(&cursor, &tensor_count_wire) ||
        tensor_count_wire == 0 || tensor_count_wire > KIPP_MAX_TENSOR_COUNT ||
        !cursor_u64(&cursor, &metadata_count) || metadata_count > 100000) {
        return fail(error, KIPP_ERROR_FORMAT, "invalid or unsupported GGUF header");
    }

    for (uint64_t index = 0; index < metadata_count; ++index) {
        if (!read_metadata_entry(&cursor, &metadata)) {
            fail(error, KIPP_ERROR_FORMAT, "invalid GGUF metadata entry %llu",
                 (unsigned long long)index);
            goto cleanup;
        }
    }
    const kipp_checkpoint_spec *spec = validate_metadata(&metadata, error);
    if (spec == NULL || finish_tokenizer_metadata(&metadata, error) != 0) {
        goto cleanup;
    }
    config_from_spec(&model->view.config, spec);
    if (metadata.quant_scheme[0] == '\0' ||
        strcmp(metadata.quant_scheme, "bf16") == 0) {
        model->view.config.quant_scheme = KIPP_QUANT_BF16;
    } else if (strcmp(metadata.quant_scheme, "q8_0") == 0) {
        model->view.config.quant_scheme = KIPP_QUANT_Q8_0;
    } else if (strcmp(metadata.quant_scheme, "affine4_gs32") == 0) {
        model->view.config.quant_scheme = KIPP_QUANT_AFFINE4_GS32;
    } else {
        fail(error, KIPP_ERROR_UNSUPPORTED,
             "unknown quantization scheme '%s'", metadata.quant_scheme);
        goto cleanup;
    }
    /* embeddings + final norm + 11 tensors per layer + untied lm_head */
    uint64_t expected_tensors =
        2u + (uint64_t)spec->block_count * 11u +
        (spec->tied_embeddings ? 0u : 1u);
    if (tensor_count_wire != expected_tensors) {
        fail(error, KIPP_ERROR_FORMAT,
             "GGUF holds %llu tensors but registry entry '%s' requires %llu",
             (unsigned long long)tensor_count_wire, spec->id,
             (unsigned long long)expected_tensors);
        goto cleanup;
    }

    tensor_count = (size_t)tensor_count_wire;
    tensors = calloc(tensor_count, sizeof(*tensors));
    if (tensors == NULL) {
        fail(error, KIPP_ERROR_MEMORY, "unable to allocate tensor directory");
        goto cleanup;
    }
    for (size_t index = 0; index < tensor_count; ++index) {
        uint32_t dimension_count;
        if (!cursor_c_string(&cursor, &tensors[index].name) ||
            !cursor_u32(&cursor, &dimension_count) ||
            dimension_count == 0 || dimension_count > 2) {
            fail(error, KIPP_ERROR_FORMAT, "invalid tensor directory entry %zu",
                 index);
            goto cleanup;
        }
        tensors[index].dimension_count = dimension_count;
        for (uint32_t dimension = 0; dimension < dimension_count; ++dimension) {
            if (!cursor_u64(&cursor, &tensors[index].dimensions[dimension]) ||
                tensors[index].dimensions[dimension] == 0) {
                fail(error, KIPP_ERROR_FORMAT,
                     "invalid dimensions for tensor '%s'", tensors[index].name);
                goto cleanup;
            }
        }
        if (!cursor_u32(&cursor, &tensors[index].type) ||
            !cursor_u64(&cursor, &tensors[index].offset)) {
            fail(error, KIPP_ERROR_FORMAT, "truncated tensor '%s'",
                 tensors[index].name);
            goto cleanup;
        }
    }

    size_t data_offset =
        (cursor.offset + (metadata.alignment - 1)) & ~(size_t)(metadata.alignment - 1);
    if (data_offset > model->mapping_size) {
        fail(error, KIPP_ERROR_FORMAT, "GGUF data section is outside the file");
        goto cleanup;
    }
    model->view.mapping = model->mapping;
    model->view.mapping_size = model->mapping_size;
    if (bind_model_weights(&model->view, tensors, tensor_count,
                           model->mapping + data_offset,
                           model->mapping_size - data_offset, error) != 0) {
        goto cleanup;
    }
    if (validate_tensor_ranges(tensors, tensor_count,
                               model->mapping_size - data_offset, error) != 0) {
        goto cleanup;
    }

    model->tokenizer = metadata.tokenizer;
    memset(&metadata.tokenizer, 0, sizeof(metadata.tokenizer));
    result = 0;

cleanup:
    free_tensor_infos(tensors, tensor_count);
    free_metadata(&metadata);
    return result;
}

int kipp_model_open(const char *path, kipp_model **out_model, kipp_error *error) {
    return kipp_model_open_backend(path, KIPP_BACKEND_CPU, out_model, error);
}

static int open_model(const char *path, kipp_backend_kind backend,
                      uint32_t kv_pool_blocks, kipp_model **out_model,
                      kipp_error *error) {
    struct stat status;
    kipp_model *model;
    const kipp_backend_ops *backend_ops;
    clear_error(error);
    if (path == NULL || out_model == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "model path and output pointer are required");
    }
    *out_model = NULL;
    switch (backend) {
    case KIPP_BACKEND_CPU:
        backend_ops = cpu_backend_operations();
        break;
    case KIPP_BACKEND_METAL:
#ifdef KIPP_ENABLE_METAL
        backend_ops = kipp_metal_backend_operations();
        break;
#else
        return fail(error, KIPP_ERROR_UNSUPPORTED,
                    "Metal support is not compiled into this binary");
#endif
    case KIPP_BACKEND_CUDA:
#ifdef KIPP_ENABLE_CUDA
        backend_ops = kipp_cuda_backend_operations();
        break;
#else
        return fail(error, KIPP_ERROR_UNSUPPORTED,
                    "CUDA support is not compiled into this binary");
#endif
    default:
        return fail(error, KIPP_ERROR_ARGUMENT, "unknown backend %d",
                    (int)backend);
    }
    model = calloc(1, sizeof(*model));
    if (model == NULL) {
        return fail(error, KIPP_ERROR_MEMORY, "unable to allocate model");
    }
    model->backend_kind = backend;
    model->backend_ops = backend_ops;
    model->file_descriptor = -1;
    model->file_descriptor = open(path, O_RDONLY);
    if (model->file_descriptor < 0) {
        fail(error, KIPP_ERROR_IO, "open '%s': %s", path, strerror(errno));
        goto failure;
    }
    if (fstat(model->file_descriptor, &status) != 0 || status.st_size <= 0 ||
        (uintmax_t)status.st_size > SIZE_MAX) {
        fail(error, KIPP_ERROR_IO, "invalid model file size");
        goto failure;
    }
    model->mapping_size = (size_t)status.st_size;
    int mapping_flags =
        backend == KIPP_BACKEND_METAL ? MAP_SHARED : MAP_PRIVATE;
    model->mapping = mmap(NULL, model->mapping_size, PROT_READ, mapping_flags,
                          model->file_descriptor, 0);
    if (model->mapping == MAP_FAILED) {
        model->mapping = NULL;
        fail(error, KIPP_ERROR_IO, "mmap '%s': %s", path, strerror(errno));
        goto failure;
    }
    if (parse_model_mapping(model, error) != 0) {
        goto failure;
    }
    model->view.config.kv_pool_blocks = kv_pool_blocks;
    if (kv_pool_blocks != 0) {
        model->kv_pool = kipp_kv_pool_create(kv_pool_blocks);
        if (model->kv_pool == NULL) {
            fail(error, KIPP_ERROR_MEMORY, "unable to allocate KV pool");
            goto failure;
        }
    }
    if (model->backend_ops->model_create(&model->view, &model->backend_model,
                                         error) != 0) {
        goto failure;
    }
    *out_model = model;
    return 0;

failure:
    destroy_model(model);
    return -1;
}

int kipp_model_open_backend(const char *path, kipp_backend_kind backend,
                            kipp_model **out_model, kipp_error *error) {
    return open_model(path, backend, 0, out_model, error);
}

int kipp_model_open_pooled(const char *path, kipp_backend_kind backend,
                           uint32_t kv_pool_blocks, kipp_model **out_model,
                           kipp_error *error) {
    clear_error(error);
    if (kv_pool_blocks == 0) {
        return fail(error, KIPP_ERROR_RANGE,
                    "pooled models need at least one KV block");
    }
    if (backend != KIPP_BACKEND_CPU && backend != KIPP_BACKEND_METAL) {
        return fail(error, KIPP_ERROR_UNSUPPORTED,
                    "pooled KV is supported on the CPU and Metal backends");
    }
    return open_model(path, backend, kv_pool_blocks, out_model, error);
}

static void destroy_model(kipp_model *model) {
    if (model == NULL) {
        return;
    }
    if (model->backend_ops != NULL && model->backend_model != NULL) {
        model->backend_ops->model_destroy(model->backend_model);
    }
    kipp_kv_pool_destroy(model->kv_pool);
    free_tokenizer(&model->tokenizer);
    if (model->mapping != NULL) {
        (void)munmap(model->mapping, model->mapping_size);
    }
    if (model->file_descriptor >= 0) {
        (void)close(model->file_descriptor);
    }
    free(model);
}

int kipp_model_close(kipp_model *model, kipp_error *error) {
    clear_error(error);
    if (model == NULL) {
        return 0;
    }
    if (model->active_session_count != 0) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "destroy all %zu active session(s) before closing the model",
                    model->active_session_count);
    }
    destroy_model(model);
    return 0;
}

int kipp_model_get_info(const kipp_model *model, kipp_model_info *out_info) {
    if (model == NULL || out_info == NULL) {
        return -1;
    }
    const kipp_model_config *config = &model->view.config;
    const kipp_checkpoint_spec *spec = config->spec;
    memset(out_info, 0, sizeof(*out_info));
    out_info->checkpoint_id = spec->id;
    out_info->repository = spec->repository;
    out_info->revision = spec->revision;
    out_info->variant = spec->variant;
    out_info->backend = model->backend_kind;
    out_info->block_count = config->block_count;
    out_info->embedding_length = config->embedding_length;
    out_info->feed_forward_length = config->feed_forward_length;
    out_info->attention_head_count = config->attention_head_count;
    out_info->attention_head_count_kv = KIPP_ATTENTION_HEAD_COUNT_KV;
    out_info->attention_head_dim = KIPP_ATTENTION_HEAD_DIM;
    out_info->vocab_size = KIPP_VOCAB_SIZE;
    out_info->context_length = config->context_length;
    out_info->rope_theta = config->rope_theta;
    out_info->tied_embeddings = config->tied_embeddings ? 1 : 0;
    out_info->quant_scheme =
        config->quant_scheme == KIPP_QUANT_Q8_0        ? "q8_0"
        : config->quant_scheme == KIPP_QUANT_AFFINE4_GS32 ? "affine4_gs32"
                                                          : "bf16";
    for (uint32_t index = 0; index < spec->stop_token_count; ++index) {
        out_info->stop_tokens[index] = spec->stop_tokens[index];
    }
    out_info->stop_token_count = spec->stop_token_count;
    out_info->mapped_bytes = model->mapping_size;
    return 0;
}

int kipp_model_is_stop_token(const kipp_model *model, uint32_t token) {
    if (model == NULL) {
        return 0;
    }
    const kipp_checkpoint_spec *spec = model->view.config.spec;
    for (uint32_t index = 0; index < spec->stop_token_count; ++index) {
        if (spec->stop_tokens[index] == token) {
            return 1;
        }
    }
    return 0;
}

const char *kipp_backend_name(kipp_backend_kind backend) {
    switch (backend) {
    case KIPP_BACKEND_CPU:
        return "cpu";
    case KIPP_BACKEND_METAL:
        return "metal";
    case KIPP_BACKEND_CUDA:
        return "cuda";
    }
    return "unknown";
}

static uint32_t vocab_lookup(const kipp_tokenizer *tokenizer,
                             const uint8_t *bytes, size_t length) {
    uint64_t hash = hash_bytes(bytes, length);
    size_t slot = (size_t)hash & (tokenizer->vocab_hash_capacity - 1);
    for (size_t probe = 0; probe < tokenizer->vocab_hash_capacity; ++probe) {
        const vocab_hash_entry *entry = &tokenizer->vocab_hash[slot];
        if (entry->token_plus_one == 0) {
            return UINT32_MAX;
        }
        uint32_t token_id = entry->token_plus_one - 1;
        const tokenizer_token *token = &tokenizer->tokens[token_id];
        if (entry->hash == hash && token->length == length &&
            memcmp(token->bytes, bytes, length) == 0) {
            return token_id;
        }
        slot = (slot + 1) & (tokenizer->vocab_hash_capacity - 1);
    }
    return UINT32_MAX;
}

static const tokenizer_merge *merge_lookup(const kipp_tokenizer *tokenizer,
                                           uint32_t left, uint32_t right) {
    uint64_t key = ((uint64_t)left << 32) | right;
    uint64_t stored_key = key + 1;
    size_t slot =
        (size_t)(key ^ (key >> 33)) & (tokenizer->merge_hash_capacity - 1);
    for (size_t probe = 0; probe < tokenizer->merge_hash_capacity; ++probe) {
        const merge_hash_entry *entry = &tokenizer->merge_hash[slot];
        if (entry->merge_plus_one == 0) {
            return NULL;
        }
        if (entry->key_plus_one == stored_key) {
            return &tokenizer->merges[entry->merge_plus_one - 1];
        }
        slot = (slot + 1) & (tokenizer->merge_hash_capacity - 1);
    }
    return NULL;
}

static bool unicode_in_ranges(uint32_t codepoint, const unicode_range *ranges,
                              size_t count) {
    size_t low = 0;
    size_t high = count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (codepoint < ranges[middle].first) {
            high = middle;
        } else if (codepoint > ranges[middle].last) {
            low = middle + 1;
        } else {
            return true;
        }
    }
    return false;
}

static bool unicode_is_letter(uint32_t codepoint) {
    return unicode_in_ranges(codepoint, kipp_unicode_letter_ranges,
                             KIPP_UNICODE_LETTER_RANGE_COUNT);
}

static bool unicode_is_number(uint32_t codepoint) {
    return unicode_in_ranges(codepoint, kipp_unicode_number_ranges,
                             KIPP_UNICODE_NUMBER_RANGE_COUNT);
}

static bool unicode_is_space(uint32_t codepoint) {
    return unicode_in_ranges(codepoint, kipp_unicode_space_ranges,
                             KIPP_UNICODE_SPACE_RANGE_COUNT);
}

static bool utf8_decode(const uint8_t *bytes, size_t length, size_t *offset,
                        uint32_t *codepoint, size_t *width) {
    size_t index = *offset;
    uint8_t first;
    if (index >= length) {
        return false;
    }
    first = bytes[index];
    if (first < 0x80) {
        *codepoint = first;
        *width = 1;
    } else if ((first & 0xe0) == 0xc0 && index + 1 < length &&
               (bytes[index + 1] & 0xc0) == 0x80) {
        *codepoint = ((uint32_t)(first & 0x1f) << 6) |
                     (uint32_t)(bytes[index + 1] & 0x3f);
        *width = 2;
        if (*codepoint < 0x80) {
            return false;
        }
    } else if ((first & 0xf0) == 0xe0 && index + 2 < length &&
               (bytes[index + 1] & 0xc0) == 0x80 &&
               (bytes[index + 2] & 0xc0) == 0x80) {
        *codepoint = ((uint32_t)(first & 0x0f) << 12) |
                     ((uint32_t)(bytes[index + 1] & 0x3f) << 6) |
                     (uint32_t)(bytes[index + 2] & 0x3f);
        *width = 3;
        if (*codepoint < 0x800 ||
            (*codepoint >= 0xd800 && *codepoint <= 0xdfff)) {
            return false;
        }
    } else if ((first & 0xf8) == 0xf0 && index + 3 < length &&
               (bytes[index + 1] & 0xc0) == 0x80 &&
               (bytes[index + 2] & 0xc0) == 0x80 &&
               (bytes[index + 3] & 0xc0) == 0x80) {
        *codepoint = ((uint32_t)(first & 0x07) << 18) |
                     ((uint32_t)(bytes[index + 1] & 0x3f) << 12) |
                     ((uint32_t)(bytes[index + 2] & 0x3f) << 6) |
                     (uint32_t)(bytes[index + 3] & 0x3f);
        *width = 4;
        if (*codepoint < 0x10000 || *codepoint > 0x10ffff) {
            return false;
        }
    } else {
        return false;
    }
    *offset += *width;
    return true;
}

static const unicode_decomposition *unicode_find_decomposition(uint32_t codepoint) {
    size_t low = 0;
    size_t high = KIPP_UNICODE_DECOMPOSITION_COUNT;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (codepoint < kipp_unicode_decompositions[middle].codepoint) {
            high = middle;
        } else if (codepoint > kipp_unicode_decompositions[middle].codepoint) {
            low = middle + 1;
        } else {
            return &kipp_unicode_decompositions[middle];
        }
    }
    return NULL;
}

static uint8_t unicode_combining_class(uint32_t codepoint) {
    size_t low = 0;
    size_t high = KIPP_UNICODE_COMBINING_COUNT;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (codepoint < kipp_unicode_combining_classes[middle].codepoint) {
            high = middle;
        } else if (codepoint >
                   kipp_unicode_combining_classes[middle].codepoint) {
            low = middle + 1;
        } else {
            return kipp_unicode_combining_classes[middle].combining_class;
        }
    }
    return 0;
}

static uint32_t unicode_compose_pair(uint32_t first, uint32_t second) {
    enum {
        S_BASE = 0xac00,
        L_BASE = 0x1100,
        V_BASE = 0x1161,
        T_BASE = 0x11a7,
        L_COUNT = 19,
        V_COUNT = 21,
        T_COUNT = 28,
        N_COUNT = V_COUNT * T_COUNT,
        S_COUNT = L_COUNT * N_COUNT
    };
    if (first >= L_BASE && first < L_BASE + L_COUNT &&
        second >= V_BASE && second < V_BASE + V_COUNT) {
        return S_BASE + ((first - L_BASE) * V_COUNT + (second - V_BASE)) *
                            T_COUNT;
    }
    if (first >= S_BASE && first < S_BASE + S_COUNT &&
        (first - S_BASE) % T_COUNT == 0 &&
        second > T_BASE && second < T_BASE + T_COUNT) {
        return first + second - T_BASE;
    }

    uint64_t key = ((uint64_t)first << 32) | second;
    size_t low = 0;
    size_t high = KIPP_UNICODE_COMPOSITION_COUNT;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        uint64_t candidate =
            ((uint64_t)kipp_unicode_compositions[middle].first << 32) |
            kipp_unicode_compositions[middle].second;
        if (key < candidate) {
            high = middle;
        } else if (key > candidate) {
            low = middle + 1;
        } else {
            return kipp_unicode_compositions[middle].result;
        }
    }
    return 0;
}

static int append_codepoint(uint32_t **codepoints, size_t *count,
                            size_t *capacity, uint32_t codepoint) {
    if (*count == *capacity) {
        size_t new_capacity = *capacity == 0 ? 32 : *capacity * 2;
        uint32_t *resized;
        if (new_capacity < *capacity ||
            new_capacity > SIZE_MAX / sizeof(**codepoints)) {
            return -1;
        }
        resized = realloc(*codepoints, new_capacity * sizeof(**codepoints));
        if (resized == NULL) {
            return -1;
        }
        *codepoints = resized;
        *capacity = new_capacity;
    }
    (*codepoints)[(*count)++] = codepoint;
    return 0;
}

static int canonical_decompose(uint32_t codepoint, uint32_t **output,
                               size_t *count, size_t *capacity,
                               unsigned depth) {
    enum {
        S_BASE = 0xac00,
        L_BASE = 0x1100,
        V_BASE = 0x1161,
        T_BASE = 0x11a7,
        V_COUNT = 21,
        T_COUNT = 28,
        N_COUNT = V_COUNT * T_COUNT,
        S_COUNT = 19 * N_COUNT
    };
    if (depth > 16) {
        return -1;
    }
    if (codepoint >= S_BASE && codepoint < S_BASE + S_COUNT) {
        uint32_t index = codepoint - S_BASE;
        uint32_t leading = L_BASE + index / N_COUNT;
        uint32_t vowel = V_BASE + (index % N_COUNT) / T_COUNT;
        uint32_t trailing = T_BASE + index % T_COUNT;
        if (append_codepoint(output, count, capacity, leading) != 0 ||
            append_codepoint(output, count, capacity, vowel) != 0 ||
            (trailing != T_BASE &&
             append_codepoint(output, count, capacity, trailing) != 0)) {
            return -1;
        }
        return 0;
    }
    const unicode_decomposition *decomposition =
        unicode_find_decomposition(codepoint);
    if (decomposition == NULL) {
        return append_codepoint(output, count, capacity, codepoint);
    }
    if (canonical_decompose(decomposition->first, output, count, capacity,
                            depth + 1) != 0) {
        return -1;
    }
    if (decomposition->count == 2 &&
        canonical_decompose(decomposition->second, output, count, capacity,
                            depth + 1) != 0) {
        return -1;
    }
    return 0;
}

static size_t utf8_encoded_width(uint32_t codepoint) {
    if (codepoint < 0x80) {
        return 1;
    }
    if (codepoint < 0x800) {
        return 2;
    }
    if (codepoint < 0x10000) {
        return 3;
    }
    return 4;
}

static size_t utf8_encode(uint8_t *output, uint32_t codepoint) {
    if (codepoint < 0x80) {
        output[0] = (uint8_t)codepoint;
        return 1;
    }
    if (codepoint < 0x800) {
        output[0] = (uint8_t)(0xc0 | (codepoint >> 6));
        output[1] = (uint8_t)(0x80 | (codepoint & 0x3f));
        return 2;
    }
    if (codepoint < 0x10000) {
        output[0] = (uint8_t)(0xe0 | (codepoint >> 12));
        output[1] = (uint8_t)(0x80 | ((codepoint >> 6) & 0x3f));
        output[2] = (uint8_t)(0x80 | (codepoint & 0x3f));
        return 3;
    }
    output[0] = (uint8_t)(0xf0 | (codepoint >> 18));
    output[1] = (uint8_t)(0x80 | ((codepoint >> 12) & 0x3f));
    output[2] = (uint8_t)(0x80 | ((codepoint >> 6) & 0x3f));
    output[3] = (uint8_t)(0x80 | (codepoint & 0x3f));
    return 4;
}

static int normalize_nfc(const uint8_t *input, size_t input_length,
                         uint8_t **output, size_t *output_length) {
    uint32_t *codepoints = NULL;
    size_t codepoint_count = 0;
    size_t codepoint_capacity = 0;
    size_t input_offset = 0;
    uint8_t *encoded = NULL;
    size_t encoded_length = 0;

    while (input_offset < input_length) {
        uint32_t codepoint;
        size_t width;
        if (!utf8_decode(input, input_length, &input_offset, &codepoint, &width) ||
            canonical_decompose(codepoint, &codepoints, &codepoint_count,
                                &codepoint_capacity, 0) != 0) {
            free(codepoints);
            return -1;
        }
    }

    for (size_t index = 1; index < codepoint_count; ++index) {
        uint8_t combining_class = unicode_combining_class(codepoints[index]);
        if (combining_class == 0) {
            continue;
        }
        size_t position = index;
        while (position > 0) {
            uint8_t previous_class =
                unicode_combining_class(codepoints[position - 1]);
            if (previous_class == 0 || previous_class <= combining_class) {
                break;
            }
            uint32_t temporary = codepoints[position - 1];
            codepoints[position - 1] = codepoints[position];
            codepoints[position] = temporary;
            --position;
        }
    }

    if (codepoint_count > 0) {
        size_t output_count = 1;
        size_t starter_position = 0;
        uint32_t starter = codepoints[0];
        uint8_t last_class = 0;
        for (size_t index = 1; index < codepoint_count; ++index) {
            uint32_t codepoint = codepoints[index];
            uint8_t combining_class = unicode_combining_class(codepoint);
            uint32_t composite = unicode_compose_pair(starter, codepoint);
            if (composite != 0 &&
                (last_class < combining_class || last_class == 0)) {
                codepoints[starter_position] = composite;
                starter = composite;
            } else {
                if (combining_class == 0) {
                    starter_position = output_count;
                    starter = codepoint;
                }
                codepoints[output_count++] = codepoint;
                last_class = combining_class;
            }
        }
        codepoint_count = output_count;
    }

    for (size_t index = 0; index < codepoint_count; ++index) {
        size_t width = utf8_encoded_width(codepoints[index]);
        if (!checked_add_size(encoded_length, width, &encoded_length)) {
            free(codepoints);
            return -1;
        }
    }
    encoded = malloc(encoded_length + 1);
    if (encoded == NULL) {
        free(codepoints);
        return -1;
    }
    size_t offset = 0;
    for (size_t index = 0; index < codepoint_count; ++index) {
        offset += utf8_encode(encoded + offset, codepoints[index]);
    }
    encoded[encoded_length] = '\0';
    free(codepoints);
    *output = encoded;
    *output_length = encoded_length;
    return 0;
}

typedef struct {
    uint32_t codepoint;
    size_t byte_offset;
    size_t byte_width;
} decoded_codepoint;

typedef struct {
    size_t offset;
    size_t length;
} byte_span;

static int decode_text(const uint8_t *bytes, size_t length,
                       decoded_codepoint **output, size_t *output_count) {
    decoded_codepoint *points = malloc((length == 0 ? 1 : length) * sizeof(*points));
    size_t count = 0;
    size_t offset = 0;
    if (points == NULL) {
        return -1;
    }
    while (offset < length) {
        size_t start = offset;
        uint32_t codepoint;
        size_t width;
        if (!utf8_decode(bytes, length, &offset, &codepoint, &width)) {
            free(points);
            return -1;
        }
        points[count++] = (decoded_codepoint){codepoint, start, width};
    }
    *output = points;
    *output_count = count;
    return 0;
}

static bool ascii_equal_ci(uint32_t codepoint, char expected) {
    if (codepoint >= 'A' && codepoint <= 'Z') {
        codepoint += 'a' - 'A';
    }
    return codepoint == (uint32_t)(unsigned char)expected;
}

static size_t contraction_length(const decoded_codepoint *points, size_t count,
                                 size_t index) {
    static const char *suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    if (points[index].codepoint != '\'') {
        return 0;
    }
    for (size_t suffix_index = 0;
         suffix_index < sizeof(suffixes) / sizeof(suffixes[0]); ++suffix_index) {
        size_t suffix_length = strlen(suffixes[suffix_index]);
        if (index + 1 + suffix_length > count) {
            continue;
        }
        bool matches = true;
        for (size_t character = 0; character < suffix_length; ++character) {
            if (!ascii_equal_ci(points[index + 1 + character].codepoint,
                                suffixes[suffix_index][character])) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return suffix_length + 1;
        }
    }
    return 0;
}

static int pretokenize_bytes(const uint8_t *bytes, size_t length,
                             byte_span **output, size_t *output_count) {
    decoded_codepoint *points = NULL;
    size_t point_count = 0;
    byte_span *spans;
    size_t span_count = 0;
    size_t index = 0;
    if (decode_text(bytes, length, &points, &point_count) != 0) {
        return -1;
    }
    spans = malloc((point_count == 0 ? 1 : point_count) * sizeof(*spans));
    if (spans == NULL) {
        free(points);
        return -1;
    }

    while (index < point_count) {
        size_t start = index;
        size_t contraction = contraction_length(points, point_count, index);
        if (contraction != 0) {
            index += contraction;
        } else if (unicode_is_letter(points[index].codepoint)) {
            while (index < point_count &&
                   unicode_is_letter(points[index].codepoint)) {
                ++index;
            }
        } else if (index + 1 < point_count &&
                   points[index].codepoint != '\r' &&
                   points[index].codepoint != '\n' &&
                   !unicode_is_letter(points[index].codepoint) &&
                   !unicode_is_number(points[index].codepoint) &&
                   unicode_is_letter(points[index + 1].codepoint)) {
            ++index;
            while (index < point_count &&
                   unicode_is_letter(points[index].codepoint)) {
                ++index;
            }
        } else if (unicode_is_number(points[index].codepoint)) {
            ++index;
        } else if (!unicode_is_space(points[index].codepoint) ||
                   (points[index].codepoint == ' ' &&
                    index + 1 < point_count &&
                    !unicode_is_space(points[index + 1].codepoint) &&
                    !unicode_is_letter(points[index + 1].codepoint) &&
                    !unicode_is_number(points[index + 1].codepoint))) {
            if (points[index].codepoint == ' ' && index + 1 < point_count &&
                !unicode_is_space(points[index + 1].codepoint) &&
                !unicode_is_letter(points[index + 1].codepoint) &&
                !unicode_is_number(points[index + 1].codepoint)) {
                ++index;
            }
            while (index < point_count &&
                   !unicode_is_space(points[index].codepoint) &&
                   !unicode_is_letter(points[index].codepoint) &&
                   !unicode_is_number(points[index].codepoint)) {
                ++index;
            }
            while (index < point_count &&
                   (points[index].codepoint == '\r' ||
                    points[index].codepoint == '\n')) {
                ++index;
            }
        } else {
            size_t whitespace_end = index;
            size_t last_newline_end = 0;
            while (whitespace_end < point_count &&
                   unicode_is_space(points[whitespace_end].codepoint)) {
                if (points[whitespace_end].codepoint == '\r' ||
                    points[whitespace_end].codepoint == '\n') {
                    last_newline_end = whitespace_end + 1;
                }
                ++whitespace_end;
            }
            if (last_newline_end != 0) {
                index = last_newline_end;
            } else if (whitespace_end < point_count &&
                       whitespace_end - index > 1) {
                index = whitespace_end - 1;
            } else {
                index = whitespace_end;
            }
        }
        if (index == start) {
            ++index;
        }
        size_t byte_start = points[start].byte_offset;
        size_t byte_end = index < point_count
                              ? points[index].byte_offset
                              : length;
        spans[span_count++] = (byte_span){byte_start, byte_end - byte_start};
    }
    free(points);
    *output = spans;
    *output_count = span_count;
    return 0;
}

static int append_token(uint32_t **tokens, size_t *count, size_t *capacity,
                        uint32_t token) {
    if (*count == *capacity) {
        size_t new_capacity = *capacity == 0 ? 32 : *capacity * 2;
        uint32_t *resized;
        if (new_capacity < *capacity ||
            new_capacity > SIZE_MAX / sizeof(**tokens)) {
            return -1;
        }
        resized = realloc(*tokens, new_capacity * sizeof(**tokens));
        if (resized == NULL) {
            return -1;
        }
        *tokens = resized;
        *capacity = new_capacity;
    }
    (*tokens)[(*count)++] = token;
    return 0;
}

static int bpe_piece(const kipp_tokenizer *tokenizer, const uint8_t *bytes,
                     size_t length, uint32_t **output, size_t *output_count,
                     size_t *output_capacity, kipp_error *error) {
    uint32_t *piece = malloc((length == 0 ? 1 : length) * sizeof(*piece));
    size_t piece_count = length;
    if (piece == NULL) {
        return fail(error, KIPP_ERROR_MEMORY, "unable to allocate BPE piece");
    }
    for (size_t index = 0; index < length; ++index) {
        piece[index] = vocab_lookup(tokenizer, bytes + index, 1);
        if (piece[index] == UINT32_MAX) {
            free(piece);
            return fail(error, KIPP_ERROR_FORMAT,
                        "tokenizer is missing byte token 0x%02x", bytes[index]);
        }
    }
    while (piece_count > 1) {
        const tokenizer_merge *best = NULL;
        size_t best_index = 0;
        for (size_t index = 0; index + 1 < piece_count; ++index) {
            const tokenizer_merge *candidate =
                merge_lookup(tokenizer, piece[index], piece[index + 1]);
            if (candidate != NULL &&
                (best == NULL || candidate->rank < best->rank)) {
                best = candidate;
                best_index = index;
            }
        }
        if (best == NULL) {
            break;
        }
        piece[best_index] = best->result;
        memmove(piece + best_index + 1, piece + best_index + 2,
                (piece_count - best_index - 2) * sizeof(*piece));
        --piece_count;
    }
    for (size_t index = 0; index < piece_count; ++index) {
        if (append_token(output, output_count, output_capacity, piece[index]) != 0) {
            free(piece);
            return fail(error, KIPP_ERROR_MEMORY, "unable to grow token output");
        }
    }
    free(piece);
    return 0;
}

static size_t match_special_token(const kipp_tokenizer *tokenizer,
                                  const uint8_t *bytes, size_t length,
                                  uint32_t *token_id) {
    size_t best_length = 0;
    uint32_t best_id = UINT32_MAX;
    for (size_t index = 0; index < tokenizer->special_token_count; ++index) {
        uint32_t candidate = tokenizer->special_tokens[index];
        const tokenizer_token *token = &tokenizer->tokens[candidate];
        if (token->length > best_length && token->length <= length &&
            memcmp(token->bytes, bytes, token->length) == 0) {
            best_length = token->length;
            best_id = candidate;
        }
    }
    *token_id = best_id;
    return best_length;
}

static int tokenize_normal_span(const kipp_tokenizer *tokenizer,
                                const uint8_t *bytes, size_t length,
                                uint32_t **output, size_t *output_count,
                                size_t *output_capacity, kipp_error *error) {
    byte_span *spans = NULL;
    size_t span_count = 0;
    if (pretokenize_bytes(bytes, length, &spans, &span_count) != 0) {
        return fail(error, KIPP_ERROR_ARGUMENT, "input is not valid UTF-8");
    }
    for (size_t index = 0; index < span_count; ++index) {
        if (bpe_piece(tokenizer, bytes + spans[index].offset,
                      spans[index].length, output, output_count,
                      output_capacity, error) != 0) {
            free(spans);
            return -1;
        }
    }
    free(spans);
    return 0;
}

int kipp_tokenize(const kipp_model *model, const char *text,
                  kipp_tokens *out_tokens, kipp_error *error) {
    const uint8_t *bytes;
    uint8_t *normalized = NULL;
    size_t length;
    size_t normal_start = 0;
    size_t offset = 0;
    uint32_t *tokens = NULL;
    size_t token_count = 0;
    size_t token_capacity = 0;
    clear_error(error);
    if (model == NULL || text == NULL || out_tokens == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "model, text, and output tokens are required");
    }
    out_tokens->data = NULL;
    out_tokens->count = 0;
    if (normalize_nfc((const uint8_t *)text, strlen(text), &normalized,
                      &length) != 0) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "input is not valid normalizable UTF-8");
    }
    bytes = normalized;

    while (offset < length) {
        uint32_t special_id;
        size_t special_length =
            match_special_token(&model->tokenizer, bytes + offset,
                                length - offset, &special_id);
        if (special_length == 0) {
            ++offset;
            continue;
        }
        if (offset > normal_start &&
            tokenize_normal_span(&model->tokenizer, bytes + normal_start,
                                 offset - normal_start, &tokens, &token_count,
                                 &token_capacity, error) != 0) {
            free(tokens);
            free(normalized);
            return -1;
        }
        if (append_token(&tokens, &token_count, &token_capacity, special_id) != 0) {
            free(tokens);
            free(normalized);
            return fail(error, KIPP_ERROR_MEMORY, "unable to grow token output");
        }
        offset += special_length;
        normal_start = offset;
    }
    if (normal_start < length &&
        tokenize_normal_span(&model->tokenizer, bytes + normal_start,
                             length - normal_start, &tokens, &token_count,
                             &token_capacity, error) != 0) {
        free(tokens);
        free(normalized);
        return -1;
    }
    free(normalized);
    out_tokens->data = tokens;
    out_tokens->count = token_count;
    return 0;
}

int kipp_detokenize(const kipp_model *model, const uint32_t *tokens,
                    size_t token_count, char **out_text, size_t *out_length,
                    kipp_error *error) {
    size_t total = 0;
    char *text;
    clear_error(error);
    if (model == NULL || (token_count != 0 && tokens == NULL) ||
        out_text == NULL || out_length == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT, "invalid detokenize arguments");
    }
    *out_text = NULL;
    *out_length = 0;
    for (size_t index = 0; index < token_count; ++index) {
        if (tokens[index] >= model->tokenizer.token_count ||
            !checked_add_size(total,
                              model->tokenizer.tokens[tokens[index]].length,
                              &total)) {
            return fail(error, KIPP_ERROR_RANGE, "invalid detokenize token");
        }
    }
    text = malloc(total + 1);
    if (text == NULL) {
        return fail(error, KIPP_ERROR_MEMORY, "unable to allocate text output");
    }
    size_t offset = 0;
    for (size_t index = 0; index < token_count; ++index) {
        const tokenizer_token *token = &model->tokenizer.tokens[tokens[index]];
        memcpy(text + offset, token->bytes, token->length);
        offset += token->length;
    }
    text[total] = '\0';
    *out_text = text;
    *out_length = total;
    return 0;
}

void kipp_tokens_free(kipp_tokens *tokens) {
    if (tokens != NULL) {
        free(tokens->data);
        tokens->data = NULL;
        tokens->count = 0;
    }
}

void kipp_text_free(char *text) {
    free(text);
}

static float bf16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

/* IEEE binary16 -> float, handling subnormals, inf, and NaN. */
static float fp16_to_float(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            /* Subnormal: normalize into a float. */
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static const uint16_t *tensor_bf16(const kipp_tensor_view *tensor) {
    return (const uint16_t *)(const void *)tensor->data;
}

static void rms_norm(const float *input, const uint16_t *weight, float *output,
                     size_t length, float epsilon) {
    float square_sum = 0.0f;
    for (size_t index = 0; index < length; ++index) {
        square_sum += input[index] * input[index];
    }
    float scale = 1.0f / sqrtf(square_sum / (float)length + epsilon);
    for (size_t index = 0; index < length; ++index) {
        output[index] = input[index] * scale * bf16_to_float(weight[index]);
    }
}

static void matvec_bf16(const uint16_t *weight, const float *input,
                        float *output, size_t rows, size_t columns) {
    for (size_t row = 0; row < rows; ++row) {
        const uint16_t *weight_row = weight + row * columns;
        float sum = 0.0f;
        for (size_t column = 0; column < columns; ++column) {
            sum += bf16_to_float(weight_row[column]) * input[column];
        }
        output[row] = (float)sum;
    }
}

/*
 * Q8_0: each 32-weight block is a little-endian fp16 scale followed by 32
 * int8 quants (34 bytes). w = scale * q. Row r begins at block r*(cols/32).
 */
static void matvec_q8_0(const uint8_t *weight, const float *input,
                        float *output, size_t rows, size_t columns) {
    const size_t blocks = columns / KIPP_QUANT_BLOCK;
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t *block =
            weight + row * blocks * KIPP_Q8_0_BLOCK_BYTES;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; ++b) {
            uint16_t scale_bits;
            memcpy(&scale_bits, block, sizeof(scale_bits));
            float scale = fp16_to_float(scale_bits);
            const int8_t *qs = (const int8_t *)(block + 2);
            const float *in = input + b * KIPP_QUANT_BLOCK;
            float dot = 0.0f;
            for (size_t i = 0; i < KIPP_QUANT_BLOCK; ++i) {
                dot += (float)qs[i] * in[i];
            }
            sum += scale * dot;
            block += KIPP_Q8_0_BLOCK_BYTES;
        }
        output[row] = sum;
    }
}

/*
 * AFFINE4_GS32: each 32-weight group is 16 packed nibbles (q[2k]=lo,
 * q[2k+1]=hi) then a little-endian fp16 scale and fp16 bias (20 bytes).
 * w = scale*q + bias, so the group contributes scale*sum(q_i*in_i) +
 * bias*sum(in_i) (the activation-sum fold).
 */
static void matvec_affine4_gs32(const uint8_t *weight, const float *input,
                                float *output, size_t rows, size_t columns) {
    const size_t groups = columns / KIPP_QUANT_BLOCK;
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t *group =
            weight + row * groups * KIPP_AFFINE4_GROUP_BYTES;
        float sum = 0.0f;
        for (size_t g = 0; g < groups; ++g) {
            const uint8_t *packed = group;
            uint16_t scale_bits;
            uint16_t bias_bits;
            memcpy(&scale_bits, group + 16, sizeof(scale_bits));
            memcpy(&bias_bits, group + 18, sizeof(bias_bits));
            float scale = fp16_to_float(scale_bits);
            float bias = fp16_to_float(bias_bits);
            const float *in = input + g * KIPP_QUANT_BLOCK;
            float dot = 0.0f;
            float activation_sum = 0.0f;
            for (size_t k = 0; k < 16; ++k) {
                uint8_t byte = packed[k];
                float lo = (float)(byte & 0x0fu);
                float hi = (float)(byte >> 4);
                dot += lo * in[2 * k] + hi * in[2 * k + 1];
                activation_sum += in[2 * k] + in[2 * k + 1];
            }
            sum += scale * dot + bias * activation_sum;
            group += KIPP_AFFINE4_GROUP_BYTES;
        }
        output[row] = sum;
    }
}

/* Dispatch a projection matvec on the weight tensor's quantization type. */
static void matvec_tensor(const kipp_tensor_view *weight, const float *input,
                          float *output, size_t rows, size_t columns) {
    switch (weight->type) {
    case KIPP_TENSOR_Q8_0:
        matvec_q8_0(weight->data, input, output, rows, columns);
        break;
    case KIPP_TENSOR_AFFINE4_GS32:
        matvec_affine4_gs32(weight->data, input, output, rows, columns);
        break;
    case KIPP_TENSOR_BF16:
    default:
        matvec_bf16((const uint16_t *)(const void *)weight->data, input,
                    output, rows, columns);
        break;
    }
}

static void rope_head(float *head, size_t head_dim, uint32_t position,
                      float theta) {
    size_t half = head_dim / 2;
    for (size_t index = 0; index < half; ++index) {
        float frequency =
            powf(theta, -(2.0f * (float)index) / (float)head_dim);
        float angle = (float)position * frequency;
        float cosine = cosf(angle);
        float sine = sinf(angle);
        float first = head[index];
        float second = head[index + half];
        head[index] = first * cosine - second * sine;
        head[index + half] = second * cosine + first * sine;
    }
}

static void softmax(float *values, size_t count) {
    float maximum = -FLT_MAX;
    float sum = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        if (values[index] > maximum) {
            maximum = values[index];
        }
    }
    for (size_t index = 0; index < count; ++index) {
        values[index] = expf(values[index] - maximum);
        sum += values[index];
    }
    float inverse = 1.0f / sum;
    for (size_t index = 0; index < count; ++index) {
        values[index] *= inverse;
    }
}

static float silu(float value) {
    return value / (1.0f + expf(-value));
}

typedef struct {
    float *x;
    float *normalized;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *projection;
    float *gate;
    float *up;
    float *scores;
} cpu_workspace;

static void free_workspace(cpu_workspace *workspace) {
    free(workspace->x);
    free(workspace->normalized);
    free(workspace->query);
    free(workspace->key);
    free(workspace->value);
    free(workspace->attention);
    free(workspace->projection);
    free(workspace->gate);
    free(workspace->up);
    free(workspace->scores);
    memset(workspace, 0, sizeof(*workspace));
}

static int allocate_floats(float **pointer, size_t first, size_t second,
                           kipp_error *error) {
    size_t count;
    size_t bytes;
    if (!checked_multiply_size(first, second, &count) ||
        !checked_multiply_size(count, sizeof(float), &bytes)) {
        return fail(error, KIPP_ERROR_MEMORY, "workspace size overflow");
    }
    *pointer = malloc(bytes == 0 ? 1 : bytes);
    if (*pointer == NULL) {
        return fail(error, KIPP_ERROR_MEMORY, "unable to allocate CPU workspace");
    }
    return 0;
}

static int allocate_workspace(cpu_workspace *workspace,
                              const kipp_model_config *config,
                              size_t token_count, size_t score_count,
                              kipp_error *error) {
    memset(workspace, 0, sizeof(*workspace));
    if (allocate_floats(&workspace->x, token_count, config->embedding_length,
                        error) != 0 ||
        allocate_floats(&workspace->normalized, token_count,
                        config->embedding_length, error) != 0 ||
        allocate_floats(&workspace->query, token_count,
                        config->attention_width, error) != 0 ||
        allocate_floats(&workspace->key, token_count,
                        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
                        error) != 0 ||
        allocate_floats(&workspace->value, token_count,
                        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
                        error) != 0 ||
        allocate_floats(&workspace->attention, token_count,
                        config->attention_width, error) != 0 ||
        allocate_floats(&workspace->projection, token_count,
                        config->embedding_length, error) != 0 ||
        allocate_floats(&workspace->gate, token_count,
                        config->feed_forward_length, error) != 0 ||
        allocate_floats(&workspace->up, token_count,
                        config->feed_forward_length, error) != 0 ||
        allocate_floats(&workspace->scores, 1, score_count, error) != 0) {
        free_workspace(workspace);
        return -1;
    }
    return 0;
}

static void normalize_heads(float *states, size_t token_count,
                            size_t head_count, const uint16_t *weight) {
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t head = 0; head < head_count; ++head) {
            float *values =
                states + (token * head_count + head) * KIPP_ATTENTION_HEAD_DIM;
            float square_sum = 0.0f;
            for (size_t dimension = 0; dimension < KIPP_ATTENTION_HEAD_DIM;
                 ++dimension) {
                square_sum += values[dimension] * values[dimension];
            }
            float scale =
                1.0f / sqrtf(square_sum / (float)KIPP_ATTENTION_HEAD_DIM +
                             KIPP_RMS_EPSILON);
            for (size_t dimension = 0; dimension < KIPP_ATTENTION_HEAD_DIM;
                 ++dimension) {
                values[dimension] *= scale * bf16_to_float(weight[dimension]);
            }
        }
    }
}

static void apply_rope(float *states, size_t token_count, size_t head_count,
                       uint32_t start_position, float theta) {
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t head = 0; head < head_count; ++head) {
            rope_head(states + (token * head_count + head) *
                                   KIPP_ATTENTION_HEAD_DIM,
                      KIPP_ATTENTION_HEAD_DIM,
                      start_position + (uint32_t)token, theta);
        }
    }
}

static void round_trip_bf16(float *values, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        values[index] = bf16_to_float(float_to_bf16(values[index]));
    }
}

static void causal_gqa(const float *query, const float *key, const float *value,
                       float *output, float *scores, size_t token_count,
                       size_t query_head_count) {
    const float scale = 1.0f / sqrtf((float)KIPP_ATTENTION_HEAD_DIM);
    const size_t queries_per_kv =
        query_head_count / KIPP_ATTENTION_HEAD_COUNT_KV;
    memset(output, 0, token_count * query_head_count *
                              KIPP_ATTENTION_HEAD_DIM * sizeof(*output));
    for (size_t token = 0; token < token_count; ++token) {
        for (size_t query_head = 0; query_head < query_head_count;
             ++query_head) {
            size_t kv_head = query_head / queries_per_kv;
            const float *query_values =
                query + (token * query_head_count + query_head) *
                            KIPP_ATTENTION_HEAD_DIM;
            for (size_t source = 0; source <= token; ++source) {
                const float *key_values =
                    key + (source * KIPP_ATTENTION_HEAD_COUNT_KV + kv_head) *
                              KIPP_ATTENTION_HEAD_DIM;
                float dot = 0.0f;
                for (size_t dimension = 0;
                     dimension < KIPP_ATTENTION_HEAD_DIM; ++dimension) {
                    dot += query_values[dimension] * key_values[dimension];
                }
                scores[source] = dot * scale;
            }
            softmax(scores, token + 1);
            float *destination =
                output + (token * query_head_count + query_head) *
                             KIPP_ATTENTION_HEAD_DIM;
            for (size_t source = 0; source <= token; ++source) {
                const float *value_values =
                    value + (source * KIPP_ATTENTION_HEAD_COUNT_KV + kv_head) *
                                KIPP_ATTENTION_HEAD_DIM;
                for (size_t dimension = 0;
                     dimension < KIPP_ATTENTION_HEAD_DIM; ++dimension) {
                    destination[dimension] +=
                        scores[source] * value_values[dimension];
                }
            }
        }
    }
}

static int cpu_forward(const kipp_model_view *view, const uint32_t *tokens,
                       size_t token_count, float *logits, kipp_error *error) {
    cpu_workspace workspace;
    const kipp_model_config *config = &view->config;
    const size_t embed = config->embedding_length;
    const size_t attention_width = config->attention_width;
    const size_t feed_forward = config->feed_forward_length;
    const size_t kv_width =
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    const uint16_t *embedding =
        tensor_bf16(&view->weights.token_embedding);
    if (allocate_workspace(&workspace, config, token_count, token_count,
                           error) != 0) {
        return -1;
    }
    for (size_t token = 0; token < token_count; ++token) {
        const uint16_t *row = embedding + (size_t)tokens[token] * embed;
        for (size_t dimension = 0; dimension < embed; ++dimension) {
            workspace.x[token * embed + dimension] =
                bf16_to_float(row[dimension]);
        }
    }

    for (size_t layer_index = 0; layer_index < config->block_count;
         ++layer_index) {
        const kipp_qwen3_layer_weights *layer =
            &view->weights.layers[layer_index];
        for (size_t token = 0; token < token_count; ++token) {
            float *x = workspace.x + token * embed;
            float *normalized = workspace.normalized + token * embed;
            rms_norm(x, tensor_bf16(&layer->attention_norm), normalized,
                     embed, KIPP_RMS_EPSILON);
            matvec_tensor(&layer->attention_q, normalized,
                        workspace.query + token * attention_width,
                        attention_width, embed);
            matvec_tensor(&layer->attention_k, normalized,
                        workspace.key + token * kv_width, kv_width, embed);
            matvec_tensor(&layer->attention_v, normalized,
                        workspace.value + token * kv_width, kv_width, embed);
        }
        normalize_heads(workspace.query, token_count,
                        config->attention_head_count,
                        tensor_bf16(&layer->attention_q_norm));
        normalize_heads(workspace.key, token_count,
                        KIPP_ATTENTION_HEAD_COUNT_KV,
                        tensor_bf16(&layer->attention_k_norm));
        apply_rope(workspace.query, token_count,
                   config->attention_head_count, 0, config->rope_theta);
        apply_rope(workspace.key, token_count, KIPP_ATTENTION_HEAD_COUNT_KV,
                   0, config->rope_theta);
        round_trip_bf16(workspace.key, token_count * kv_width);
        round_trip_bf16(workspace.value, token_count * kv_width);
        causal_gqa(workspace.query, workspace.key, workspace.value,
                   workspace.attention, workspace.scores, token_count,
                   config->attention_head_count);

        for (size_t token = 0; token < token_count; ++token) {
            float *x = workspace.x + token * embed;
            matvec_tensor(&layer->attention_output,
                        workspace.attention + token * attention_width,
                        workspace.projection + token * embed, embed,
                        attention_width);
            for (size_t dimension = 0; dimension < embed; ++dimension) {
                x[dimension] +=
                    workspace.projection[token * embed + dimension];
            }
            rms_norm(x, tensor_bf16(&layer->feed_forward_norm),
                     workspace.normalized + token * embed, embed,
                     KIPP_RMS_EPSILON);
            matvec_tensor(&layer->feed_forward_gate,
                        workspace.normalized + token * embed,
                        workspace.gate + token * feed_forward, feed_forward,
                        embed);
            matvec_tensor(&layer->feed_forward_up,
                        workspace.normalized + token * embed,
                        workspace.up + token * feed_forward, feed_forward,
                        embed);
            for (size_t hidden = 0; hidden < feed_forward; ++hidden) {
                size_t index = token * feed_forward + hidden;
                workspace.gate[index] =
                    silu(workspace.gate[index]) * workspace.up[index];
            }
            matvec_tensor(&layer->feed_forward_down,
                        workspace.gate + token * feed_forward,
                        workspace.projection + token * embed, embed,
                        feed_forward);
            for (size_t dimension = 0; dimension < embed; ++dimension) {
                x[dimension] +=
                    workspace.projection[token * embed + dimension];
            }
        }
    }

    float *last = workspace.x + (token_count - 1) * embed;
    rms_norm(last, tensor_bf16(&view->weights.output_norm),
             workspace.normalized, embed, KIPP_RMS_EPSILON);
    matvec_bf16(tensor_bf16(&view->weights.lm_head), workspace.normalized,
                logits, KIPP_VOCAB_SIZE, embed);
    free_workspace(&workspace);
    return 0;
}

typedef struct {
    const kipp_model_view *view;
    /* Pooled mode (config->kv_pool_blocks > 0): one shared KV slab, owned by
     * the backend model; pooled sessions point into it and carry no storage
     * of their own. */
    uint16_t *pool_key_cache;
    uint16_t *pool_value_cache;
    uint32_t pool_slab_positions;
} cpu_backend_model;

/*
 * The KV store is paged: physical storage is a run of fixed-size blocks
 * (KIPP_KV_BLOCK_TOKENS positions each) and `block_table` maps a logical
 * block index (position >> 5) to the physical block that holds it. With the
 * identity table this is byte-for-byte the old contiguous layout; a
 * non-identity table (e.g. shared prefix blocks, or the scramble the paged
 * gate uses) relocates positions without changing any computed value. The
 * per-layer stride is `slab_positions` (capacity rounded up to a whole
 * number of blocks) so physical block ids address cleanly.
 */
typedef struct {
    uint32_t capacity;
    uint32_t length;
    uint32_t block_count;    /* transformer layers */
    uint32_t block_capacity; /* physical blocks allocated */
    uint32_t slab_positions; /* block_capacity * KIPP_KV_BLOCK_TOKENS */
    uint32_t *block_table;    /* logical block -> physical block */
    size_t slab_elements;
    uint16_t *key_cache;
    uint16_t *value_cache;
    cpu_workspace workspace;
    /* Pooled sessions borrow the model slab and the core's block table; the
     * core owns length tracking and per-eval table contents. */
    bool pooled;
} cpu_backend_session;

static uint64_t kv_cache_bytes_for_capacity(uint32_t block_count,
                                            uint32_t capacity) {
    return (uint64_t)capacity * block_count *
           KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM *
           sizeof(uint16_t) * 2u;
}

static size_t kv_cache_offset_for_capacity(uint32_t capacity, uint32_t layer,
                                           uint32_t position, uint32_t head,
                                           uint32_t dimension) {
    return (((size_t)layer * capacity + position) *
                KIPP_ATTENTION_HEAD_COUNT_KV +
            head) *
               KIPP_ATTENTION_HEAD_DIM +
           dimension;
}

static size_t kv_cache_offset(const cpu_backend_session *session,
                              uint32_t layer, uint32_t position,
                              uint32_t head, uint32_t dimension) {
    uint32_t logical_block = position / KIPP_KV_BLOCK_TOKENS;
    uint32_t slot = position % KIPP_KV_BLOCK_TOKENS;
    uint32_t physical = session->block_table[logical_block];
    uint32_t physical_position = physical * KIPP_KV_BLOCK_TOKENS + slot;
    return kv_cache_offset_for_capacity(session->slab_positions, layer,
                                        physical_position, head, dimension);
}

#ifdef KIPP_FAULT_INJECT
/*
 * Mutation-study scaffolding: never compiled into production targets. The
 * KIPP_FAULT environment variable seeds one deliberate KV-addressing bug so
 * the gates' detection power can be measured. Faults are asymmetric (read
 * side or write side only) because a mapping error applied identically to
 * both sides is a consistent relocation, which placement invariance proves
 * harmless — real paging bugs live in one side's copy of the indirection.
 *
 *   1  read-block:  attention reads resolve the next logical block.
 *   2  read-slot:   attention reads resolve the next slot within the block.
 *   3  rollover:    each block's first position is written one past the end
 *                   of the previous logical block's physical block. Under
 *                   the identity table those bytes land exactly where the
 *                   position belongs, so no identity-mapped comparison can
 *                   see it; any non-identity table corrupts.
 *   4  swap-kv:     layer 0 writes keys into the value store and values
 *                   into the key store.
 */
static int kipp_fault_selected(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *value = getenv("KIPP_FAULT");
        mode = value == NULL ? 0 : atoi(value);
    }
    return mode;
}

static size_t kv_cache_offset_read(const cpu_backend_session *session,
                                   uint32_t layer, uint32_t position,
                                   uint32_t head, uint32_t dimension) {
    uint32_t logical_block = position / KIPP_KV_BLOCK_TOKENS;
    uint32_t slot = position % KIPP_KV_BLOCK_TOKENS;
    if (kipp_fault_selected() == 1) {
        logical_block = (logical_block + 1u) % session->block_capacity;
    } else if (kipp_fault_selected() == 2) {
        slot = (slot + 1u) % KIPP_KV_BLOCK_TOKENS;
    } else {
        return kv_cache_offset(session, layer, position, head, dimension);
    }
    uint32_t physical = session->block_table[logical_block];
    uint32_t physical_position = physical * KIPP_KV_BLOCK_TOKENS + slot;
    return kv_cache_offset_for_capacity(session->slab_positions, layer,
                                        physical_position, head, dimension);
}

static size_t kv_cache_offset_write(const cpu_backend_session *session,
                                    uint32_t layer, uint32_t position,
                                    uint32_t head, uint32_t dimension) {
    uint32_t logical_block = position / KIPP_KV_BLOCK_TOKENS;
    uint32_t slot = position % KIPP_KV_BLOCK_TOKENS;
    if (kipp_fault_selected() == 3 && slot == 0u && logical_block > 0u) {
        uint32_t stale = session->block_table[logical_block - 1u];
        uint32_t physical_position =
            (stale * KIPP_KV_BLOCK_TOKENS + KIPP_KV_BLOCK_TOKENS) %
            session->slab_positions;
        return kv_cache_offset_for_capacity(session->slab_positions, layer,
                                            physical_position, head,
                                            dimension);
    }
    return kv_cache_offset(session, layer, position, head, dimension);
}
#else
#define kv_cache_offset_read kv_cache_offset
#define kv_cache_offset_write kv_cache_offset
#endif

static void cached_gqa(const cpu_backend_session *session, uint32_t layer,
                       uint32_t position, const float *query, float *output,
                       float *scores, uint32_t query_head_count) {
    const float scale = 1.0f / sqrtf((float)KIPP_ATTENTION_HEAD_DIM);
    const uint32_t queries_per_kv =
        query_head_count / KIPP_ATTENTION_HEAD_COUNT_KV;
    memset(output, 0, (size_t)query_head_count * KIPP_ATTENTION_HEAD_DIM *
                          sizeof(*output));
    for (uint32_t query_head = 0; query_head < query_head_count;
         ++query_head) {
        uint32_t kv_head = query_head / queries_per_kv;
        const float *query_values =
            query + (size_t)query_head * KIPP_ATTENTION_HEAD_DIM;
        for (uint32_t source = 0; source <= position; ++source) {
            size_t cache_base =
                kv_cache_offset_read(session, layer, source, kv_head, 0);
            float dot = 0.0f;
            for (uint32_t dimension = 0;
                 dimension < KIPP_ATTENTION_HEAD_DIM; ++dimension) {
                dot += query_values[dimension] *
                       bf16_to_float(session->key_cache[cache_base + dimension]);
            }
            scores[source] = dot * scale;
        }
        softmax(scores, (size_t)position + 1);
        float *destination =
            output + (size_t)query_head * KIPP_ATTENTION_HEAD_DIM;
        for (uint32_t source = 0; source <= position; ++source) {
            size_t cache_base =
                kv_cache_offset_read(session, layer, source, kv_head, 0);
            for (uint32_t dimension = 0;
                 dimension < KIPP_ATTENTION_HEAD_DIM; ++dimension) {
                destination[dimension] +=
                    scores[source] *
                    bf16_to_float(
                        session->value_cache[cache_base + dimension]);
            }
        }
    }
}

static int cpu_cached_token(const kipp_model_view *view,
                            cpu_backend_session *session, uint32_t token,
                            uint32_t position, float *logits,
                            kipp_error *error) {
    cpu_workspace *workspace = &session->workspace;
    const kipp_model_config *config = &view->config;
    const size_t embed = config->embedding_length;
    const size_t attention_width = config->attention_width;
    const size_t feed_forward = config->feed_forward_length;
    const size_t kv_width =
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    const uint16_t *embedding =
        tensor_bf16(&view->weights.token_embedding);
    const uint16_t *embedding_row = embedding + (size_t)token * embed;
    (void)error;

    for (size_t dimension = 0; dimension < embed; ++dimension) {
        workspace->x[dimension] = bf16_to_float(embedding_row[dimension]);
    }

    for (uint32_t layer_index = 0; layer_index < config->block_count;
         ++layer_index) {
        const kipp_qwen3_layer_weights *layer =
            &view->weights.layers[layer_index];
        rms_norm(workspace->x, tensor_bf16(&layer->attention_norm),
                 workspace->normalized, embed, KIPP_RMS_EPSILON);
        matvec_tensor(&layer->attention_q,
                    workspace->normalized, workspace->query,
                    attention_width, embed);
        matvec_tensor(&layer->attention_k,
                    workspace->normalized, workspace->key, kv_width, embed);
        matvec_tensor(&layer->attention_v,
                    workspace->normalized, workspace->value, kv_width, embed);

        normalize_heads(workspace->query, 1, config->attention_head_count,
                        tensor_bf16(&layer->attention_q_norm));
        normalize_heads(workspace->key, 1, KIPP_ATTENTION_HEAD_COUNT_KV,
                        tensor_bf16(&layer->attention_k_norm));
        apply_rope(workspace->query, 1, config->attention_head_count,
                   position, config->rope_theta);
        apply_rope(workspace->key, 1, KIPP_ATTENTION_HEAD_COUNT_KV, position,
                   config->rope_theta);

        uint16_t *key_store = session->key_cache;
        uint16_t *value_store = session->value_cache;
#ifdef KIPP_FAULT_INJECT
        if (kipp_fault_selected() == 4 && layer_index == 0) {
            key_store = session->value_cache;
            value_store = session->key_cache;
        }
#endif
        for (uint32_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT_KV; ++head) {
            size_t cache_base =
                kv_cache_offset_write(session, layer_index, position, head, 0);
            size_t state_base = (size_t)head * KIPP_ATTENTION_HEAD_DIM;
            for (uint32_t dimension = 0;
                 dimension < KIPP_ATTENTION_HEAD_DIM; ++dimension) {
                key_store[cache_base + dimension] =
                    float_to_bf16(workspace->key[state_base + dimension]);
                value_store[cache_base + dimension] =
                    float_to_bf16(workspace->value[state_base + dimension]);
            }
        }
        cached_gqa(session, layer_index, position, workspace->query,
                   workspace->attention, workspace->scores,
                   config->attention_head_count);

        matvec_tensor(&layer->attention_output,
                    workspace->attention, workspace->projection, embed,
                    attention_width);
        for (size_t dimension = 0; dimension < embed; ++dimension) {
            workspace->x[dimension] += workspace->projection[dimension];
        }
        rms_norm(workspace->x, tensor_bf16(&layer->feed_forward_norm),
                 workspace->normalized, embed, KIPP_RMS_EPSILON);
        matvec_tensor(&layer->feed_forward_gate,
                    workspace->normalized, workspace->gate, feed_forward,
                    embed);
        matvec_tensor(&layer->feed_forward_up,
                    workspace->normalized, workspace->up, feed_forward,
                    embed);
        for (size_t hidden = 0; hidden < feed_forward; ++hidden) {
            workspace->gate[hidden] =
                silu(workspace->gate[hidden]) * workspace->up[hidden];
        }
        matvec_tensor(&layer->feed_forward_down, workspace->gate,
                    workspace->projection, embed, feed_forward);
        for (size_t dimension = 0; dimension < embed; ++dimension) {
            workspace->x[dimension] += workspace->projection[dimension];
        }
    }

    if (logits != NULL) {
        rms_norm(workspace->x, tensor_bf16(&view->weights.output_norm),
                 workspace->normalized, embed, KIPP_RMS_EPSILON);
        matvec_bf16(tensor_bf16(&view->weights.lm_head),
                    workspace->normalized, logits, KIPP_VOCAB_SIZE, embed);
    }
    return 0;
}

static int cpu_backend_model_create(const kipp_model_view *view,
                                    void **backend_model, kipp_error *error) {
    cpu_backend_model *state;
    if (view == NULL || backend_model == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "CPU backend model arguments are required");
    }
    state = calloc(1, sizeof(*state));
    if (state == NULL) {
        return fail(error, KIPP_ERROR_MEMORY,
                    "unable to allocate CPU backend model");
    }
    state->view = view;
    if (view->config.kv_pool_blocks != 0) {
        size_t positions =
            (size_t)view->config.kv_pool_blocks * KIPP_KV_BLOCK_TOKENS;
        size_t elements;
        size_t bytes;
        if (positions > UINT32_MAX ||
            !checked_multiply_size(view->config.block_count, positions,
                                   &elements) ||
            !checked_multiply_size(elements, KIPP_ATTENTION_HEAD_COUNT_KV,
                                   &elements) ||
            !checked_multiply_size(elements, KIPP_ATTENTION_HEAD_DIM,
                                   &elements) ||
            !checked_multiply_size(elements, sizeof(uint16_t), &bytes)) {
            free(state);
            return fail(error, KIPP_ERROR_MEMORY, "KV pool size overflows");
        }
        state->pool_slab_positions = (uint32_t)positions;
        state->pool_key_cache = malloc(bytes);
        state->pool_value_cache = malloc(bytes);
        if (state->pool_key_cache == NULL ||
            state->pool_value_cache == NULL) {
            free(state->pool_key_cache);
            free(state->pool_value_cache);
            free(state);
            return fail(error, KIPP_ERROR_MEMORY,
                        "unable to allocate pooled KV slab");
        }
    }
    *backend_model = state;
    return 0;
}

static void cpu_backend_model_destroy(void *backend_model) {
    cpu_backend_model *state = backend_model;
    if (state == NULL) {
        return;
    }
    free(state->pool_key_cache);
    free(state->pool_value_cache);
    free(state);
}

static int cpu_backend_session_create(void *backend_model, uint32_t capacity,
                                      void **backend_session,
                                      kipp_error *error) {
    cpu_backend_session *session;
    size_t elements;
    size_t bytes;
    if (backend_model == NULL || backend_session == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "CPU backend session arguments are required");
    }
    cpu_backend_model *state = backend_model;
    const kipp_model_config *config = &state->view->config;
    *backend_session = NULL;
    if (capacity == 0 || capacity > config->context_length) {
        return fail(error, KIPP_ERROR_RANGE,
                    "session capacity must be between 1 and %u",
                    config->context_length);
    }
    /* Round the physical store up to whole blocks so paged addressing is
     * always in bounds; the reported logical cache size is unchanged. */
    uint32_t block_capacity =
        (capacity + KIPP_KV_BLOCK_TOKENS - 1) / KIPP_KV_BLOCK_TOKENS;
    size_t slab_positions = (size_t)block_capacity * KIPP_KV_BLOCK_TOKENS;
    if (!checked_multiply_size(config->block_count, slab_positions,
                               &elements) ||
        !checked_multiply_size(elements, KIPP_ATTENTION_HEAD_COUNT_KV,
                               &elements) ||
        !checked_multiply_size(elements, KIPP_ATTENTION_HEAD_DIM, &elements) ||
        !checked_multiply_size(elements, sizeof(uint16_t), &bytes)) {
        return fail(error, KIPP_ERROR_MEMORY, "KV cache size overflows");
    }
    session = calloc(1, sizeof(*session));
    if (session == NULL) {
        return fail(error, KIPP_ERROR_MEMORY,
                    "unable to allocate CPU session");
    }
    session->capacity = capacity;
    session->block_count = config->block_count;
    session->block_capacity = block_capacity;
    if (config->kv_pool_blocks != 0) {
        /* Pooled: borrow the model slab; the core supplies a block table
         * with every eval item, so no per-session storage exists. */
        session->pooled = true;
        session->slab_positions = state->pool_slab_positions;
        session->key_cache = state->pool_key_cache;
        session->value_cache = state->pool_value_cache;
        if (allocate_workspace(&session->workspace, config, 1, capacity,
                               error) != 0) {
            free(session);
            return -1;
        }
        *backend_session = session;
        return 0;
    }
    session->slab_positions = (uint32_t)slab_positions;
    session->slab_elements = elements;
    session->block_table = malloc((size_t)block_capacity *
                                  sizeof(*session->block_table));
    session->key_cache = malloc(bytes);
    session->value_cache = malloc(bytes);
    if (session->block_table == NULL || session->key_cache == NULL ||
        session->value_cache == NULL ||
        allocate_workspace(&session->workspace, config, 1, capacity,
                           error) != 0) {
        free(session->block_table);
        free(session->key_cache);
        free(session->value_cache);
        free_workspace(&session->workspace);
        free(session);
        if (error == NULL || error->code == KIPP_OK) {
            return fail(error, KIPP_ERROR_MEMORY,
                        "unable to allocate CPU KV cache");
        }
        return -1;
    }
    /* Identity mapping: logical block i lives in physical block i. */
    for (uint32_t block = 0; block < block_capacity; ++block) {
        session->block_table[block] = block;
    }
    *backend_session = session;
    return 0;
}

static void cpu_backend_session_destroy(void *backend_session) {
    cpu_backend_session *session = backend_session;
    if (session == NULL) {
        return;
    }
    if (!session->pooled) {
        free(session->block_table);
        free(session->key_cache);
        free(session->value_cache);
    }
    free_workspace(&session->workspace);
    free(session);
}

static int cpu_backend_session_reset(void *backend_session, kipp_error *error) {
    cpu_backend_session *session = backend_session;
    if (session == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "CPU backend session is required");
    }
    session->length = 0;
    return 0;
}

static int cpu_backend_session_truncate(void *backend_session,
                                        uint32_t length, kipp_error *error) {
    cpu_backend_session *session = backend_session;
    if (session == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "CPU backend session is required");
    }
    if (session->pooled) {
        /* The core owns pooled length tracking and block release. */
        return 0;
    }
    if (length > session->length) {
        return fail(error, KIPP_ERROR_RANGE,
                    "cannot truncate session of length %u to %u",
                    session->length, length);
    }
    session->length = length;
    return 0;
}

static int cpu_backend_eval(void *backend_model, kipp_eval_item *items,
                            size_t item_count, kipp_error *error) {
    cpu_backend_model *state = backend_model;
    if (state == NULL || items == NULL || item_count == 0) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "CPU backend evaluation items are required");
    }
    for (size_t index = 0; index < item_count; ++index) {
        kipp_eval_item *item = &items[index];
        if (item->tokens == NULL || item->token_count == 0 ||
            item->logits == NULL) {
            return fail(error, KIPP_ERROR_ARGUMENT,
                        "CPU evaluation item is incomplete");
        }
        uint32_t rows = item->logits_count == 0 ? 1 : item->logits_count;
        if (rows > item->token_count) {
            return fail(error, KIPP_ERROR_RANGE,
                        "logits_count %u exceeds token_count %u", rows,
                        item->token_count);
        }
        if (item->session == NULL) {
            if (item->start_position != 0) {
                return fail(error, KIPP_ERROR_ARGUMENT,
                            "stateless evaluation must start at position zero");
            }
            if (rows != 1) {
                return fail(error, KIPP_ERROR_UNSUPPORTED,
                            "stateless evaluation writes only the final logits");
            }
            if (cpu_forward(state->view, item->tokens, item->token_count,
                            item->logits, error) != 0) {
                return -1;
            }
            continue;
        }

        cpu_backend_session *session = item->session;
        if (session->pooled) {
            /* The core owns pooled length tracking and supplies the block
             * table with every item. */
            if (item->block_table == NULL) {
                return fail(error, KIPP_ERROR_ARGUMENT,
                            "pooled evaluation requires a block table");
            }
            if (item->token_count > session->capacity - item->start_position) {
                return fail(error, KIPP_ERROR_RANGE,
                            "CPU session append exceeds capacity %u",
                            session->capacity);
            }
            session->block_table = (uint32_t *)item->block_table;
        } else {
            if (item->start_position != session->length) {
                return fail(error, KIPP_ERROR_ARGUMENT,
                            "start position %u does not match session length %u",
                            item->start_position, session->length);
            }
            if (item->token_count > session->capacity - session->length) {
                return fail(error, KIPP_ERROR_RANGE,
                            "CPU session append exceeds capacity %u",
                            session->capacity);
            }
        }
        /* Logits are written for the last `rows` tokens, row-major. */
        uint32_t first_logit_token = item->token_count - rows;
        for (uint32_t token_index = 0; token_index < item->token_count;
             ++token_index) {
            float *token_logits =
                token_index >= first_logit_token
                    ? item->logits +
                          (size_t)(token_index - first_logit_token) *
                              KIPP_VOCAB_SIZE
                    : NULL;
            if (cpu_cached_token(state->view, session,
                                 item->tokens[token_index],
                                 item->start_position + token_index,
                                 token_logits, error) != 0) {
                return -1;
            }
        }
        if (!session->pooled) {
            session->length += item->token_count;
        }
    }
    return 0;
}

static const kipp_backend_ops *cpu_backend_operations(void) {
    static const kipp_backend_ops operations = {
        cpu_backend_model_create,
        cpu_backend_model_destroy,
        cpu_backend_session_create,
        cpu_backend_session_destroy,
        cpu_backend_session_reset,
        cpu_backend_eval,
        cpu_backend_session_truncate,
    };
    return &operations;
}

typedef struct {
    uint32_t token;
    float logit;
} sample_candidate;

static int compare_sample_candidates(const void *left, const void *right) {
    const sample_candidate *a = left;
    const sample_candidate *b = right;
    if (a->logit > b->logit) {
        return -1;
    }
    if (a->logit < b->logit) {
        return 1;
    }
    return a->token < b->token ? -1 : 1;
}

static uint64_t xorshift64_star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static double uniform_unit(uint64_t *state) {
    return (double)(xorshift64_star(state) >> 11) / 9007199254740992.0;
}

/*
 * Sampling scratch. kipp_sample_ex runs once per generated token per choice,
 * and a fresh working-copy + candidate malloc/free pair per call dominated
 * its cost at the full 151,936-entry vocabulary. The scratch is thread-local
 * so the library stays safe if a host embeds it in threads, lazily grown to
 * the largest vocabulary seen, and never shrunk or freed.
 */
static _Thread_local float *sample_working;
static _Thread_local sample_candidate *sample_candidates;
static _Thread_local size_t sample_scratch_count;

static int ensure_sample_scratch(size_t logits_count, kipp_error *error) {
    if (logits_count <= sample_scratch_count) {
        return 0;
    }
    float *working = realloc(sample_working,
                             logits_count * sizeof(*sample_working));
    if (working == NULL) {
        return fail(error, KIPP_ERROR_MEMORY,
                    "unable to allocate sampling logits");
    }
    sample_working = working;
    sample_candidate *candidates =
        realloc(sample_candidates, logits_count * sizeof(*sample_candidates));
    if (candidates == NULL) {
        return fail(error, KIPP_ERROR_MEMORY,
                    "unable to allocate sampling candidates");
    }
    sample_candidates = candidates;
    sample_scratch_count = logits_count;
    return 0;
}

/*
 * Bias and penalties mutate logits, which forces the working copy; plain
 * temperature sampling reads the caller's logits directly. Mirrors the
 * no-op conditions of apply_bias_and_penalties exactly.
 */
static bool sample_needs_working_copy(const kipp_sample_params *params) {
    bool has_penalty = params->frequency_penalty != 0.0f ||
                       params->presence_penalty != 0.0f ||
                       (params->repetition_penalty != 0.0f &&
                        params->repetition_penalty != 1.0f);
    return params->logit_bias_count > 0 ||
           (params->recent_count > 0 && has_penalty);
}

/*
 * Apply logit_bias and the recent-token penalties in place on a working
 * copy of the logits. Penalties look back over params->recent_tokens.
 */
static void apply_bias_and_penalties(float *logits, size_t logits_count,
                                     const kipp_sample_params *params) {
    for (size_t index = 0; index < params->logit_bias_count; ++index) {
        uint32_t token = params->logit_bias_tokens[index];
        if (token < logits_count) {
            logits[token] += params->logit_bias_values[index];
        }
    }
    bool has_penalty = params->frequency_penalty != 0.0f ||
                       params->presence_penalty != 0.0f ||
                       (params->repetition_penalty != 0.0f &&
                        params->repetition_penalty != 1.0f);
    if (!has_penalty || params->recent_count == 0) {
        return;
    }
    /* Count occurrences of each recent token once, then adjust its logit. */
    for (size_t position = 0; position < params->recent_count; ++position) {
        uint32_t token = params->recent_tokens[position];
        if (token >= logits_count) {
            continue;
        }
        /* Skip tokens already handled earlier in the window. */
        bool seen = false;
        for (size_t earlier = 0; earlier < position; ++earlier) {
            if (params->recent_tokens[earlier] == token) {
                seen = true;
                break;
            }
        }
        if (seen) {
            continue;
        }
        uint32_t count = 0;
        for (size_t other = 0; other < params->recent_count; ++other) {
            count += params->recent_tokens[other] == token;
        }
        if (params->repetition_penalty != 0.0f &&
            params->repetition_penalty != 1.0f) {
            logits[token] = logits[token] > 0.0f
                                ? logits[token] / params->repetition_penalty
                                : logits[token] * params->repetition_penalty;
        }
        logits[token] -= params->presence_penalty;
        logits[token] -= params->frequency_penalty * (float)count;
    }
}

int kipp_sample_ex(const float *logits, size_t logits_count,
                   const kipp_sample_params *params, uint64_t *rng_state,
                   uint32_t *out_token, kipp_error *error) {
    clear_error(error);
    if (logits == NULL || logits_count == 0 || params == NULL ||
        out_token == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "logits, params, and an output token are required");
    }

    /*
     * Bias and penalties can reorder logits, so they work on a copy; plain
     * temperature sampling reads the caller's logits in place. The values
     * are bitwise-identical either way (the copy is an exact memcpy), so
     * every downstream draw matches the always-copy behavior exactly.
     */
    const float *values = logits;
    if (sample_needs_working_copy(params)) {
        if (ensure_sample_scratch(logits_count, error) != 0) {
            return -1;
        }
        memcpy(sample_working, logits, logits_count * sizeof(*sample_working));
        apply_bias_and_penalties(sample_working, logits_count, params);
        values = sample_working;
    }

    size_t argmax = 0;
    for (size_t index = 1; index < logits_count; ++index) {
        if (values[index] > values[argmax]) {
            argmax = index;
        }
    }
    if (params->temperature <= 0.0f) {
        *out_token = (uint32_t)argmax;
        return 0;
    }
    if (!(params->top_p > 0.0f && params->top_p <= 1.0f)) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "top_p must be greater than 0 and at most 1");
    }
    if (params->min_p < 0.0f || params->min_p > 1.0f) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "min_p must be between 0 and 1");
    }
    if (rng_state == NULL || *rng_state == 0) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "sampling requires a nonzero RNG state");
    }
    if (ensure_sample_scratch(logits_count, error) != 0) {
        return -1;
    }

    /*
     * Scaled logits more than 20 below the maximum contribute at most
     * exp(-20) each, which is negligible even across the whole vocabulary;
     * skipping them keeps the sort small during ordinary decoding.
     */
    float inverse_temperature = 1.0f / params->temperature;
    float scaled_maximum = values[argmax] * inverse_temperature;
    float cutoff = scaled_maximum - 20.0f;
    sample_candidate *candidates = sample_candidates;
    size_t candidate_count = 0;
    for (size_t index = 0; index < logits_count; ++index) {
        float scaled = values[index] * inverse_temperature;
        if (scaled >= cutoff) {
            candidates[candidate_count].token = (uint32_t)index;
            candidates[candidate_count].logit = scaled;
            ++candidate_count;
        }
    }
    qsort(candidates, candidate_count, sizeof(*candidates),
          compare_sample_candidates);

    /* top_k caps the candidate set to the k highest. */
    if (params->top_k != 0 && params->top_k < candidate_count) {
        candidate_count = params->top_k;
    }

    double total = 0.0;
    for (size_t index = 0; index < candidate_count; ++index) {
        candidates[index].logit =
            expf(candidates[index].logit - scaled_maximum);
        total += candidates[index].logit;
    }

    /* min_p drops the low tail relative to the top probability. */
    if (params->min_p > 0.0f && candidate_count > 0) {
        double threshold = (double)params->min_p * candidates[0].logit;
        size_t kept_min_p = candidate_count;
        for (size_t index = 1; index < candidate_count; ++index) {
            if (candidates[index].logit < threshold) {
                kept_min_p = index;
                break;
            }
        }
        candidate_count = kept_min_p;
        total = 0.0;
        for (size_t index = 0; index < candidate_count; ++index) {
            total += candidates[index].logit;
        }
    }

    double target = (double)params->top_p * total;
    double kept = 0.0;
    size_t nucleus_count = 0;
    while (nucleus_count < candidate_count && kept < target) {
        kept += candidates[nucleus_count].logit;
        ++nucleus_count;
    }
    if (nucleus_count == 0) {
        nucleus_count = 1;
        kept = candidates[0].logit;
    }

    double draw = uniform_unit(rng_state) * kept;
    uint32_t token = candidates[nucleus_count - 1].token;
    double cumulative = 0.0;
    for (size_t index = 0; index < nucleus_count; ++index) {
        cumulative += candidates[index].logit;
        if (draw < cumulative) {
            token = candidates[index].token;
            break;
        }
    }
    *out_token = token;
    return 0;
}

int kipp_sample(const float *logits, size_t logits_count, float temperature,
                float top_p, uint64_t *rng_state, uint32_t *out_token,
                kipp_error *error) {
    kipp_sample_params params = {0};
    params.temperature = temperature;
    params.top_p = top_p;
    params.repetition_penalty = 1.0f;
    return kipp_sample_ex(logits, logits_count, &params, rng_state, out_token,
                          error);
}

int kipp_model_eval(const kipp_model *model, const uint32_t *tokens,
                    size_t token_count, float *logits, size_t logits_count,
                    kipp_error *error) {
    clear_error(error);
    if (model == NULL || tokens == NULL || logits == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "model, tokens, and logits are required");
    }
    if (token_count == 0 || token_count > KIPP_PHASE1_MAX_TOKENS) {
        return fail(error, KIPP_ERROR_RANGE,
                    "Phase 1 token count must be between 1 and %u",
                    KIPP_PHASE1_MAX_TOKENS);
    }
    if (logits_count < KIPP_VOCAB_SIZE) {
        return fail(error, KIPP_ERROR_RANGE,
                    "logit output must hold %u values", KIPP_VOCAB_SIZE);
    }
    for (size_t index = 0; index < token_count; ++index) {
        if (tokens[index] >= KIPP_VOCAB_SIZE) {
            return fail(error, KIPP_ERROR_RANGE, "token %zu is out of range",
                        index);
        }
    }
    kipp_eval_item item = {
        NULL, tokens, (uint32_t)token_count, 0, logits, 1, NULL, false,
    };
    return model->backend_ops->eval(model->backend_model, &item, 1, error);
}

static bool session_is_pooled(const kipp_session *session) {
    return session->model->kv_pool != NULL;
}

static uint32_t session_block_capacity(const kipp_session *session) {
    return (session->capacity + KIPP_KV_BLOCK_TOKENS - 1) /
           KIPP_KV_BLOCK_TOKENS;
}

/*
 * Map every logical block covering positions [0, new_length) that is not yet
 * mapped, claiming private pool blocks. On exhaustion the just-claimed blocks
 * are released and the session is left untouched. Returns the previous
 * mapped count through *first_new so a failed backend call can roll back.
 */
static int pooled_map_blocks(kipp_session *session, uint32_t new_length,
                             uint32_t *first_new, kipp_error *error) {
    kipp_kv_pool *pool = session->model->kv_pool;
    uint32_t needed =
        (new_length + KIPP_KV_BLOCK_TOKENS - 1) / KIPP_KV_BLOCK_TOKENS;
    *first_new = session->mapped_blocks;
    while (session->mapped_blocks < needed) {
        uint32_t id = kipp_kv_pool_alloc(pool);
        if (id == KIPP_KV_INVALID_BLOCK) {
            while (session->mapped_blocks > *first_new) {
                --session->mapped_blocks;
                kipp_kv_pool_release(
                    pool, session->block_ids[session->mapped_blocks]);
            }
            return fail(error, KIPP_ERROR_RANGE, "KV pool exhausted");
        }
        session->block_ids[session->mapped_blocks++] = id;
    }
    return 0;
}

static void pooled_unmap_from(kipp_session *session, uint32_t first_new) {
    kipp_kv_pool *pool = session->model->kv_pool;
    while (session->mapped_blocks > first_new) {
        --session->mapped_blocks;
        kipp_kv_pool_release(pool,
                             session->block_ids[session->mapped_blocks]);
    }
}

/*
 * Publish-at-finish: seal each fully evaluated, non-adopted block so later
 * sessions can adopt the prefix, then release every mapped block. Runs on
 * reset and destroy — never mid-generation, which is what keeps truncation
 * (speculative rollback) away from shared blocks.
 */
static void pooled_publish_and_release(kipp_session *session) {
    kipp_kv_pool *pool = session->model->kv_pool;
    uint32_t full_blocks = session->length / KIPP_KV_BLOCK_TOKENS;
    uint64_t parent = 0;
    for (uint32_t block = 0; block < full_blocks; ++block) {
        const uint32_t *tokens =
            session->timeline + (size_t)block * KIPP_KV_BLOCK_TOKENS;
        if (block >= session->adopted_blocks) {
            (void)kipp_kv_pool_seal(pool, session->block_ids[block], parent,
                                    tokens);
        }
        parent = kipp_kv_pool_hash(parent, tokens, KIPP_KV_BLOCK_TOKENS);
    }
    pooled_unmap_from(session, 0);
    session->adopted_blocks = 0;
}

int kipp_session_create(kipp_model *model, uint32_t capacity,
                        kipp_session **out_session, kipp_error *error) {
    kipp_session *session;
    clear_error(error);
    if (model == NULL || out_session == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "model and session output are required");
    }
    *out_session = NULL;
    if (capacity == 0 || capacity > model->view.config.context_length) {
        return fail(error, KIPP_ERROR_RANGE,
                    "session capacity must be between 1 and %u",
                    model->view.config.context_length);
    }
    if (model->active_session_count == SIZE_MAX) {
        return fail(error, KIPP_ERROR_RANGE, "active session count overflows");
    }
    session = calloc(1, sizeof(*session));
    if (session == NULL) {
        return fail(error, KIPP_ERROR_MEMORY, "unable to allocate session");
    }
    if (model->kv_pool != NULL) {
        uint32_t blocks = (capacity + KIPP_KV_BLOCK_TOKENS - 1) /
                          KIPP_KV_BLOCK_TOKENS;
        session->timeline =
            malloc((size_t)capacity * sizeof(*session->timeline));
        session->block_ids =
            malloc((size_t)blocks * sizeof(*session->block_ids));
        if (session->timeline == NULL || session->block_ids == NULL) {
            free(session->timeline);
            free(session->block_ids);
            free(session);
            return fail(error, KIPP_ERROR_MEMORY,
                        "unable to allocate pooled session bookkeeping");
        }
    }
    if (model->backend_ops->session_create(model->backend_model, capacity,
                                           &session->backend_session,
                                           error) != 0) {
        free(session->timeline);
        free(session->block_ids);
        free(session);
        return -1;
    }
    session->model = model;
    session->capacity = capacity;
    session->cache_bytes = kv_cache_bytes_for_capacity(
        model->view.config.block_count, capacity);
    ++model->active_session_count;
    *out_session = session;
    return 0;
}

int kipp_session_reset(kipp_session *session, kipp_error *error) {
    clear_error(error);
    if (session == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT, "session is required");
    }
    if (session_is_pooled(session)) {
        pooled_publish_and_release(session);
    }
    if (session->model->backend_ops->session_reset(session->backend_session,
                                                   error) != 0) {
        return -1;
    }
    session->length = 0;
    return 0;
}

int kipp_session_truncate(kipp_session *session, uint32_t length,
                          kipp_error *error) {
    clear_error(error);
    if (session == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT, "session is required");
    }
    if (length > session->length) {
        return fail(error, KIPP_ERROR_RANGE,
                    "cannot truncate session of length %u to %u",
                    session->length, length);
    }
    const kipp_backend_ops *ops = session->model->backend_ops;
    if (ops->session_truncate == NULL) {
        return fail(error, KIPP_ERROR_UNSUPPORTED,
                    "this backend does not support session truncation");
    }
    if (session_is_pooled(session)) {
        /* Publish-at-finish guarantees every block past the adopted prefix
         * is private, so truncation only releases private blocks. */
        if (length < session->adopted_blocks * KIPP_KV_BLOCK_TOKENS) {
            return fail(error, KIPP_ERROR_RANGE,
                        "cannot truncate into an adopted shared prefix");
        }
        uint32_t keep =
            (length + KIPP_KV_BLOCK_TOKENS - 1) / KIPP_KV_BLOCK_TOKENS;
        if (keep < session->adopted_blocks) {
            keep = session->adopted_blocks;
        }
        pooled_unmap_from(session, keep);
    }
    if (ops->session_truncate(session->backend_session, length, error) != 0) {
        return -1;
    }
    session->length = length;
    return 0;
}

void kipp_session_destroy(kipp_session *session) {
    if (session == NULL) {
        return;
    }
    kipp_model *model = session->model;
    if (session_is_pooled(session)) {
        pooled_publish_and_release(session);
    }
    model->backend_ops->session_destroy(session->backend_session);
    if (model->active_session_count > 0) {
        --model->active_session_count;
    }
    free(session->timeline);
    free(session->block_ids);
    free(session);
}

int kipp_session_match_prefix(kipp_session *session, const uint32_t *tokens,
                              size_t token_count, uint32_t *matched_tokens,
                              kipp_error *error) {
    clear_error(error);
    if (matched_tokens != NULL) {
        *matched_tokens = 0;
    }
    if (session == NULL || tokens == NULL || token_count == 0) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "session and tokens are required");
    }
    if (!session_is_pooled(session)) {
        return fail(error, KIPP_ERROR_UNSUPPORTED,
                    "prefix matching requires a pooled model");
    }
    if (session->length != 0 || session->mapped_blocks != 0) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "prefix matching requires a fresh session");
    }
    if (token_count > session->capacity) {
        token_count = session->capacity;
    }
    uint32_t matched = 0;
    uint32_t blocks = kipp_kv_pool_prefix_match(
        session->model->kv_pool, tokens, (uint32_t)token_count,
        session->block_ids, session_block_capacity(session), &matched, NULL);
    if (blocks != 0) {
        memcpy(session->timeline, tokens,
               (size_t)matched * sizeof(*session->timeline));
        session->mapped_blocks = blocks;
        session->adopted_blocks = blocks;
        session->length = matched;
    }
    if (matched_tokens != NULL) {
        *matched_tokens = matched;
    }
    return 0;
}

int kipp_model_kv_pool_stats(const kipp_model *model,
                             kipp_kv_pool_stats_public *out_stats) {
    if (model == NULL || out_stats == NULL || model->kv_pool == NULL) {
        return -1;
    }
    kipp_kv_pool_stats stats;
    kipp_kv_pool_get_stats(model->kv_pool, &stats);
    out_stats->total_blocks = stats.total_blocks;
    out_stats->free_blocks = stats.free_blocks;
    out_stats->reused_blocks_total = stats.reused_blocks_total;
    out_stats->evicted_blocks_total = stats.evicted_blocks_total;
    return 0;
}

int kipp_session_get_info(const kipp_session *session,
                          kipp_session_info *out_info) {
    if (session == NULL || out_info == NULL) {
        return -1;
    }
    *out_info = (kipp_session_info){
        session->capacity,
        session->length,
        session->cache_bytes,
    };
    return 0;
}

int kipp_session_eval(kipp_session *session, const uint32_t *tokens,
                      size_t token_count, float *logits, size_t logits_count,
                      kipp_error *error) {
    clear_error(error);
    if (session == NULL || tokens == NULL || logits == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "session, tokens, and logits are required");
    }
    if (token_count == 0 || token_count > UINT32_MAX) {
        return fail(error, KIPP_ERROR_RANGE,
                    "session evaluation requires at least one token");
    }
    if (token_count > session->capacity - session->length) {
        return fail(error, KIPP_ERROR_RANGE,
                    "append of %zu token(s) exceeds session capacity %u",
                    token_count, session->capacity);
    }
    if (logits_count < KIPP_VOCAB_SIZE) {
        return fail(error, KIPP_ERROR_RANGE,
                    "logit output must hold %u values", KIPP_VOCAB_SIZE);
    }
    for (size_t index = 0; index < token_count; ++index) {
        if (tokens[index] >= KIPP_VOCAB_SIZE) {
            return fail(error, KIPP_ERROR_RANGE, "token %zu is out of range",
                        index);
        }
    }
    kipp_eval_item item = {
        session->backend_session, tokens, (uint32_t)token_count,
        session->length,         logits, 1,      NULL, false,
    };
    uint32_t first_new = 0;
    if (session_is_pooled(session)) {
        if (pooled_map_blocks(session,
                              session->length + (uint32_t)token_count,
                              &first_new, error) != 0) {
            return -1;
        }
        item.block_table = session->block_ids;
    }
    if (session->model->backend_ops->eval(session->model->backend_model,
                                          &item, 1, error) != 0) {
        if (session_is_pooled(session)) {
            pooled_unmap_from(session, first_new);
        }
        return -1;
    }
    if (session_is_pooled(session)) {
        memcpy(session->timeline + session->length, tokens,
               token_count * sizeof(*tokens));
    }
    session->length += (uint32_t)token_count;
    return 0;
}

static int session_eval_rows(kipp_session *session, const uint32_t *tokens,
                             size_t token_count, float *logits, uint32_t rows,
                             bool relaxed_order, kipp_error *error) {
    clear_error(error);
    if (session == NULL || tokens == NULL || logits == NULL) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "session, tokens, and logits are required");
    }
    if (token_count == 0 || token_count > UINT32_MAX) {
        return fail(error, KIPP_ERROR_RANGE,
                    "session evaluation requires at least one token");
    }
    if (rows == 0 || rows > token_count) {
        return fail(error, KIPP_ERROR_RANGE,
                    "rows must be between 1 and the token count");
    }
    if (token_count > session->capacity - session->length) {
        return fail(error, KIPP_ERROR_RANGE,
                    "append of %zu token(s) exceeds session capacity %u",
                    token_count, session->capacity);
    }
    for (size_t index = 0; index < token_count; ++index) {
        if (tokens[index] >= KIPP_VOCAB_SIZE) {
            return fail(error, KIPP_ERROR_RANGE, "token %zu is out of range",
                        index);
        }
    }
    kipp_eval_item item = {
        session->backend_session, tokens, (uint32_t)token_count,
        session->length,          logits, rows,   NULL,
        relaxed_order,
    };
    uint32_t first_new = 0;
    if (session_is_pooled(session)) {
        if (pooled_map_blocks(session,
                              session->length + (uint32_t)token_count,
                              &first_new, error) != 0) {
            return -1;
        }
        item.block_table = session->block_ids;
    }
    if (session->model->backend_ops->eval(session->model->backend_model,
                                          &item, 1, error) != 0) {
        if (session_is_pooled(session)) {
            pooled_unmap_from(session, first_new);
        }
        return -1;
    }
    if (session_is_pooled(session)) {
        memcpy(session->timeline + session->length, tokens,
               token_count * sizeof(*tokens));
    }
    session->length += (uint32_t)token_count;
    return 0;
}

int kipp_session_eval_n(kipp_session *session, const uint32_t *tokens,
                        size_t token_count, float *logits, uint32_t rows,
                        kipp_error *error) {
    return session_eval_rows(session, tokens, token_count, logits, rows,
                             false, error);
}

int kipp_session_eval_scored(kipp_session *session, const uint32_t *tokens,
                             size_t token_count, float *logits, uint32_t rows,
                             kipp_error *error) {
    return session_eval_rows(session, tokens, token_count, logits, rows,
                             true, error);
}

int kipp_eval_batch(kipp_model *model, kipp_batch_item *items,
                    size_t item_count, kipp_error *error) {
    clear_error(error);
    if (model == NULL || items == NULL || item_count == 0) {
        return fail(error, KIPP_ERROR_ARGUMENT,
                    "model and batch items are required");
    }
    if (item_count > KIPP_EVAL_BATCH_LIMIT) {
        return fail(error, KIPP_ERROR_RANGE,
                    "batch accepts at most %u items", KIPP_EVAL_BATCH_LIMIT);
    }
    kipp_eval_item backend_items[KIPP_EVAL_BATCH_LIMIT];
    for (size_t index = 0; index < item_count; ++index) {
        kipp_batch_item *item = &items[index];
        kipp_session *session = item->session;
        if (session == NULL || item->tokens == NULL || item->logits == NULL) {
            return fail(error, KIPP_ERROR_ARGUMENT,
                        "batch item %zu is incomplete", index);
        }
        if (session->model != model) {
            return fail(error, KIPP_ERROR_ARGUMENT,
                        "batch item %zu belongs to another model", index);
        }
        for (size_t previous = 0; previous < index; ++previous) {
            if (items[previous].session == session) {
                return fail(error, KIPP_ERROR_ARGUMENT,
                            "batch items %zu and %zu share a session",
                            previous, index);
            }
        }
        if (item->token_count == 0 || item->token_count > UINT32_MAX) {
            return fail(error, KIPP_ERROR_RANGE,
                        "batch item %zu requires at least one token", index);
        }
        if (item->token_count > session->capacity - session->length) {
            return fail(error, KIPP_ERROR_RANGE,
                        "batch item %zu exceeds session capacity %u", index,
                        session->capacity);
        }
        for (size_t token = 0; token < item->token_count; ++token) {
            if (item->tokens[token] >= KIPP_VOCAB_SIZE) {
                return fail(error, KIPP_ERROR_RANGE,
                            "batch item %zu token %zu is out of range",
                            index, token);
            }
        }
        backend_items[index] = (kipp_eval_item){
            session->backend_session,
            item->tokens,
            (uint32_t)item->token_count,
            session->length,
            item->logits,
            1,
            NULL,
            false,
        };
    }
    /* Map pooled blocks for every item before the single backend call; on
     * any failure, unmap exactly what this call mapped. */
    uint32_t first_new[KIPP_EVAL_BATCH_LIMIT];
    for (size_t index = 0; index < item_count; ++index) {
        kipp_session *session = items[index].session;
        first_new[index] = session->mapped_blocks;
        if (!session_is_pooled(session)) {
            continue;
        }
        if (pooled_map_blocks(session,
                              session->length +
                                  (uint32_t)items[index].token_count,
                              &first_new[index], error) != 0) {
            for (size_t previous = 0; previous < index; ++previous) {
                if (session_is_pooled(items[previous].session)) {
                    pooled_unmap_from(items[previous].session,
                                      first_new[previous]);
                }
            }
            return -1;
        }
        backend_items[index].block_table = session->block_ids;
    }
    if (model->backend_ops->eval(model->backend_model, backend_items,
                                 item_count, error) != 0) {
        for (size_t index = 0; index < item_count; ++index) {
            if (session_is_pooled(items[index].session)) {
                pooled_unmap_from(items[index].session, first_new[index]);
            }
        }
        return -1;
    }
    for (size_t index = 0; index < item_count; ++index) {
        kipp_session *session = items[index].session;
        if (session_is_pooled(session)) {
            memcpy(session->timeline + session->length, items[index].tokens,
                   items[index].token_count * sizeof(uint32_t));
        }
        session->length += (uint32_t)items[index].token_count;
    }
    return 0;
}

#ifdef KIPP_TESTING
float kipp_test_bf16_to_float(uint16_t value) {
    return bf16_to_float(value);
}

uint16_t kipp_test_float_to_bf16(float value) {
    return float_to_bf16(value);
}

float kipp_test_fp16_to_float(uint16_t value) {
    return fp16_to_float(value);
}

void kipp_test_matvec_q8_0(const uint8_t *weight, const float *input,
                           float *output, size_t rows, size_t columns) {
    matvec_q8_0(weight, input, output, rows, columns);
}

void kipp_test_matvec_affine4_gs32(const uint8_t *weight, const float *input,
                                   float *output, size_t rows, size_t columns) {
    matvec_affine4_gs32(weight, input, output, rows, columns);
}

int kipp_test_checked_add_size(size_t left, size_t right, size_t *result) {
    return checked_add_size(left, right, result) ? 0 : -1;
}

int kipp_test_checked_multiply_size(size_t left, size_t right, size_t *result) {
    return checked_multiply_size(left, right, result) ? 0 : -1;
}

uint64_t kipp_test_kv_cache_bytes(uint32_t block_count, uint32_t capacity) {
    return kv_cache_bytes_for_capacity(block_count, capacity);
}

size_t kipp_test_kv_cache_offset(uint32_t capacity, uint32_t layer,
                                 uint32_t position, uint32_t head,
                                 uint32_t dimension) {
    return kv_cache_offset_for_capacity(capacity, layer, position, head,
                                        dimension);
}

int kipp_test_scramble_session_kv(kipp_session *session) {
    if (session == NULL || session->backend_session == NULL ||
        session->model == NULL) {
        return -1;
    }
    /* Reverse the block table so every logical block resolves to a different
     * physical block; a correct paged read/write path yields identical
     * results regardless of physical placement. Must precede any eval. */
    if (session->model->kv_pool != NULL) {
        return -1; /* pooled sessions have no private identity table */
    }
    if (session->model->backend_ops == cpu_backend_operations()) {
        cpu_backend_session *cpu = session->backend_session;
        uint32_t low = 0;
        uint32_t high = cpu->block_capacity;
        while (low + 1 < high) {
            --high;
            uint32_t swap = cpu->block_table[low];
            cpu->block_table[low] = cpu->block_table[high];
            cpu->block_table[high] = swap;
            ++low;
        }
        return 0;
    }
#ifdef KIPP_ENABLE_METAL
    if (session->model->backend_ops == kipp_metal_backend_operations()) {
        return kipp_metal_test_scramble_session(session->backend_session);
    }
#endif
    return -1;
}

void kipp_test_rms_norm(const float *input, const uint16_t *weight,
                        float *output, size_t length, float epsilon) {
    rms_norm(input, weight, output, length, epsilon);
}

void kipp_test_matvec_bf16(const uint16_t *weight, const float *input,
                           float *output, size_t rows, size_t columns) {
    matvec_bf16(weight, input, output, rows, columns);
}

void kipp_test_rope(float *head, size_t head_dim, uint32_t position,
                    float theta) {
    rope_head(head, head_dim, position, theta);
}

void kipp_test_softmax(float *values, size_t count) {
    softmax(values, count);
}

float kipp_test_silu(float value) {
    return silu(value);
}

void kipp_test_causal_gqa(const float *query, const float *key,
                          const float *value, float *output, float *scores,
                          size_t token_count, size_t query_head_count) {
    causal_gqa(query, key, value, output, scores, token_count,
               query_head_count);
}

int kipp_test_pretokenize(const char *text, size_t **offsets,
                          size_t **lengths, size_t *count) {
    byte_span *spans = NULL;
    size_t span_count = 0;
    if (text == NULL || offsets == NULL || lengths == NULL || count == NULL ||
        pretokenize_bytes((const uint8_t *)text, strlen(text), &spans,
                          &span_count) != 0) {
        return -1;
    }
    *offsets = malloc((span_count == 0 ? 1 : span_count) * sizeof(**offsets));
    *lengths = malloc((span_count == 0 ? 1 : span_count) * sizeof(**lengths));
    if (*offsets == NULL || *lengths == NULL) {
        free(*offsets);
        free(*lengths);
        free(spans);
        return -1;
    }
    for (size_t index = 0; index < span_count; ++index) {
        (*offsets)[index] = spans[index].offset;
        (*lengths)[index] = spans[index].length;
    }
    *count = span_count;
    free(spans);
    return 0;
}

int kipp_test_normalize_nfc(const char *text, char **normalized) {
    uint8_t *result = NULL;
    size_t length = 0;
    if (text == NULL || normalized == NULL ||
        normalize_nfc((const uint8_t *)text, strlen(text), &result, &length) != 0) {
        return -1;
    }
    (void)length;
    *normalized = (char *)result;
    return 0;
}
#endif
