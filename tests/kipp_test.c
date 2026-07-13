#define _POSIX_C_SOURCE 200809L

#include "kipp.h"
#ifdef KIPP_ENABLE_METAL
#include "metal/kipp_metal.h"
#endif
#ifdef KIPP_ENABLE_CUDA
#include "cuda/kipp_cuda.h"
#endif

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;
static int tests_run;

#define CHECK(condition)                                                         \
    do {                                                                         \
        if (!(condition)) {                                                       \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,    \
                    #condition);                                                  \
            ++failures;                                                          \
        }                                                                         \
    } while (0)

#define RUN(test)                                                                \
    do {                                                                         \
        int before = failures;                                                    \
        ++tests_run;                                                              \
        test();                                                                   \
        fprintf(stderr, "%s %s\n", failures == before ? "PASS" : "FAIL", #test);\
    } while (0)

static int nearly_equal(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

static void test_error_names(void) {
    CHECK(strcmp(kipp_error_code_name(KIPP_OK), "ok") == 0);
    CHECK(strcmp(kipp_error_code_name(KIPP_ERROR_FORMAT), "format") == 0);
    CHECK(strcmp(kipp_error_code_name((kipp_error_code)999), "unknown") == 0);
}

static void test_bf16(void) {
    CHECK(kipp_test_float_to_bf16(1.0f) == UINT16_C(0x3f80));
    CHECK(kipp_test_float_to_bf16(-2.0f) == UINT16_C(0xc000));
    CHECK(kipp_test_bf16_to_float(UINT16_C(0x3f80)) == 1.0f);
    CHECK(kipp_test_bf16_to_float(UINT16_C(0xc000)) == -2.0f);
    CHECK(isinf(kipp_test_bf16_to_float(UINT16_C(0x7f80))));
}

static void test_checked_arithmetic(void) {
    size_t result = 0;
    CHECK(kipp_test_checked_add_size(10, 20, &result) == 0);
    CHECK(result == 30);
    CHECK(kipp_test_checked_add_size(SIZE_MAX, 1, &result) != 0);
    CHECK(kipp_test_checked_multiply_size(10, 20, &result) == 0);
    CHECK(result == 200);
    CHECK(kipp_test_checked_multiply_size(SIZE_MAX, 2, &result) != 0);
}

static void test_kv_layout(void) {
    CHECK(kipp_test_kv_cache_bytes(1) == UINT64_C(147456));
    CHECK(kipp_test_kv_cache_bytes(8192) == UINT64_C(1207959552));
    CHECK(kipp_test_kv_cache_offset(4, 0, 0, 0, 0) == 0);
    CHECK(kipp_test_kv_cache_offset(4, 0, 0, 1, 0) == 128);
    CHECK(kipp_test_kv_cache_offset(4, 0, 1, 0, 0) == 1024);
    CHECK(kipp_test_kv_cache_offset(4, 1, 0, 0, 0) == 4096);
    CHECK(kipp_test_kv_cache_offset(4, 35, 3, 7, 127) ==
          (size_t)36 * 4 * 8 * 128 - 1);
}

static void test_rms_norm(void) {
    const float input[] = {3.0f, 4.0f};
    const uint16_t weight[] = {UINT16_C(0x3f80), UINT16_C(0x3f80)};
    float output[2];
    float scale = 1.0f / sqrtf(12.5f);
    kipp_test_rms_norm(input, weight, output, 2, 0.0f);
    CHECK(nearly_equal(output[0], 3.0f * scale, 1.0e-6f));
    CHECK(nearly_equal(output[1], 4.0f * scale, 1.0e-6f));
}

static void test_matvec(void) {
    const uint16_t weight[] = {
        UINT16_C(0x3f80), UINT16_C(0x4000), UINT16_C(0x4040),
        UINT16_C(0xbf80), UINT16_C(0x3f00), UINT16_C(0x0000),
    };
    const float input[] = {2.0f, -1.0f, 0.5f};
    float output[2];
    kipp_test_matvec_bf16(weight, input, output, 2, 3);
    CHECK(nearly_equal(output[0], 1.5f, 1.0e-6f));
    CHECK(nearly_equal(output[1], -2.5f, 1.0e-6f));
}

static void test_rope(void) {
    float head[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float cosine = cosf(1.0f);
    const float sine = sinf(1.0f);
    const float slow_cosine = cosf(0.01f);
    const float slow_sine = sinf(0.01f);
    kipp_test_rope(head, 4, 1, 10000.0f);
    CHECK(nearly_equal(head[0], cosine - 3.0f * sine, 1.0e-6f));
    CHECK(nearly_equal(head[2], 3.0f * cosine + sine, 1.0e-6f));
    CHECK(nearly_equal(head[1], 2.0f * slow_cosine - 4.0f * slow_sine,
                       1.0e-6f));
    CHECK(nearly_equal(head[3], 4.0f * slow_cosine + 2.0f * slow_sine,
                       1.0e-6f));
}

static void test_softmax_and_swiglu(void) {
    float values[] = {1.0f, 2.0f, 3.0f};
    kipp_test_softmax(values, 3);
    CHECK(nearly_equal(values[0] + values[1] + values[2], 1.0f, 1.0e-6f));
    CHECK(values[0] < values[1] && values[1] < values[2]);
    CHECK(kipp_test_silu(0.0f) == 0.0f);
    CHECK(nearly_equal(kipp_test_silu(1.0f), 0.7310586f, 1.0e-6f));
    CHECK(nearly_equal(kipp_test_silu(1.0f) * 2.0f, 1.4621172f, 1.0e-6f));
}

static void test_causal_gqa(void) {
    const size_t token_count = 2;
    const size_t query_values =
        token_count * KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM;
    const size_t kv_values =
        token_count * KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    float *query = calloc(query_values, sizeof(*query));
    float *key = calloc(kv_values, sizeof(*key));
    float *value = calloc(kv_values, sizeof(*value));
    float *output = calloc(query_values, sizeof(*output));
    float scores[2] = {0.0f, 0.0f};
    CHECK(query != NULL && key != NULL && value != NULL && output != NULL);
    if (query == NULL || key == NULL || value == NULL || output == NULL) {
        free(query);
        free(key);
        free(value);
        free(output);
        return;
    }
    for (size_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT_KV; ++head) {
        value[head * KIPP_ATTENTION_HEAD_DIM] = 1.0f;
        value[(KIPP_ATTENTION_HEAD_COUNT_KV + head) *
              KIPP_ATTENTION_HEAD_DIM] = 3.0f;
    }
    kipp_test_causal_gqa(query, key, value, output, scores, token_count);
    for (size_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT; ++head) {
        CHECK(nearly_equal(
            output[head * KIPP_ATTENTION_HEAD_DIM], 1.0f, 1.0e-6f));
        CHECK(nearly_equal(
            output[(KIPP_ATTENTION_HEAD_COUNT + head) *
                   KIPP_ATTENTION_HEAD_DIM],
            2.0f, 1.0e-6f));
    }
    CHECK(nearly_equal(scores[0], 0.5f, 1.0e-6f));
    CHECK(nearly_equal(scores[1], 0.5f, 1.0e-6f));
    free(query);
    free(key);
    free(value);
    free(output);
}

static void test_pretokenizer(void) {
    const char *text = "Hello, world! 1234 café 世界";
    size_t *offsets = NULL;
    size_t *lengths = NULL;
    size_t count = 0;
    CHECK(kipp_test_pretokenize(text, &offsets, &lengths, &count) == 0);
    CHECK(count >= 7);
    size_t reconstructed = 0;
    for (size_t index = 0; index < count; ++index) {
        CHECK(offsets[index] == reconstructed);
        reconstructed += lengths[index];
    }
    CHECK(reconstructed == strlen(text));
    CHECK(lengths[0] == 5 && memcmp(text + offsets[0], "Hello", 5) == 0);
    CHECK(lengths[1] == 1 && text[offsets[1]] == ',');
    CHECK(lengths[2] == 6 && memcmp(text + offsets[2], " world", 6) == 0);
    free(offsets);
    free(lengths);

    const char invalid[] = {(char)0xc0, (char)0x80, '\0'};
    CHECK(kipp_test_pretokenize(invalid, &offsets, &lengths, &count) != 0);
}

static void test_nfc_normalization(void) {
    char *normalized = NULL;
    CHECK(kipp_test_normalize_nfc("cafe\xcc\x81", &normalized) == 0);
    CHECK(normalized != NULL && strcmp(normalized, "caf\xc3\xa9") == 0);
    free(normalized);
    normalized = NULL;

    CHECK(kipp_test_normalize_nfc("\xe1\x84\x80\xe1\x85\xa1", &normalized) == 0);
    CHECK(normalized != NULL && strcmp(normalized, "\xea\xb0\x80") == 0);
    free(normalized);
}

static uint64_t fuzz_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

/* Encode one random scalar codepoint, biased toward composition-heavy
 * ranges so NFC recomposition and Hangul algorithms get exercised. */
static size_t fuzz_codepoint_utf8(uint64_t *state, char *output) {
    uint32_t codepoint;
    switch (fuzz_next(state) % 6) {
    case 0:
        codepoint = 0x20 + (uint32_t)(fuzz_next(state) % 0x5f);
        break;
    case 1: /* combining marks */
        codepoint = 0x300 + (uint32_t)(fuzz_next(state) % 0x70);
        break;
    case 2: /* precomposed Latin */
        codepoint = 0xc0 + (uint32_t)(fuzz_next(state) % 0x140);
        break;
    case 3: /* Hangul jamo */
        codepoint = 0x1100 + (uint32_t)(fuzz_next(state) % 0xc0);
        break;
    case 4: /* Hangul syllables */
        codepoint = 0xac00 + (uint32_t)(fuzz_next(state) % 0x2ba4);
        break;
    default: /* CJK and beyond */
        codepoint = 0x4e00 + (uint32_t)(fuzz_next(state) % 0x5000);
        break;
    }
    if (codepoint < 0x80) {
        output[0] = (char)codepoint;
        return 1;
    }
    if (codepoint < 0x800) {
        output[0] = (char)(0xc0 | (codepoint >> 6));
        output[1] = (char)(0x80 | (codepoint & 0x3f));
        return 2;
    }
    output[0] = (char)(0xe0 | (codepoint >> 12));
    output[1] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
    output[2] = (char)(0x80 | (codepoint & 0x3f));
    return 3;
}

static void test_pretokenizer_fuzz(void) {
    uint64_t state = UINT64_C(0x853c49e6748fea9b);
    char buffer[256];
    for (int round = 0; round < 400; ++round) {
        size_t length;
        if (round % 2 == 0) {
            /* Arbitrary bytes: must either fail cleanly or span the text. */
            length = fuzz_next(&state) % 48;
            for (size_t index = 0; index < length; ++index) {
                buffer[index] = (char)(fuzz_next(&state) & 0xff);
            }
        } else {
            /* Valid UTF-8: pretokenization must succeed and cover it. */
            size_t codepoints = fuzz_next(&state) % 24;
            length = 0;
            while (codepoints-- > 0 && length + 4 < sizeof(buffer)) {
                length += fuzz_codepoint_utf8(&state, buffer + length);
            }
        }
        buffer[length] = '\0';
        size_t *offsets = NULL;
        size_t *lengths = NULL;
        size_t count = 0;
        int status = kipp_test_pretokenize(buffer, &offsets, &lengths,
                                           &count);
        if (round % 2 != 0) {
            CHECK(status == 0);
        }
        if (status == 0) {
            size_t position = 0;
            for (size_t index = 0; index < count; ++index) {
                CHECK(offsets[index] == position);
                position += lengths[index];
            }
            CHECK(position == strlen(buffer));
            free(offsets);
            free(lengths);
        }
    }
}

static void test_nfc_fuzz(void) {
    uint64_t state = UINT64_C(0xda3e39cb94b95bdb);
    char buffer[256];
    for (int round = 0; round < 400; ++round) {
        size_t codepoints = fuzz_next(&state) % 24;
        size_t length = 0;
        while (codepoints-- > 0 && length + 4 < sizeof(buffer)) {
            length += fuzz_codepoint_utf8(&state, buffer + length);
        }
        buffer[length] = '\0';
        char *once = NULL;
        char *twice = NULL;
        CHECK(kipp_test_normalize_nfc(buffer, &once) == 0);
        CHECK(kipp_test_normalize_nfc(once, &twice) == 0);
        CHECK(strcmp(once, twice) == 0);
        free(once);
        free(twice);
    }
}

static void test_gguf_reject_fuzz(void) {
    uint64_t state = UINT64_C(0x2545f4914f6cdd1d);
    for (int round = 0; round < 60; ++round) {
        char path[] = "/tmp/kipp-fuzz-XXXXXX";
        int descriptor = mkstemp(path);
        CHECK(descriptor >= 0);
        if (descriptor < 0) {
            return;
        }
        uint8_t contents[512];
        size_t size = 16 + fuzz_next(&state) % (sizeof(contents) - 16);
        for (size_t index = 0; index < size; ++index) {
            contents[index] = (uint8_t)(fuzz_next(&state) & 0xff);
        }
        /* Half the rounds present a plausible header so parsing reaches
         * the metadata and tensor-directory validation paths. */
        if (round % 2 == 0) {
            memcpy(contents, "GGUF", 4);
            uint32_t version = 3;
            memcpy(contents + 4, &version, sizeof(version));
            uint64_t tensor_count = 398;
            memcpy(contents + 8, &tensor_count, sizeof(tensor_count));
        }
        CHECK(write(descriptor, contents, size) == (ssize_t)size);
        CHECK(close(descriptor) == 0);
        kipp_model *model = NULL;
        kipp_error error = {0};
        CHECK(kipp_model_open(path, &model, &error) != 0);
        CHECK(model == NULL);
        CHECK(error.code != KIPP_OK);
        CHECK(unlink(path) == 0);
    }
}

static void test_sampler(void) {
    kipp_error error = {0};
    float logits[8] = {0.1f, 4.0f, -1.0f, 3.5f, 0.0f, -7.0f, 2.0f, 1.0f};
    uint32_t token = UINT32_MAX;
    uint64_t rng = 42;

    CHECK(kipp_sample(logits, 8, 0.0f, 1.0f, NULL, &token, &error) == 0);
    CHECK(token == 1);
    CHECK(kipp_sample(logits, 8, 1.0f, 1.0e-6f, &rng, &token, &error) == 0);
    CHECK(token == 1);

    rng = 42;
    uint32_t first = UINT32_MAX;
    CHECK(kipp_sample(logits, 8, 1.0f, 0.9f, &rng, &first, &error) == 0);
    rng = 42;
    CHECK(kipp_sample(logits, 8, 1.0f, 0.9f, &rng, &token, &error) == 0);
    CHECK(token == first);
    CHECK(token < 8);

    int seen[8] = {0};
    rng = 7;
    for (int draw = 0; draw < 200; ++draw) {
        CHECK(kipp_sample(logits, 8, 1.5f, 1.0f, &rng, &token, &error) == 0);
        CHECK(token < 8);
        seen[token] = 1;
    }
    CHECK(seen[1] && seen[3]);
    CHECK(!seen[5]);

    CHECK(kipp_sample(NULL, 8, 0.0f, 1.0f, NULL, &token, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
    CHECK(kipp_sample(logits, 8, 1.0f, 0.0f, &rng, &token, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
    CHECK(kipp_sample(logits, 8, 1.0f, 1.5f, &rng, &token, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
    uint64_t zero_rng = 0;
    CHECK(kipp_sample(logits, 8, 1.0f, 1.0f, &zero_rng, &token, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
}

static void test_public_argument_checks(void) {
    kipp_error error = {0};
    kipp_model *model = (kipp_model *)(uintptr_t)1;
    kipp_session *session = (kipp_session *)(uintptr_t)1;
    float logits[1] = {0.0f};
    uint32_t token = 0;
    CHECK(kipp_model_open(NULL, &model, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
    CHECK(model == (kipp_model *)(uintptr_t)1);
    CHECK(kipp_model_get_info(NULL, NULL) != 0);
    CHECK(kipp_session_create(NULL, 1, &session, &error) != 0);
    CHECK(session == (kipp_session *)(uintptr_t)1);
    CHECK(kipp_session_reset(NULL, &error) != 0);
    CHECK(kipp_session_get_info(NULL, NULL) != 0);
    CHECK(kipp_session_eval(NULL, &token, 1, logits, 1, &error) != 0);
    CHECK(kipp_session_truncate(NULL, 0, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
    kipp_batch_item batch_item = {NULL, &token, 1, logits};
    CHECK(kipp_eval_batch(NULL, &batch_item, 1, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
#ifndef KIPP_ENABLE_METAL
    kipp_model *metal_model = NULL;
    CHECK(kipp_model_open_backend("unused.gguf", KIPP_BACKEND_METAL,
                                  &metal_model, &error) != 0);
    CHECK(error.code == KIPP_ERROR_UNSUPPORTED);
    CHECK(metal_model == NULL);
#endif
#ifndef KIPP_ENABLE_CUDA
    kipp_model *cuda_model = NULL;
    CHECK(kipp_model_open_backend("unused.gguf", KIPP_BACKEND_CUDA,
                                  &cuda_model, &error) != 0);
    CHECK(error.code == KIPP_ERROR_UNSUPPORTED);
    CHECK(cuda_model == NULL);
#endif
    kipp_session_destroy(NULL);
    kipp_tokens_free(NULL);
    CHECK(kipp_model_close(NULL, &error) == 0);
}

static void test_malformed_gguf(void) {
    char path[] = "/tmp/kipp-malformed-XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    if (descriptor < 0) {
        return;
    }
    const char content[] = "not a gguf";
    CHECK(write(descriptor, content, sizeof(content)) == (ssize_t)sizeof(content));
    CHECK(close(descriptor) == 0);

    kipp_model *model = NULL;
    kipp_error error = {0};
    CHECK(kipp_model_open(path, &model, &error) != 0);
    CHECK(error.code == KIPP_ERROR_FORMAT);
    CHECK(model == NULL);
    CHECK(unlink(path) == 0);
}

static int read_file(const char *path, void **data, size_t *size) {
    FILE *file = fopen(path, "rb");
    long length;
    void *buffer;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) {
            fclose(file);
        }
        return -1;
    }
    buffer = malloc(length == 0 ? 1 : (size_t)length);
    if (buffer == NULL ||
        fread(buffer, 1, (size_t)length, file) != (size_t)length ||
        fclose(file) != 0) {
        free(buffer);
        return -1;
    }
    *data = buffer;
    *size = (size_t)length;
    return 0;
}

static char *join_path(const char *directory, const char *name) {
    size_t length = strlen(directory) + strlen(name) + 2;
    char *path = malloc(length);
    if (path != NULL) {
        (void)snprintf(path, length, "%s/%s", directory, name);
    }
    return path;
}

static int argmax(const float *values, size_t count) {
    size_t best = 0;
    for (size_t index = 1; index < count; ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return (int)best;
}

static double logit_nmse(const float *actual, const float *reference,
                         size_t count) {
    double error_square = 0.0;
    double reference_square = 0.0;
    for (size_t index = 0; index < count; ++index) {
        double difference = (double)actual[index] - reference[index];
        error_square += difference * difference;
        reference_square += (double)reference[index] * reference[index];
    }
    return error_square / reference_square;
}

static int read_u32_le(const uint8_t *data, size_t size, size_t *offset,
                       uint32_t *value) {
    if (*offset > size || size - *offset < 4) {
        return -1;
    }
    const uint8_t *bytes = data + *offset;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    *offset += 4;
    return 0;
}

static int run_tokenizer_cases(const kipp_model *model, const char *path) {
    uint8_t *data = NULL;
    size_t size = 0;
    size_t offset = 0;
    uint32_t case_count;
    int result = -1;
    if (read_file(path, (void **)&data, &size) != 0 || size < 8 ||
        memcmp(data, "KTOK", 4) != 0) {
        fprintf(stderr, "unable to read tokenizer cases\n");
        goto cleanup;
    }
    offset = 4;
    if (read_u32_le(data, size, &offset, &case_count) != 0) {
        goto cleanup;
    }
    for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
        uint32_t input_length;
        uint32_t normalized_length;
        uint32_t token_count;
        if (read_u32_le(data, size, &offset, &input_length) != 0 ||
            input_length > size - offset) {
            goto cleanup;
        }
        char *input = malloc((size_t)input_length + 1);
        if (input == NULL) {
            goto cleanup;
        }
        memcpy(input, data + offset, input_length);
        input[input_length] = '\0';
        offset += input_length;
        if (read_u32_le(data, size, &offset, &normalized_length) != 0 ||
            normalized_length > size - offset) {
            free(input);
            goto cleanup;
        }
        const uint8_t *normalized = data + offset;
        offset += normalized_length;
        if (read_u32_le(data, size, &offset, &token_count) != 0 ||
            token_count > (size - offset) / sizeof(uint32_t)) {
            free(input);
            goto cleanup;
        }

        uint32_t *expected = malloc((token_count == 0 ? 1 : token_count) *
                                    sizeof(*expected));
        if (expected == NULL) {
            free(input);
            goto cleanup;
        }
        for (uint32_t token = 0; token < token_count; ++token) {
            if (read_u32_le(data, size, &offset, &expected[token]) != 0) {
                free(expected);
                free(input);
                goto cleanup;
            }
        }
        kipp_tokens actual = {0};
        kipp_error error = {0};
        char *decoded = NULL;
        size_t decoded_length = 0;
        if (kipp_tokenize(model, input, &actual, &error) != 0 ||
            actual.count != token_count ||
            memcmp(actual.data, expected,
                   (size_t)token_count * sizeof(*expected)) != 0 ||
            kipp_detokenize(model, actual.data, actual.count, &decoded,
                            &decoded_length, &error) != 0 ||
            decoded_length != normalized_length ||
            memcmp(decoded, normalized, normalized_length) != 0) {
            fprintf(stderr, "tokenizer case %u failed: %s\n", case_index,
                    error.message);
            kipp_text_free(decoded);
            kipp_tokens_free(&actual);
            free(expected);
            free(input);
            goto cleanup;
        }
        kipp_text_free(decoded);
        kipp_tokens_free(&actual);
        free(expected);
        free(input);
    }
    if (offset != size) {
        fprintf(stderr, "tokenizer case file has trailing data\n");
        goto cleanup;
    }
    fprintf(stderr, "TOKENIZER %u golden cases passed\n", case_count);
    result = 0;

cleanup:
    free(data);
    return result;
}

static int run_model_test(const char *model_path, const char *vector_directory) {
    char *prompt_path = join_path(vector_directory, "prompt.txt");
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    char *logits_path = join_path(vector_directory, "logits.f32");
    char *tokenizer_cases_path =
        join_path(vector_directory, "tokenizer-cases.bin");
    char *prompt = NULL;
    uint32_t *expected_tokens = NULL;
    float *expected_logits = NULL;
    size_t prompt_bytes = 0;
    size_t token_bytes = 0;
    size_t logit_bytes = 0;
    kipp_model *model = NULL;
    kipp_tokens actual_tokens = {0};
    float *actual_logits = NULL;
    kipp_error error = {0};
    int result = -1;

    if (prompt_path == NULL || tokens_path == NULL || logits_path == NULL ||
        tokenizer_cases_path == NULL ||
        read_file(prompt_path, (void **)&prompt, &prompt_bytes) != 0 ||
        read_file(tokens_path, (void **)&expected_tokens, &token_bytes) != 0 ||
        read_file(logits_path, (void **)&expected_logits, &logit_bytes) != 0) {
        fprintf(stderr, "unable to read model test vectors: %s\n", strerror(errno));
        goto cleanup;
    }
    prompt = realloc(prompt, prompt_bytes + 1);
    if (prompt == NULL) {
        goto cleanup;
    }
    prompt[prompt_bytes] = '\0';
    if (token_bytes % sizeof(uint32_t) != 0 ||
        logit_bytes != KIPP_VOCAB_SIZE * sizeof(float)) {
        fprintf(stderr, "test vectors have invalid lengths\n");
        goto cleanup;
    }
    size_t expected_token_count = token_bytes / sizeof(uint32_t);

    if (kipp_model_open(model_path, &model, &error) != 0 ||
        run_tokenizer_cases(model, tokenizer_cases_path) != 0 ||
        kipp_tokenize(model, prompt, &actual_tokens, &error) != 0) {
        fprintf(stderr, "model setup failed: %s\n", error.message);
        goto cleanup;
    }
    if (actual_tokens.count != expected_token_count ||
        memcmp(actual_tokens.data, expected_tokens, token_bytes) != 0) {
        fprintf(stderr, "token IDs do not match the golden vector\n");
        goto cleanup;
    }
    actual_logits = malloc(logit_bytes);
    if (actual_logits == NULL ||
        kipp_model_eval(model, actual_tokens.data, actual_tokens.count,
                        actual_logits, KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "model evaluation failed: %s\n", error.message);
        goto cleanup;
    }

    double error_square = 0.0;
    double reference_square = 0.0;
    for (size_t index = 0; index < KIPP_VOCAB_SIZE; ++index) {
        double difference = (double)actual_logits[index] - expected_logits[index];
        error_square += difference * difference;
        reference_square +=
            (double)expected_logits[index] * expected_logits[index];
    }
    double nmse = error_square / reference_square;
    int expected_argmax = argmax(expected_logits, KIPP_VOCAB_SIZE);
    int actual_argmax = argmax(actual_logits, KIPP_VOCAB_SIZE);
    fprintf(stderr, "MODEL nmse=%.9g expected_argmax=%d actual_argmax=%d\n",
            nmse, expected_argmax, actual_argmax);
    if (nmse > 1.0e-5 || expected_argmax != actual_argmax) {
        fprintf(stderr, "model output failed the Phase 1 tolerance\n");
        goto cleanup;
    }
    result = 0;

cleanup:
    free(prompt_path);
    free(tokens_path);
    free(logits_path);
    free(tokenizer_cases_path);
    free(prompt);
    free(expected_tokens);
    free(expected_logits);
    free(actual_logits);
    kipp_tokens_free(&actual_tokens);
    (void)kipp_model_close(model, NULL);
    return result;
}

static int phase2_compare(const char *label, const float *actual,
                          const float *reference, double tolerance) {
    double nmse = logit_nmse(actual, reference, KIPP_VOCAB_SIZE);
    int actual_argmax = argmax(actual, KIPP_VOCAB_SIZE);
    int reference_argmax = argmax(reference, KIPP_VOCAB_SIZE);
    fprintf(stderr,
            "PHASE2 %s nmse=%.9g reference_argmax=%d cached_argmax=%d\n",
            label, nmse, reference_argmax, actual_argmax);
    return nmse <= tolerance && actual_argmax == reference_argmax ? 0 : -1;
}

static int run_phase2_test(const char *model_path,
                           const char *vector_directory) {
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *tokens = NULL;
    size_t token_bytes = 0;
    size_t token_count;
    kipp_model *model = NULL;
    kipp_session *session = NULL;
    kipp_session *other = NULL;
    kipp_session *invalid = NULL;
    float *oracle_logits = NULL;
    float *cached_logits = NULL;
    float *first_oracle = NULL;
    float *final_cached = NULL;
    kipp_error error = {0};
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&tokens, &token_bytes) != 0 ||
        token_bytes == 0 || token_bytes % sizeof(*tokens) != 0) {
        fprintf(stderr, "unable to read Phase 2 token vector\n");
        goto cleanup;
    }
    token_count = token_bytes / sizeof(*tokens);
    if (token_count > UINT32_MAX) {
        fprintf(stderr, "Phase 2 token vector is too large\n");
        goto cleanup;
    }
    oracle_logits = malloc(KIPP_VOCAB_SIZE * sizeof(*oracle_logits));
    cached_logits = malloc(KIPP_VOCAB_SIZE * sizeof(*cached_logits));
    first_oracle = malloc(KIPP_VOCAB_SIZE * sizeof(*first_oracle));
    final_cached = malloc(KIPP_VOCAB_SIZE * sizeof(*final_cached));
    if (oracle_logits == NULL || cached_logits == NULL || first_oracle == NULL ||
        final_cached == NULL) {
        fprintf(stderr, "unable to allocate Phase 2 logits\n");
        goto cleanup;
    }
    if (kipp_model_open(model_path, &model, &error) != 0) {
        fprintf(stderr, "Phase 2 model open failed: %s\n", error.message);
        goto cleanup;
    }

    if (kipp_session_create(model, 0, &invalid, &error) == 0 ||
        error.code != KIPP_ERROR_RANGE || invalid != NULL ||
        kipp_session_create(model, KIPP_CONTEXT_LENGTH + 1, &invalid, &error) ==
            0 ||
        error.code != KIPP_ERROR_RANGE || invalid != NULL) {
        fprintf(stderr, "Phase 2 capacity validation failed\n");
        goto cleanup;
    }
    if (kipp_session_create(model, (uint32_t)token_count, &session, &error) !=
        0) {
        fprintf(stderr, "Phase 2 session create failed: %s\n", error.message);
        goto cleanup;
    }
    kipp_session_info info;
    if (kipp_session_get_info(session, &info) != 0 ||
        info.capacity != token_count || info.length != 0 ||
        info.cache_bytes != kipp_test_kv_cache_bytes((uint32_t)token_count)) {
        fprintf(stderr, "Phase 2 session metadata is incorrect\n");
        goto cleanup;
    }
    if (kipp_session_eval(session, tokens, 0, cached_logits, KIPP_VOCAB_SIZE,
                          &error) == 0 ||
        error.code != KIPP_ERROR_RANGE) {
        fprintf(stderr, "Phase 2 zero-token append was not rejected\n");
        goto cleanup;
    }
    if (kipp_model_close(model, &error) == 0 ||
        error.code != KIPP_ERROR_ARGUMENT) {
        fprintf(stderr, "model close did not reject an active session\n");
        goto cleanup;
    }

    for (size_t position = 0; position < token_count; ++position) {
        if (kipp_model_eval(model, tokens, position + 1, oracle_logits,
                            KIPP_VOCAB_SIZE, &error) != 0 ||
            kipp_session_eval(session, &tokens[position], 1, cached_logits,
                              KIPP_VOCAB_SIZE, &error) != 0) {
            fprintf(stderr, "Phase 2 position %zu evaluation failed: %s\n",
                    position, error.message);
            goto cleanup;
        }
        char label[32];
        (void)snprintf(label, sizeof(label), "position-%zu", position);
        if (phase2_compare(label, cached_logits, oracle_logits, 1.0e-6) != 0) {
            fprintf(stderr, "Phase 2 position %zu failed tolerance\n", position);
            goto cleanup;
        }
        if (position == 0) {
            memcpy(first_oracle, oracle_logits,
                   KIPP_VOCAB_SIZE * sizeof(*first_oracle));
        }
        if (position + 1 == token_count) {
            memcpy(final_cached, cached_logits,
                   KIPP_VOCAB_SIZE * sizeof(*final_cached));
        }
        if (kipp_session_get_info(session, &info) != 0 ||
            info.length != position + 1) {
            fprintf(stderr, "Phase 2 logical length is incorrect\n");
            goto cleanup;
        }
    }
    if (kipp_session_eval(session, tokens, 1, cached_logits, KIPP_VOCAB_SIZE,
                          &error) == 0 ||
        error.code != KIPP_ERROR_RANGE ||
        kipp_session_get_info(session, &info) != 0 ||
        info.length != token_count) {
        fprintf(stderr, "Phase 2 exact-capacity overflow check failed\n");
        goto cleanup;
    }

    if (kipp_session_create(model, (uint32_t)token_count, &other, &error) != 0 ||
        kipp_session_get_info(other, &info) != 0 || info.length != 0 ||
        kipp_session_eval(other, tokens, token_count, cached_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        phase2_compare("multi-token-prefill", cached_logits, final_cached,
                       1.0e-6) != 0) {
        fprintf(stderr, "Phase 2 independent prefill session failed: %s\n",
                error.message);
        goto cleanup;
    }
    if (kipp_session_reset(other, &error) != 0 ||
        kipp_session_get_info(other, &info) != 0 || info.length != 0 ||
        kipp_session_eval(other, tokens, 1, cached_logits, KIPP_VOCAB_SIZE,
                          &error) != 0 ||
        phase2_compare("reset-reuse", cached_logits, first_oracle, 1.0e-6) !=
            0) {
        fprintf(stderr, "Phase 2 reset/reuse failed: %s\n", error.message);
        goto cleanup;
    }
    if (kipp_session_reset(session, &error) != 0 ||
        kipp_session_get_info(session, &info) != 0 || info.length != 0 ||
        kipp_session_get_info(other, &info) != 0 || info.length != 1) {
        fprintf(stderr, "Phase 2 session independence failed\n");
        goto cleanup;
    }

    if (kipp_session_eval(session, tokens, token_count, cached_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_session_truncate(session, (uint32_t)token_count + 1, &error) ==
            0 ||
        error.code != KIPP_ERROR_RANGE ||
        kipp_session_truncate(session, 1, &error) != 0 ||
        kipp_session_get_info(session, &info) != 0 || info.length != 1 ||
        kipp_session_eval(session, tokens + 1, token_count - 1,
                          cached_logits, KIPP_VOCAB_SIZE, &error) != 0 ||
        phase2_compare("truncate-resume", cached_logits, final_cached,
                       1.0e-6) != 0) {
        fprintf(stderr, "Phase 2 truncate/resume failed: %s\n",
                error.message);
        goto cleanup;
    }

    {
        kipp_session *first = NULL;
        kipp_session *second = NULL;
        float *first_logits = malloc(KIPP_VOCAB_SIZE * sizeof(*first_logits));
        float *second_logits =
            malloc(KIPP_VOCAB_SIZE * sizeof(*second_logits));
        int batch_ok = 0;
        if (first_logits == NULL || second_logits == NULL ||
            kipp_session_create(model, (uint32_t)token_count, &first,
                                &error) != 0 ||
            kipp_session_create(model, (uint32_t)token_count, &second,
                                &error) != 0) {
            fprintf(stderr, "Phase 2 batch setup failed: %s\n",
                    error.message);
        } else {
            kipp_batch_item duplicate[2] = {
                {first, tokens, token_count, first_logits},
                {first, tokens, token_count, second_logits},
            };
            kipp_batch_item mixed[2] = {
                {first, tokens, token_count, first_logits},
                {second, tokens + 1, token_count - 1, second_logits},
            };
            if (kipp_eval_batch(model, duplicate, 2, &error) == 0 ||
                error.code != KIPP_ERROR_ARGUMENT) {
                fprintf(stderr,
                        "Phase 2 batch accepted a duplicated session\n");
            } else if (kipp_eval_batch(model, mixed, 2, &error) != 0) {
                fprintf(stderr, "Phase 2 batch eval failed: %s\n",
                        error.message);
            } else if (phase2_compare("batch-full", first_logits,
                                      final_cached, 1.0e-6) != 0 ||
                       kipp_session_reset(other, &error) != 0 ||
                       kipp_session_eval(other, tokens + 1,
                                         token_count - 1, cached_logits,
                                         KIPP_VOCAB_SIZE, &error) != 0 ||
                       phase2_compare("batch-suffix", second_logits,
                                      cached_logits, 1.0e-6) != 0) {
                fprintf(stderr, "Phase 2 batch outputs diverged\n");
            } else {
                kipp_session_info first_info;
                kipp_session_info second_info;
                if (kipp_session_get_info(first, &first_info) != 0 ||
                    first_info.length != token_count ||
                    kipp_session_get_info(second, &second_info) != 0 ||
                    second_info.length != token_count - 1) {
                    fprintf(stderr, "Phase 2 batch lengths are wrong\n");
                } else {
                    batch_ok = 1;
                }
            }
        }
        kipp_session_destroy(first);
        kipp_session_destroy(second);
        free(first_logits);
        free(second_logits);
        if (!batch_ok) {
            goto cleanup;
        }
    }

    fprintf(stderr, "PHASE2 lifecycle passed cache_bytes=%llu capacity=%zu\n",
            (unsigned long long)kipp_test_kv_cache_bytes((uint32_t)token_count),
            token_count);
    result = 0;

cleanup:
    kipp_session_destroy(other);
    kipp_session_destroy(session);
    if (model != NULL && kipp_model_close(model, &error) != 0) {
        fprintf(stderr, "Phase 2 model close failed: %s\n", error.message);
        result = -1;
    }
    free(tokens_path);
    free(tokens);
    free(oracle_logits);
    free(cached_logits);
    free(first_oracle);
    free(final_cached);
    return result;
}

#ifdef KIPP_ENABLE_METAL
static int phase3_compare(const char *label, const float *metal,
                          const float *cpu) {
    double nmse = logit_nmse(metal, cpu, KIPP_VOCAB_SIZE);
    int metal_argmax = argmax(metal, KIPP_VOCAB_SIZE);
    int cpu_argmax = argmax(cpu, KIPP_VOCAB_SIZE);
    fprintf(stderr,
            "PHASE3 %s nmse=%.9g cpu_argmax=%d metal_argmax=%d\n", label,
            nmse, cpu_argmax, metal_argmax);
    return nmse <= 1.0e-4 && metal_argmax == cpu_argmax ? 0 : -1;
}
#endif

static int run_phase3_test(const char *model_path,
                           const char *vector_directory) {
#ifndef KIPP_ENABLE_METAL
    (void)model_path;
    (void)vector_directory;
    fprintf(stderr, "Metal is not compiled into this test binary\n");
    return -1;
#else
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *tokens = NULL;
    size_t token_bytes = 0;
    size_t token_count;
    kipp_model *cpu_model = NULL;
    kipp_model *metal_model = NULL;
    kipp_session *cpu_session = NULL;
    kipp_session *metal_session = NULL;
    float *cpu_logits = NULL;
    float *metal_logits = NULL;
    kipp_error error = {0};
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&tokens, &token_bytes) != 0 ||
        token_bytes == 0 || token_bytes % sizeof(*tokens) != 0) {
        fprintf(stderr, "unable to read Phase 3 token vector\n");
        goto cleanup;
    }
    token_count = token_bytes / sizeof(*tokens);
    if (token_count > UINT32_MAX) {
        fprintf(stderr, "Phase 3 token vector is too large\n");
        goto cleanup;
    }
    cpu_logits = malloc(KIPP_VOCAB_SIZE * sizeof(*cpu_logits));
    metal_logits = malloc(KIPP_VOCAB_SIZE * sizeof(*metal_logits));
    if (cpu_logits == NULL || metal_logits == NULL) {
        fprintf(stderr, "unable to allocate Phase 3 logits\n");
        goto cleanup;
    }
    if (kipp_model_open_backend(model_path, KIPP_BACKEND_CPU, &cpu_model,
                                &error) != 0 ||
        kipp_model_open_backend(model_path, KIPP_BACKEND_METAL, &metal_model,
                                &error) != 0) {
        fprintf(stderr, "Phase 3 model open failed: %s\n", error.message);
        goto cleanup;
    }
    kipp_model_info cpu_info;
    kipp_model_info metal_info;
    if (kipp_model_get_info(cpu_model, &cpu_info) != 0 ||
        kipp_model_get_info(metal_model, &metal_info) != 0 ||
        cpu_info.backend != KIPP_BACKEND_CPU ||
        metal_info.backend != KIPP_BACKEND_METAL) {
        fprintf(stderr, "Phase 3 backend selection was not preserved\n");
        goto cleanup;
    }
    if (kipp_session_create(cpu_model, (uint32_t)token_count, &cpu_session,
                            &error) != 0 ||
        kipp_session_create(metal_model, (uint32_t)token_count, &metal_session,
                            &error) != 0) {
        fprintf(stderr, "Phase 3 session create failed: %s\n", error.message);
        goto cleanup;
    }
    for (size_t position = 0; position < token_count; ++position) {
        if (kipp_session_eval(cpu_session, &tokens[position], 1, cpu_logits,
                              KIPP_VOCAB_SIZE, &error) != 0 ||
            kipp_session_eval(metal_session, &tokens[position], 1,
                              metal_logits, KIPP_VOCAB_SIZE, &error) != 0) {
            fprintf(stderr, "Phase 3 decode position %zu failed: %s\n",
                    position, error.message);
            goto cleanup;
        }
        char label[32];
        (void)snprintf(label, sizeof(label), "decode-position-%zu", position);
        if (phase3_compare(label, metal_logits, cpu_logits) != 0) {
            fprintf(stderr, "Phase 3 decode position %zu failed tolerance\n",
                    position);
            goto cleanup;
        }
    }

    if (kipp_session_reset(cpu_session, &error) != 0 ||
        kipp_session_reset(metal_session, &error) != 0 ||
        kipp_session_eval(cpu_session, tokens, token_count, cpu_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_session_eval(metal_session, tokens, token_count, metal_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        phase3_compare("multi-token-prefill", metal_logits, cpu_logits) != 0) {
        fprintf(stderr, "Phase 3 prefill failed: %s\n", error.message);
        goto cleanup;
    }

    if (kipp_model_eval(cpu_model, tokens, token_count, cpu_logits,
                        KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_model_eval(metal_model, tokens, token_count, metal_logits,
                        KIPP_VOCAB_SIZE, &error) != 0 ||
        phase3_compare("stateless-prefill", metal_logits, cpu_logits) != 0) {
        fprintf(stderr, "Phase 3 stateless path failed: %s\n", error.message);
        goto cleanup;
    }

    if (kipp_session_reset(metal_session, &error) != 0 ||
        kipp_session_eval(metal_session, tokens, 1, metal_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_model_eval(cpu_model, tokens, 1, cpu_logits, KIPP_VOCAB_SIZE,
                        &error) != 0 ||
        phase3_compare("reset-reuse", metal_logits, cpu_logits) != 0) {
        fprintf(stderr, "Phase 3 Metal reset failed: %s\n", error.message);
        goto cleanup;
    }

    if (kipp_session_reset(metal_session, &error) != 0 ||
        kipp_session_eval(metal_session, tokens, token_count, metal_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_session_truncate(metal_session, 1, &error) != 0 ||
        kipp_session_eval(metal_session, tokens + 1, token_count - 1,
                          metal_logits, KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_model_eval(cpu_model, tokens, token_count, cpu_logits,
                        KIPP_VOCAB_SIZE, &error) != 0 ||
        phase3_compare("truncate-resume", metal_logits, cpu_logits) != 0) {
        fprintf(stderr, "Phase 3 truncate/resume failed: %s\n",
                error.message);
        goto cleanup;
    }

    {
        kipp_session *first = NULL;
        kipp_session *second = NULL;
        float *second_logits =
            malloc(KIPP_VOCAB_SIZE * sizeof(*second_logits));
        int batch_ok = 0;
        if (second_logits == NULL ||
            kipp_session_create(metal_model, (uint32_t)token_count, &first,
                                &error) != 0 ||
            kipp_session_create(metal_model, (uint32_t)token_count, &second,
                                &error) != 0) {
            fprintf(stderr, "Phase 3 batch setup failed: %s\n",
                    error.message);
        } else {
            kipp_batch_item mixed[2] = {
                {first, tokens, token_count, metal_logits},
                {second, tokens + 1, token_count - 1, second_logits},
            };
            if (kipp_eval_batch(metal_model, mixed, 2, &error) != 0) {
                fprintf(stderr, "Phase 3 batch eval failed: %s\n",
                        error.message);
            } else if (phase3_compare("batch-full", metal_logits,
                                      cpu_logits) != 0) {
                fprintf(stderr, "Phase 3 batch full-item diverged\n");
            } else if (kipp_model_eval(cpu_model, tokens + 1,
                                       token_count - 1, cpu_logits,
                                       KIPP_VOCAB_SIZE, &error) != 0 ||
                       phase3_compare("batch-suffix", second_logits,
                                      cpu_logits) != 0) {
                fprintf(stderr, "Phase 3 batch suffix diverged\n");
            } else {
                batch_ok = 1;
            }
        }
        kipp_session_destroy(first);
        kipp_session_destroy(second);
        free(second_logits);
        if (!batch_ok) {
            goto cleanup;
        }
    }
    fprintf(stderr, "PHASE3 device=%s prefill_tokens=%zu\n",
            kipp_metal_device_name(), token_count);
    result = 0;

cleanup:
    kipp_session_destroy(metal_session);
    kipp_session_destroy(cpu_session);
    if (metal_model != NULL && kipp_model_close(metal_model, &error) != 0) {
        fprintf(stderr, "Phase 3 Metal close failed: %s\n", error.message);
        result = -1;
    }
    if (cpu_model != NULL && kipp_model_close(cpu_model, &error) != 0) {
        fprintf(stderr, "Phase 3 CPU close failed: %s\n", error.message);
        result = -1;
    }
    free(tokens_path);
    free(tokens);
    free(cpu_logits);
    free(metal_logits);
    return result;
#endif
}

#ifdef KIPP_ENABLE_CUDA
static int phase4_compare(const char *label, const float *cuda,
                          const float *cpu) {
    double nmse = logit_nmse(cuda, cpu, KIPP_VOCAB_SIZE);
    int cuda_argmax = argmax(cuda, KIPP_VOCAB_SIZE);
    int cpu_argmax = argmax(cpu, KIPP_VOCAB_SIZE);
    fprintf(stderr,
            "PHASE4 %s nmse=%.9g cpu_argmax=%d cuda_argmax=%d\n", label,
            nmse, cpu_argmax, cuda_argmax);
    return nmse <= 1.0e-4 && cuda_argmax == cpu_argmax ? 0 : -1;
}
#endif

static int run_phase4_test(const char *model_path,
                           const char *vector_directory) {
#ifndef KIPP_ENABLE_CUDA
    (void)model_path;
    (void)vector_directory;
    fprintf(stderr, "CUDA is not compiled into this test binary\n");
    return -1;
#else
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *tokens = NULL;
    size_t token_bytes = 0;
    size_t token_count;
    kipp_model *cpu_model = NULL;
    kipp_model *cuda_model = NULL;
    kipp_session *cpu_session = NULL;
    kipp_session *cuda_session = NULL;
    float *cpu_logits = NULL;
    float *cuda_logits = NULL;
    kipp_error error = {0};
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&tokens, &token_bytes) != 0 ||
        token_bytes == 0 || token_bytes % sizeof(*tokens) != 0) {
        fprintf(stderr, "unable to read Phase 4 token vector\n");
        goto cleanup;
    }
    token_count = token_bytes / sizeof(*tokens);
    if (token_count > UINT32_MAX) {
        fprintf(stderr, "Phase 4 token vector is too large\n");
        goto cleanup;
    }
    cpu_logits = malloc(KIPP_VOCAB_SIZE * sizeof(*cpu_logits));
    cuda_logits = malloc(KIPP_VOCAB_SIZE * sizeof(*cuda_logits));
    if (cpu_logits == NULL || cuda_logits == NULL) {
        fprintf(stderr, "unable to allocate Phase 4 logits\n");
        goto cleanup;
    }
    if (kipp_model_open_backend(model_path, KIPP_BACKEND_CPU, &cpu_model,
                                &error) != 0 ||
        kipp_model_open_backend(model_path, KIPP_BACKEND_CUDA, &cuda_model,
                                &error) != 0) {
        fprintf(stderr, "Phase 4 model open failed: %s\n", error.message);
        goto cleanup;
    }
    kipp_model_info cpu_info;
    kipp_model_info cuda_info;
    if (kipp_model_get_info(cpu_model, &cpu_info) != 0 ||
        kipp_model_get_info(cuda_model, &cuda_info) != 0 ||
        cpu_info.backend != KIPP_BACKEND_CPU ||
        cuda_info.backend != KIPP_BACKEND_CUDA) {
        fprintf(stderr, "Phase 4 backend selection was not preserved\n");
        goto cleanup;
    }
    if (kipp_session_create(cpu_model, (uint32_t)token_count, &cpu_session,
                            &error) != 0 ||
        kipp_session_create(cuda_model, (uint32_t)token_count, &cuda_session,
                            &error) != 0) {
        fprintf(stderr, "Phase 4 session create failed: %s\n", error.message);
        goto cleanup;
    }
    for (size_t position = 0; position < token_count; ++position) {
        if (kipp_session_eval(cpu_session, &tokens[position], 1, cpu_logits,
                              KIPP_VOCAB_SIZE, &error) != 0 ||
            kipp_session_eval(cuda_session, &tokens[position], 1, cuda_logits,
                              KIPP_VOCAB_SIZE, &error) != 0) {
            fprintf(stderr, "Phase 4 decode position %zu failed: %s\n",
                    position, error.message);
            goto cleanup;
        }
        char label[32];
        (void)snprintf(label, sizeof(label), "decode-position-%zu", position);
        if (phase4_compare(label, cuda_logits, cpu_logits) != 0) {
            fprintf(stderr, "Phase 4 decode position %zu failed tolerance\n",
                    position);
            goto cleanup;
        }
    }

    if (kipp_session_reset(cpu_session, &error) != 0 ||
        kipp_session_reset(cuda_session, &error) != 0 ||
        kipp_session_eval(cpu_session, tokens, token_count, cpu_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_session_eval(cuda_session, tokens, token_count, cuda_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        phase4_compare("multi-token-prefill", cuda_logits, cpu_logits) != 0) {
        fprintf(stderr, "Phase 4 prefill failed: %s\n", error.message);
        goto cleanup;
    }

    if (kipp_model_eval(cpu_model, tokens, token_count, cpu_logits,
                        KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_model_eval(cuda_model, tokens, token_count, cuda_logits,
                        KIPP_VOCAB_SIZE, &error) != 0 ||
        phase4_compare("stateless-prefill", cuda_logits, cpu_logits) != 0) {
        fprintf(stderr, "Phase 4 stateless path failed: %s\n", error.message);
        goto cleanup;
    }

    if (kipp_session_reset(cuda_session, &error) != 0 ||
        kipp_session_eval(cuda_session, tokens, 1, cuda_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_model_eval(cpu_model, tokens, 1, cpu_logits, KIPP_VOCAB_SIZE,
                        &error) != 0 ||
        phase4_compare("reset-reuse", cuda_logits, cpu_logits) != 0) {
        fprintf(stderr, "Phase 4 CUDA reset failed: %s\n", error.message);
        goto cleanup;
    }
    if (kipp_session_reset(cuda_session, &error) != 0 ||
        kipp_session_eval(cuda_session, tokens, token_count, cuda_logits,
                          KIPP_VOCAB_SIZE, &error) != 0 ||
        kipp_session_eval(cuda_session, tokens, 1, cuda_logits,
                          KIPP_VOCAB_SIZE, &error) == 0 ||
        error.code != KIPP_ERROR_RANGE) {
        fprintf(stderr, "Phase 4 CUDA capacity guard failed\n");
        goto cleanup;
    }
    fprintf(stderr, "PHASE4 device=%s prefill_tokens=%zu\n",
            kipp_cuda_device_name(), token_count);
    result = 0;

cleanup:
    kipp_session_destroy(cuda_session);
    kipp_session_destroy(cpu_session);
    if (cuda_model != NULL && kipp_model_close(cuda_model, &error) != 0) {
        fprintf(stderr, "Phase 4 CUDA close failed: %s\n", error.message);
        result = -1;
    }
    if (cpu_model != NULL && kipp_model_close(cpu_model, &error) != 0) {
        fprintf(stderr, "Phase 4 CPU close failed: %s\n", error.message);
        result = -1;
    }
    free(tokens_path);
    free(tokens);
    free(cpu_logits);
    free(cuda_logits);
    return result;
#endif
}

int main(int argc, char **argv) {
    RUN(test_error_names);
    RUN(test_bf16);
    RUN(test_checked_arithmetic);
    RUN(test_kv_layout);
    RUN(test_rms_norm);
    RUN(test_matvec);
    RUN(test_rope);
    RUN(test_softmax_and_swiglu);
    RUN(test_causal_gqa);
    RUN(test_pretokenizer);
    RUN(test_nfc_normalization);
    RUN(test_pretokenizer_fuzz);
    RUN(test_nfc_fuzz);
    RUN(test_gguf_reject_fuzz);
    RUN(test_sampler);
    RUN(test_public_argument_checks);
    RUN(test_malformed_gguf);

    if (argc == 2 && strcmp(argv[1], "--metal-operators") == 0) {
#ifdef KIPP_ENABLE_METAL
        kipp_error error = {0};
        ++tests_run;
        if (kipp_metal_run_operator_tests(&error) != 0) {
            ++failures;
            fprintf(stderr, "FAIL metal_operators: %s\n", error.message);
        } else {
            fprintf(stderr, "PASS metal_operators device=%s\n",
                    kipp_metal_device_name());
        }
#else
        fprintf(stderr, "Metal is not compiled into this test binary\n");
        return 2;
#endif
    } else if (argc == 2 && strcmp(argv[1], "--cuda-operators") == 0) {
#ifdef KIPP_ENABLE_CUDA
        kipp_error error = {0};
        ++tests_run;
        if (kipp_cuda_run_operator_tests(&error) != 0) {
            ++failures;
            fprintf(stderr, "FAIL cuda_operators: %s\n", error.message);
        } else {
            fprintf(stderr, "PASS cuda_operators device=%s\n",
                    kipp_cuda_device_name());
        }
#else
        fprintf(stderr, "CUDA is not compiled into this test binary\n");
        return 2;
#endif
    } else if (argc == 4 && strcmp(argv[1], "--model") == 0) {
        ++tests_run;
        if (run_model_test(argv[2], argv[3]) != 0) {
            ++failures;
            fprintf(stderr, "FAIL model_integration\n");
        } else {
            fprintf(stderr, "PASS model_integration\n");
        }
    } else if (argc == 4 && strcmp(argv[1], "--phase2-model") == 0) {
        ++tests_run;
        if (run_phase2_test(argv[2], argv[3]) != 0) {
            ++failures;
            fprintf(stderr, "FAIL phase2_integration\n");
        } else {
            fprintf(stderr, "PASS phase2_integration\n");
        }
    } else if (argc == 4 && strcmp(argv[1], "--phase3-metal") == 0) {
        ++tests_run;
        if (run_phase3_test(argv[2], argv[3]) != 0) {
            ++failures;
            fprintf(stderr, "FAIL phase3_metal\n");
        } else {
            fprintf(stderr, "PASS phase3_metal\n");
        }
    } else if (argc == 4 && strcmp(argv[1], "--phase4-cuda") == 0) {
        ++tests_run;
        if (run_phase4_test(argv[2], argv[3]) != 0) {
            ++failures;
            fprintf(stderr, "FAIL phase4_cuda\n");
        } else {
            fprintf(stderr, "PASS phase4_cuda\n");
        }
    } else if (argc != 1) {
        fprintf(stderr,
                "Usage: %s [--metal-operators|--cuda-operators] "
                "[--model|--phase2-model|--phase3-metal|--phase4-cuda "
                "MODEL.gguf "
                "VECTOR_DIRECTORY]\n",
                argv[0]);
        return 2;
    }
    fprintf(stderr, "%d test%s, %d failure%s\n", tests_run,
            tests_run == 1 ? "" : "s", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
