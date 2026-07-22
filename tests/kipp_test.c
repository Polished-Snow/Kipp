#define _POSIX_C_SOURCE 200809L

#include "kipp.h"
#include "kipp_chat.h"
#include "kipp_http.h"
#include "kipp_json.h"
#include "kipp_kv_pool.h"
#include "kipp_spec.h"
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
    CHECK(kipp_test_kv_cache_bytes(36, 1) == UINT64_C(147456));
    CHECK(kipp_test_kv_cache_bytes(36, 8192) == UINT64_C(1207959552));
    CHECK(kipp_test_kv_cache_bytes(28, 1) == UINT64_C(114688));
    CHECK(kipp_test_kv_cache_bytes(64, 1) == UINT64_C(262144));
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
    const size_t query_head_count = 32; /* the 4B/8B layout */
    const size_t query_values =
        token_count * query_head_count * KIPP_ATTENTION_HEAD_DIM;
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
    kipp_test_causal_gqa(query, key, value, output, scores, token_count,
                         query_head_count);
    for (size_t head = 0; head < query_head_count; ++head) {
        CHECK(nearly_equal(
            output[head * KIPP_ATTENTION_HEAD_DIM], 1.0f, 1.0e-6f));
        CHECK(nearly_equal(
            output[(query_head_count + head) * KIPP_ATTENTION_HEAD_DIM],
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

/*
 * The JSON parser reads untrusted request bodies, so fuzz it the same way
 * the tokenizer and GGUF loader are fuzzed: deterministic xorshift rounds
 * alternating between mutated valid documents (to get past the first
 * bytes) and pure garbage. Sanitizer builds are the real oracle; here we
 * assert no crash, a sane result type, and clean rejection of trailing
 * garbage and depth bombs.
 */
static void test_json_parse_fuzz(void) {
    static const char *const corpus[] = {
        ("{\"prompt\": \"The capital of France is\", \"max_tokens\": 8, "
         "\"temperature\": 0, \"stop\": [\"\\n\"], "
         "\"logit_bias\": {\"42\": -1.5}}"),
        ("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], "
         "\"stream\": true, \"chat_template_kwargs\": "
         "{\"enable_thinking\": false}}"),
        ("[[1, 2.5, -3e-2], {\"k\": [null, true, false]}, \"\\u00e9\", "
         "\"\\ud83d\\ude00\", \"\\\\\\\"\\n\"]"),
        "{\"a\": {\"b\": {\"c\": [0, 1e10, -0.25, 151935]}}}",
    };
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    char buffer[512];

    /* The unmutated corpus must parse. */
    for (size_t doc = 0; doc < sizeof(corpus) / sizeof(corpus[0]); ++doc) {
        kipp_json_value value;
        CHECK(kipp_json_parse(corpus[doc], strlen(corpus[doc]), &value));
        CHECK(value.type == KIPP_JSON_OBJECT ||
              value.type == KIPP_JSON_ARRAY);
        kipp_json_free(&value);
    }

    /* Trailing garbage and over-deep nesting are rejections. */
    kipp_json_value rejected;
    CHECK(!kipp_json_parse("{} x", 4, &rejected));
    char deep[2 * (KIPP_JSON_DEPTH_LIMIT + 8)];
    size_t half = KIPP_JSON_DEPTH_LIMIT + 8;
    for (size_t index = 0; index < half; ++index) {
        deep[index] = '[';
        deep[half + index] = ']';
    }
    CHECK(!kipp_json_parse(deep, sizeof(deep), &rejected));

    for (int round = 0; round < 400; ++round) {
        size_t length;
        if (round % 2 == 0) {
            const char *doc =
                corpus[fuzz_next(&state) %
                       (sizeof(corpus) / sizeof(corpus[0]))];
            length = strlen(doc);
            memcpy(buffer, doc, length);
            size_t mutations = 1 + fuzz_next(&state) % 4;
            for (size_t index = 0; index < mutations; ++index) {
                switch (fuzz_next(&state) % 3) {
                case 0: /* flip a byte */
                    buffer[fuzz_next(&state) % length] =
                        (char)(fuzz_next(&state) & 0xff);
                    break;
                case 1: /* truncate */
                    length = 1 + fuzz_next(&state) % length;
                    break;
                default: /* duplicate a prefix byte at the end */
                    if (length + 1 < sizeof(buffer)) {
                        buffer[length] =
                            buffer[fuzz_next(&state) % length];
                        ++length;
                    }
                    break;
                }
            }
        } else {
            length = 1 + fuzz_next(&state) % 256;
            for (size_t index = 0; index < length; ++index) {
                buffer[index] = (char)(fuzz_next(&state) & 0xff);
            }
        }
        kipp_json_value value;
        if (kipp_json_parse(buffer, length, &value)) {
            CHECK(value.type <= KIPP_JSON_OBJECT);
            kipp_json_free(&value);
        }
    }
}

/*
 * Header lookup scans raw network bytes; fuzz it with hostile header
 * blocks and require that any returned pointer lands inside the input.
 */
static void test_http_header_fuzz(void) {
    uint64_t state = UINT64_C(0xda3e39cb94b95bdb);
    char buffer[512];
    for (int round = 0; round < 400; ++round) {
        size_t length = 0;
        if (round % 2 == 0) {
            /* Plausible header lines with mutations. */
            static const char *const lines[] = {
                "Content-Length: 42\r\n",
                "content-length:9\r\n",
                "CONTENT-TYPE: application/json\r\n",
                "X-Long-Name-That-Never-Matches-Anything: v\r\n",
                "NoColonHere\r\n",
                "Content-Length\r: 7\r\n",
                ": empty-name\r\n",
                "\r\n",
            };
            size_t line_count = 1 + fuzz_next(&state) % 6;
            for (size_t index = 0; index < line_count; ++index) {
                const char *line =
                    lines[fuzz_next(&state) %
                          (sizeof(lines) / sizeof(lines[0]))];
                size_t line_length = strlen(line);
                if (length + line_length + 1 >= sizeof(buffer)) {
                    break;
                }
                memcpy(buffer + length, line, line_length);
                length += line_length;
            }
            if (length > 0 && fuzz_next(&state) % 4 == 0) {
                buffer[fuzz_next(&state) % length] =
                    (char)(fuzz_next(&state) & 0xff);
            }
        } else {
            length = fuzz_next(&state) % 400;
            for (size_t index = 0; index < length; ++index) {
                char byte = (char)(fuzz_next(&state) & 0xff);
                buffer[index] = byte == '\0' ? ' ' : byte;
            }
        }
        buffer[length] = '\0';
        static const char *const names[] = {"Content-Length",
                                            "Content-Type", "Host"};
        for (size_t name = 0; name < 3; ++name) {
            const char *value =
                kipp_http_header_value(buffer, names[name]);
            CHECK(value == NULL ||
                  (value >= buffer && value <= buffer + length));
        }
    }

    /* Deterministic behavior checks. */
    const char *block =
        "Host: localhost\r\ncontent-length:  17\r\nX: y\r\n";
    const char *value = kipp_http_header_value(block, "Content-Length");
    CHECK(value != NULL && strncmp(value, "17", 2) == 0);
    CHECK(kipp_http_header_value(block, "Accept") == NULL);
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

static void test_quant_matvec(void) {
    /* fp16 decode: 0.5 = 0x3800, -1.0 = 0xBC00, 0.25 = 0x3400. */
    CHECK(kipp_test_fp16_to_float(0x3800) == 0.5f);
    CHECK(kipp_test_fp16_to_float(0xBC00) == -1.0f);
    CHECK(kipp_test_fp16_to_float(0x3400) == 0.25f);
    CHECK(kipp_test_fp16_to_float(0x0000) == 0.0f);

    /* Q8_0, one 32-block row: d=0.5, qs=[2,-3,0..], input=[10,4,7,0..].
     * dot = 2*10 + (-3)*4 = 8; output = 0.5*8 = 4.0. */
    uint8_t q8[34] = {0};
    q8[0] = 0x00;
    q8[1] = 0x38; /* 0x3800 little-endian */
    q8[2] = 2;
    q8[3] = (uint8_t)(int8_t)-3;
    float q8_input[32] = {10.0f, 4.0f, 7.0f};
    float q8_out = 0.0f;
    kipp_test_matvec_q8_0(q8, q8_input, &q8_out, 1, 32);
    CHECK(nearly_equal(q8_out, 4.0f, 1.0e-6f));

    /* AFFINE4_GS32, one 32-group row: scale=0.25, bias=-1.0,
     * q=[3,12,0..] -> packed[0]=0xC3; input=[8,2,5,0..].
     * dot=3*8+12*2=48, actsum=15; out=0.25*48 + (-1)*15 = -3.0. */
    uint8_t a4[20] = {0};
    a4[0] = 0xC3; /* q0=3 (lo), q1=12 (hi) */
    a4[16] = 0x00;
    a4[17] = 0x34; /* scale 0.25 */
    a4[18] = 0x00;
    a4[19] = 0xBC; /* bias -1.0 */
    float a4_input[32] = {8.0f, 2.0f, 5.0f};
    float a4_out = 0.0f;
    kipp_test_matvec_affine4_gs32(a4, a4_input, &a4_out, 1, 32);
    CHECK(nearly_equal(a4_out, -3.0f, 1.0e-6f));
}

static void test_sample_ex(void) {
    kipp_error error = {0};
    float logits[6] = {1.0f, 5.0f, 2.0f, 4.0f, 0.0f, 3.0f};
    uint32_t token = UINT32_MAX;
    uint64_t rng = 99;

    /* Greedy over bias-adjusted logits: lift token 4 above the rest. */
    kipp_sample_params biased = {0};
    biased.temperature = 0.0f;
    biased.top_p = 1.0f;
    biased.repetition_penalty = 1.0f;
    uint32_t bias_token = 4;
    float bias_value = 10.0f;
    biased.logit_bias_tokens = &bias_token;
    biased.logit_bias_values = &bias_value;
    biased.logit_bias_count = 1;
    CHECK(kipp_sample_ex(logits, 6, &biased, NULL, &token, &error) == 0);
    CHECK(token == 4);

    /* top_k = 1 with temperature always returns the argmax (token 1). */
    kipp_sample_params topk = {0};
    topk.temperature = 1.0f;
    topk.top_p = 1.0f;
    topk.top_k = 1;
    topk.repetition_penalty = 1.0f;
    for (int draw = 0; draw < 20; ++draw) {
        CHECK(kipp_sample_ex(logits, 6, &topk, &rng, &token, &error) == 0);
        CHECK(token == 1);
    }

    /* Repetition penalty pushes a heavily-repeated top token below a rival. */
    kipp_sample_params penalized = {0};
    penalized.temperature = 0.0f;
    penalized.top_p = 1.0f;
    penalized.repetition_penalty = 2.0f;
    uint32_t recent[4] = {1, 1, 1, 1};
    penalized.recent_tokens = recent;
    penalized.recent_count = 4;
    CHECK(kipp_sample_ex(logits, 6, &penalized, NULL, &token, &error) == 0);
    /* logit[1] 5.0/2 = 2.5 now below logit[3] = 4.0 -> argmax is 3. */
    CHECK(token == 3);

    /* min_p keeps only the dominant token, so sampling is deterministic. */
    float peaked[4] = {10.0f, 0.0f, 0.0f, 0.0f};
    kipp_sample_params minp = {0};
    minp.temperature = 1.0f;
    minp.top_p = 1.0f;
    minp.min_p = 0.5f;
    minp.repetition_penalty = 1.0f;
    rng = 7;
    for (int draw = 0; draw < 20; ++draw) {
        CHECK(kipp_sample_ex(peaked, 4, &minp, &rng, &token, &error) == 0);
        CHECK(token == 0);
    }

    /* Invalid min_p is rejected. */
    kipp_sample_params bad = {0};
    bad.temperature = 1.0f;
    bad.top_p = 1.0f;
    bad.min_p = 1.5f;
    bad.repetition_penalty = 1.0f;
    CHECK(kipp_sample_ex(logits, 6, &bad, &rng, &token, &error) != 0);
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
    /* Pooled opens validate before touching the file or any hardware. */
    kipp_model *pooled_model = NULL;
    CHECK(kipp_model_open_pooled("unused.gguf", KIPP_BACKEND_CPU, 0,
                                 &pooled_model, &error) != 0);
    CHECK(error.code == KIPP_ERROR_RANGE);
    CHECK(pooled_model == NULL);
    CHECK(kipp_model_open_pooled("unused.gguf", KIPP_BACKEND_METAL, 8,
                                 &pooled_model, &error) != 0);
#ifdef KIPP_ENABLE_METAL
    /* Pooled Metal is supported: the open proceeds past the backend check
     * and fails on the missing file instead. */
    CHECK(error.code != KIPP_ERROR_UNSUPPORTED);
#else
    CHECK(error.code == KIPP_ERROR_UNSUPPORTED);
#endif
    CHECK(kipp_model_open_pooled("unused.gguf", KIPP_BACKEND_CUDA, 8,
                                 &pooled_model, &error) != 0);
    CHECK(error.code == KIPP_ERROR_UNSUPPORTED);
    CHECK(kipp_session_match_prefix(NULL, &token, 1, NULL, &error) != 0);
    CHECK(error.code == KIPP_ERROR_ARGUMENT);
    CHECK(kipp_model_kv_pool_stats(NULL, NULL) != 0);
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

static void append_bytes(uint8_t *buffer, size_t *offset, const void *data,
                         size_t count) {
    memcpy(buffer + *offset, data, count);
    *offset += count;
}

static void append_u32(uint8_t *buffer, size_t *offset, uint32_t value) {
    append_bytes(buffer, offset, &value, sizeof(value));
}

static void append_u64(uint8_t *buffer, size_t *offset, uint64_t value) {
    append_bytes(buffer, offset, &value, sizeof(value));
}

static void append_gguf_string(uint8_t *buffer, size_t *offset,
                               const char *text) {
    append_u64(buffer, offset, strlen(text));
    append_bytes(buffer, offset, text, strlen(text));
}

static void append_string_entry(uint8_t *buffer, size_t *offset,
                                const char *key, const char *value) {
    append_gguf_string(buffer, offset, key);
    append_u32(buffer, offset, 8); /* GGUF string type */
    append_gguf_string(buffer, offset, value);
}

/* A structurally valid GGUF whose checkpoint is not in the registry (or
 * whose dimensions contradict its registry entry) must be rejected as
 * unsupported before any tensor data is touched. */
static void test_registry_rejection(void) {
    static const struct {
        const char *repository;
        const char *revision;
    } cases[] = {
        /* Unknown repository. */
        {"Qwen/Qwen3-Unknown", "906bfd4b4dc7f14ee4320094d8b41684abff8539"},
        /* Known repository at an unpinned revision. */
        {"Qwen/Qwen3-4B-Base", "0000000000000000000000000000000000000000"},
        /* Pinned checkpoint whose metadata dims (absent => zero) differ. */
        {"Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539"},
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]);
         ++index) {
        uint8_t buffer[1024];
        size_t offset = 0;
        append_bytes(buffer, &offset, "GGUF", 4);
        append_u32(buffer, &offset, 3);   /* version */
        append_u64(buffer, &offset, 398); /* tensor count (plausible) */
        append_u64(buffer, &offset, 3);   /* metadata count */
        append_string_entry(buffer, &offset, "general.architecture", "qwen3");
        append_string_entry(buffer, &offset, "kipp.source.repository",
                            cases[index].repository);
        append_string_entry(buffer, &offset, "kipp.source.revision",
                            cases[index].revision);

        char path[] = "/tmp/kipp-registry-XXXXXX";
        int descriptor = mkstemp(path);
        CHECK(descriptor >= 0);
        if (descriptor < 0) {
            return;
        }
        CHECK(write(descriptor, buffer, offset) == (ssize_t)offset);
        CHECK(close(descriptor) == 0);
        kipp_model *model = NULL;
        kipp_error error = {0};
        CHECK(kipp_model_open(path, &model, &error) != 0);
        CHECK(model == NULL);
        CHECK(error.code == KIPP_ERROR_UNSUPPORTED);
        CHECK(unlink(path) == 0);
    }
}

/* Byte-for-byte checks against Hugging Face apply_chat_template output
 * (captured in tools/generate_chat_vectors.py). The 2507 instruct variant
 * is non-thinking, so enable_thinking is inert. */
static void test_prompt_lookup(void) {
    uint32_t drafts[8];
    /* Tail 3-gram [1,2,3] recurs at index 0; the 4 tokens after it follow. */
    uint32_t repeat[] = {1, 2, 3, 4, 1, 2, 3};
    size_t count =
        kipp_spec_prompt_lookup(repeat, 7, 3, 1, 4, drafts);
    CHECK(count == 4);
    CHECK(drafts[0] == 4 && drafts[1] == 1 && drafts[2] == 2 &&
          drafts[3] == 3);

    /* max_draft caps the count. */
    count = kipp_spec_prompt_lookup(repeat, 7, 3, 1, 2, drafts);
    CHECK(count == 2 && drafts[0] == 4 && drafts[1] == 1);

    /* No recurrence -> no draft. */
    uint32_t unique[] = {5, 6, 7};
    CHECK(kipp_spec_prompt_lookup(unique, 3, 3, 1, 4, drafts) == 0);

    /* Falls back to a shorter n-gram: tail [9] recurs at index 0, followed
     * by 8. */
    uint32_t single[] = {9, 8, 7, 9};
    count = kipp_spec_prompt_lookup(single, 4, 3, 1, 4, drafts);
    CHECK(count == 3 && drafts[0] == 8 && drafts[1] == 7 && drafts[2] == 9);

    /* Degenerate inputs are safe. */
    CHECK(kipp_spec_prompt_lookup(NULL, 0, 3, 1, 4, drafts) == 0);
    CHECK(kipp_spec_prompt_lookup(repeat, 7, 3, 1, 0, drafts) == 0);
}

static void test_spec_gate(void) {
    kipp_spec_gate gate;

    /* Fresh gate drafts immediately. */
    kipp_spec_gate_init(&gate);
    CHECK(kipp_spec_gate_should_draft(&gate));

    /* Two zero-acceptance drafts suspend it (0.5 -> 0.25 -> 0.125). */
    kipp_spec_gate_record(&gate, 8, 0);
    CHECK(kipp_spec_gate_should_draft(&gate));
    kipp_spec_gate_record(&gate, 8, 0);
    CHECK(!kipp_spec_gate_should_draft(&gate));

    /* Suspended: no draft until the probe interval elapses. */
    for (uint32_t i = 0; i < KIPP_SPEC_GATE_PROBE_INTERVAL - 1; ++i) {
        kipp_spec_gate_tick(&gate);
        CHECK(!kipp_spec_gate_should_draft(&gate));
    }
    kipp_spec_gate_tick(&gate);
    CHECK(kipp_spec_gate_should_draft(&gate));

    /* A weak probe stays suspended and restarts the probe clock. */
    kipp_spec_gate_record(&gate, 8, 1);
    CHECK(!kipp_spec_gate_should_draft(&gate));

    /* A strong probe re-enables and restarts the EMA from the probe. */
    for (uint32_t i = 0; i < KIPP_SPEC_GATE_PROBE_INTERVAL; ++i) {
        kipp_spec_gate_tick(&gate);
    }
    CHECK(kipp_spec_gate_should_draft(&gate));
    kipp_spec_gate_record(&gate, 8, 8);
    CHECK(kipp_spec_gate_should_draft(&gate));

    /* Full acceptance never suspends. */
    for (int i = 0; i < 16; ++i) {
        kipp_spec_gate_record(&gate, 8, 8);
        CHECK(kipp_spec_gate_should_draft(&gate));
    }

    /* Sustained mediocre acceptance below the threshold suspends again. */
    for (int i = 0; i < 16 && kipp_spec_gate_should_draft(&gate); ++i) {
        kipp_spec_gate_record(&gate, 8, 1);
    }
    CHECK(!kipp_spec_gate_should_draft(&gate));

    /* A zero-drafted record is ignored. */
    kipp_spec_gate_init(&gate);
    kipp_spec_gate_record(&gate, 0, 0);
    CHECK(kipp_spec_gate_should_draft(&gate));
}

static void test_chat_render(void) {
    kipp_error error = {0};
    char *out = NULL;

    kipp_chat_message user_only[] = {{KIPP_ROLE_USER, "Hi there"}};
    kipp_chat_options opt = {KIPP_VARIANT_INSTRUCT_2507, true, true};
    CHECK(kipp_chat_render(user_only, 1, &opt, &out, &error) == 0);
    CHECK(out != NULL && strcmp(out,
        "<|im_start|>user\nHi there<|im_end|>\n"
        "<|im_start|>assistant\n") == 0);
    kipp_text_free(out);
    out = NULL;

    kipp_chat_message sys_user[] = {
        {KIPP_ROLE_SYSTEM, "You are terse."},
        {KIPP_ROLE_USER, "Say hi"}};
    CHECK(kipp_chat_render(sys_user, 2, &opt, &out, &error) == 0);
    CHECK(out != NULL && strcmp(out,
        "<|im_start|>system\nYou are terse.<|im_end|>\n"
        "<|im_start|>user\nSay hi<|im_end|>\n"
        "<|im_start|>assistant\n") == 0);
    kipp_text_free(out);
    out = NULL;

    kipp_chat_message multiturn[] = {
        {KIPP_ROLE_USER, "1+1?"},
        {KIPP_ROLE_ASSISTANT, "2"},
        {KIPP_ROLE_USER, "times 3?"}};
    CHECK(kipp_chat_render(multiturn, 3, &opt, &out, &error) == 0);
    CHECK(out != NULL && strcmp(out,
        "<|im_start|>user\n1+1?<|im_end|>\n"
        "<|im_start|>assistant\n2<|im_end|>\n"
        "<|im_start|>user\ntimes 3?<|im_end|>\n"
        "<|im_start|>assistant\n") == 0);
    kipp_text_free(out);
    out = NULL;

    /* No generation prompt. */
    kipp_chat_options no_gen = {KIPP_VARIANT_INSTRUCT_2507, false, true};
    kipp_chat_message hello[] = {{KIPP_ROLE_USER, "Hello"}};
    CHECK(kipp_chat_render(hello, 1, &no_gen, &out, &error) == 0);
    CHECK(out != NULL && strcmp(out,
        "<|im_start|>user\nHello<|im_end|>\n") == 0);
    kipp_text_free(out);
    out = NULL;

    /* Hybrid instruct with thinking off injects the empty think block. */
    kipp_chat_options hybrid_off = {KIPP_VARIANT_INSTRUCT, true, false};
    CHECK(kipp_chat_render(hello, 1, &hybrid_off, &out, &error) == 0);
    CHECK(out != NULL && strcmp(out,
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n") == 0);
    kipp_text_free(out);
    out = NULL;

    /* Thinking-2507 forces an opening <think>. */
    kipp_chat_options thinking = {KIPP_VARIANT_THINKING_2507, true, true};
    CHECK(kipp_chat_render(hello, 1, &thinking, &out, &error) == 0);
    CHECK(out != NULL && strcmp(out,
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n") == 0);
    kipp_text_free(out);
    out = NULL;

    /* Base checkpoints have no chat template. */
    kipp_chat_options base = {KIPP_VARIANT_BASE, true, true};
    CHECK(kipp_chat_render(hello, 1, &base, &out, &error) != 0);
    CHECK(error.code == KIPP_ERROR_UNSUPPORTED);
    CHECK(out == NULL);
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
    /*
     * The per-vector NMSE bound is recorded next to the reference logits
     * (nmse-max.txt): 5e-5 for the FP32 CPU reference (dominated by Kipp's
     * BF16 KV storage contract), looser for a BF16 GPU reference used on
     * checkpoints too large for an FP32 host reference. Argmax must match
     * exactly regardless. Missing file falls back to the FP32 default.
     */
    double nmse_max = 5.0e-5;
    char *nmse_path = join_path(vector_directory, "nmse-max.txt");
    if (nmse_path != NULL) {
        FILE *nmse_file = fopen(nmse_path, "r");
        if (nmse_file != NULL) {
            double parsed;
            if (fscanf(nmse_file, "%lf", &parsed) == 1 && parsed > 0.0) {
                nmse_max = parsed;
            }
            (void)fclose(nmse_file);
        }
        free(nmse_path);
    }
    /*
     * A quantized artifact is gated against the same BF16 reference, so its
     * full-logit NMSE reflects the quantization loss: Q8_0 is near-lossless
     * (stays under the BF16 bound), but 4-bit affine needs a Q4-class bound.
     * Argmax must still match the reference exactly.
     */
    kipp_model_info quant_info;
    if (kipp_model_get_info(model, &quant_info) == 0 &&
        strcmp(quant_info.quant_scheme, "affine4_gs32") == 0 &&
        nmse_max < 3.0e-2) {
        nmse_max = 3.0e-2;
    }
    fprintf(stderr,
            "MODEL nmse=%.9g nmse_max=%.9g expected_argmax=%d "
            "actual_argmax=%d\n",
            nmse, nmse_max, expected_argmax, actual_argmax);
    if (nmse > nmse_max || expected_argmax != actual_argmax) {
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

/*
 * kipp_session_eval_n must write logits for the last N tokens identical to
 * evaluating each of those positions on its own. On CPU the two paths do the
 * same scalar work, so the rows are expected to be bitwise-identical.
 */
static int run_multilogit_test(const char *model_path,
                               const char *vector_directory) {
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *tokens = NULL;
    size_t token_bytes = 0;
    size_t token_count;
    kipp_model *model = NULL;
    kipp_session *multi = NULL;
    kipp_session *single = NULL;
    float *rows = NULL;
    float *reference = NULL;
    kipp_error error = {0};
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&tokens, &token_bytes) != 0 ||
        token_bytes < 2 * sizeof(*tokens) || token_bytes % sizeof(*tokens)) {
        fprintf(stderr, "unable to read multi-logit token vector\n");
        goto cleanup;
    }
    token_count = token_bytes / sizeof(*tokens);
    uint32_t n = (uint32_t)token_count; /* logits for every position */
    rows = malloc((size_t)n * KIPP_VOCAB_SIZE * sizeof(*rows));
    reference = malloc(KIPP_VOCAB_SIZE * sizeof(*reference));
    if (rows == NULL || reference == NULL ||
        kipp_model_open(model_path, &model, &error) != 0) {
        fprintf(stderr, "multi-logit setup failed: %s\n", error.message);
        goto cleanup;
    }

    if (kipp_session_create(model, (uint32_t)token_count, &multi, &error) != 0 ||
        kipp_session_eval_n(multi, tokens, token_count, rows, n, &error) != 0) {
        fprintf(stderr, "multi-logit eval failed: %s\n", error.message);
        goto cleanup;
    }
    if (kipp_session_create(model, (uint32_t)token_count, &single, &error) !=
        0) {
        fprintf(stderr, "multi-logit single session failed: %s\n",
                error.message);
        goto cleanup;
    }
    for (uint32_t position = 0; position < n; ++position) {
        if (kipp_session_eval(single, &tokens[position], 1, reference,
                              KIPP_VOCAB_SIZE, &error) != 0) {
            fprintf(stderr, "multi-logit reference eval failed: %s\n",
                    error.message);
            goto cleanup;
        }
        if (memcmp(rows + (size_t)position * KIPP_VOCAB_SIZE, reference,
                   KIPP_VOCAB_SIZE * sizeof(*reference)) != 0) {
            fprintf(stderr, "MULTILOGIT row %u differs from single eval\n",
                    position);
            goto cleanup;
        }
    }
    fprintf(stderr, "MULTILOGIT %u rows bitwise-identical to single eval\n", n);
    result = 0;

cleanup:
    kipp_session_destroy(single);
    kipp_session_destroy(multi);
    if (model != NULL) {
        (void)kipp_model_close(model, NULL);
    }
    free(tokens_path);
    free(tokens);
    free(rows);
    free(reference);
    return result;
}

/*
 * Paged KV gate: a backend addresses its KV through a per-session block
 * table. Evaluating a prompt with the identity mapping and again with a
 * scrambled (reversed) block table must produce bitwise-identical logits,
 * proving the paged indirection is correct for any physical placement. The
 * identity mapping is itself byte-for-byte the old contiguous path (the
 * --model / --phase3-metal gates confirm it still matches the reference), so
 * together they establish paged == contiguous for that backend.
 */
static int run_paged_test(const char *model_path, const char *vector_directory,
                          kipp_backend_kind backend, const char *label) {
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *seed = NULL;
    uint32_t *tokens = NULL;
    size_t token_bytes = 0;
    size_t seed_count;
    kipp_model *model = NULL;
    kipp_session *identity = NULL;
    kipp_session *scrambled = NULL;
    float *rows_identity = NULL;
    float *rows_scrambled = NULL;
    kipp_error error = {0};
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&seed, &token_bytes) != 0 ||
        token_bytes < sizeof(*seed) || token_bytes % sizeof(*seed)) {
        fprintf(stderr, "unable to read paged-cpu token vector\n");
        goto cleanup;
    }
    seed_count = token_bytes / sizeof(*seed);
    /* Synthesize a multi-block sequence (cycling the pinned, valid token ids)
     * so the final position's attention spans several physically-scrambled
     * blocks. Compare only the final logits: they already depend on every
     * cached position, so this stays cheap (one lm_head) yet exercises the
     * whole paged read/write path.
     *
     * Three blocks, not two: reversing a 2-entry table is a swap, and for a
     * block-rollover bug ("first slot of block b written one past the end of
     * block b-1's physical block, wrapping") the swap composed with the wrap
     * lands on the correct byte again — the mutation study caught the gate
     * itself being degenerate at 2 blocks. Reversal over >= 3 blocks breaks
     * physical adjacency, which is what exposes that fault class. */
    uint32_t n = 3 * KIPP_KV_BLOCK_TOKENS;
    uint32_t capacity = n;
    tokens = malloc((size_t)n * sizeof(*tokens));
    if (tokens == NULL) {
        fprintf(stderr, "paged-cpu token synthesis failed\n");
        goto cleanup;
    }
    for (uint32_t index = 0; index < n; ++index) {
        tokens[index] = seed[index % seed_count];
    }
    size_t token_count = n;
    rows_identity = malloc(KIPP_VOCAB_SIZE * sizeof(float));
    rows_scrambled = malloc(KIPP_VOCAB_SIZE * sizeof(float));
    if (rows_identity == NULL || rows_scrambled == NULL ||
        kipp_model_open_backend(model_path, backend, &model, &error) != 0) {
        fprintf(stderr, "paged-%s setup failed: %s\n", label, error.message);
        goto cleanup;
    }
    if (kipp_session_create(model, capacity, &identity, &error) != 0 ||
        kipp_session_eval(identity, tokens, token_count, rows_identity,
                          KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "paged-%s identity eval failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (kipp_session_create(model, capacity, &scrambled, &error) != 0) {
        fprintf(stderr, "paged-%s scrambled session failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (kipp_test_scramble_session_kv(scrambled) != 0) {
        fprintf(stderr, "paged-%s scramble hook unavailable\n", label);
        goto cleanup;
    }
    if (kipp_session_eval(scrambled, tokens, token_count, rows_scrambled,
                          KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "paged-%s scrambled eval failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (memcmp(rows_identity, rows_scrambled,
               KIPP_VOCAB_SIZE * sizeof(float)) != 0) {
        fprintf(stderr, "PAGED-%s scrambled block table changed logits\n",
                label);
        goto cleanup;
    }
    fprintf(stderr,
            "PAGED-%s final logits over %u tokens (%u blocks) bitwise-"
            "identical under a scrambled block table\n",
            label, n, n / KIPP_KV_BLOCK_TOKENS);
    result = 0;

cleanup:
    kipp_session_destroy(scrambled);
    kipp_session_destroy(identity);
    if (model != NULL) {
        (void)kipp_model_close(model, NULL);
    }
    free(tokens_path);
    free(seed);
    free(tokens);
    free(rows_identity);
    free(rows_scrambled);
    return result;
}

/*
 * Phase 5 pooled-KV gate (CPU). Proves, with the real model, that pooled
 * sessions are bitwise-equal to private-slab sessions; that publish-at-finish
 * prefix sharing reproduces unshared logits bitwise; that shared and cold
 * sessions batch together correctly; that pool exhaustion fails cleanly
 * without disturbing other sessions; that truncation into the private tail
 * stays correct; and that eviction only shrinks future matches.
 */
static int run_pooled_test(const char *model_path,
                           const char *vector_directory,
                           kipp_backend_kind backend, const char *label) {
    enum { POOL_SEQ = 3 * KIPP_KV_BLOCK_TOKENS };
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *seed = NULL;
    size_t token_bytes = 0;
    uint32_t sequence[POOL_SEQ];
    uint32_t alternate[POOL_SEQ];
    kipp_model *reference_model = NULL;
    kipp_model *pooled_model = NULL;
    kipp_model *tiny_model = NULL;
    kipp_session *session = NULL;
    kipp_session *second = NULL;
    kipp_session *third = NULL;
    float *reference_full = NULL;   /* sequence[0..96) */
    float *reference_mixed = NULL;  /* sequence[0..64) + alternate[64..96) */
    float *reference_cut = NULL;    /* sequence[0..70) + alternate[70..96) */
    float *actual = NULL;
    float *actual_second = NULL;
    float *actual_third = NULL;
    kipp_error error = {0};
    kipp_kv_pool_stats_public stats;
    uint32_t matched = 0;
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&seed, &token_bytes) != 0 ||
        token_bytes < sizeof(*seed) || token_bytes % sizeof(*seed)) {
        fprintf(stderr, "unable to read pooled-cpu token vector\n");
        goto cleanup;
    }
    size_t seed_count = token_bytes / sizeof(*seed);
    for (uint32_t index = 0; index < POOL_SEQ; ++index) {
        sequence[index] = seed[index % seed_count];
        /* A diverging continuation built from the same valid ids. */
        alternate[index] = seed[(index + seed_count / 2 + 1) % seed_count];
    }
    reference_full = malloc(KIPP_VOCAB_SIZE * sizeof(*reference_full));
    reference_mixed = malloc(KIPP_VOCAB_SIZE * sizeof(*reference_mixed));
    reference_cut = malloc(KIPP_VOCAB_SIZE * sizeof(*reference_cut));
    actual = malloc(KIPP_VOCAB_SIZE * sizeof(*actual));
    actual_second = malloc(KIPP_VOCAB_SIZE * sizeof(*actual_second));
    actual_third = malloc(KIPP_VOCAB_SIZE * sizeof(*actual_third));
    if (reference_full == NULL || reference_mixed == NULL ||
        reference_cut == NULL || actual == NULL || actual_second == NULL ||
        actual_third == NULL) {
        fprintf(stderr, "pooled-%s allocation failed\n", label);
        goto cleanup;
    }

    /* References from ordinary private-slab sessions. */
    if (kipp_model_open_backend(model_path, backend, &reference_model,
                                &error) != 0) {
        fprintf(stderr, "pooled-%s reference open failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    {
        uint32_t mixed[POOL_SEQ];
        uint32_t cut[POOL_SEQ];
        memcpy(mixed, sequence, 64 * sizeof(*mixed));
        memcpy(mixed + 64, alternate + 64, 32 * sizeof(*mixed));
        memcpy(cut, sequence, 70 * sizeof(*cut));
        memcpy(cut + 70, alternate + 70, 26 * sizeof(*cut));
        const uint32_t *inputs[3] = {sequence, mixed, cut};
        float *outputs[3] = {reference_full, reference_mixed, reference_cut};
        for (int variant = 0; variant < 3; ++variant) {
            if (kipp_session_create(reference_model, POOL_SEQ, &session,
                                    &error) != 0 ||
                kipp_session_eval(session, inputs[variant], POOL_SEQ,
                                  outputs[variant], KIPP_VOCAB_SIZE,
                                  &error) != 0) {
                fprintf(stderr, "pooled-%s reference eval failed: %s\n", label,
                        error.message);
                goto cleanup;
            }
            kipp_session_destroy(session);
            session = NULL;
        }
    }

    if (kipp_model_open_pooled(model_path, backend, 32, &pooled_model,
                               &error) != 0) {
        fprintf(stderr, "pooled-%s pooled open failed: %s\n", label, error.message);
        goto cleanup;
    }

    /* (a) Pooled identity: bitwise-equal to the private-slab session. */
    if (kipp_session_create(pooled_model, POOL_SEQ, &session, &error) != 0 ||
        kipp_session_eval(session, sequence, POOL_SEQ, actual,
                          KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "pooled-%s identity eval failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (memcmp(actual, reference_full,
               KIPP_VOCAB_SIZE * sizeof(*actual)) != 0) {
        fprintf(stderr, "POOLED-%s identity logits differ\n", label);
        goto cleanup;
    }
    if (backend != KIPP_BACKEND_CPU) {
        /* Cross-backend anchor: pooled GPU logits must also sit within the
         * standard tolerance of the CPU oracle. */
        kipp_model *oracle_model = NULL;
        kipp_session *oracle_session = NULL;
        float *oracle = malloc(KIPP_VOCAB_SIZE * sizeof(*oracle));
        int oracle_ok =
            oracle != NULL &&
            kipp_model_open_backend(model_path, KIPP_BACKEND_CPU,
                                    &oracle_model, &error) == 0 &&
            kipp_session_create(oracle_model, POOL_SEQ, &oracle_session,
                                &error) == 0 &&
            kipp_session_eval(oracle_session, sequence, POOL_SEQ, oracle,
                              KIPP_VOCAB_SIZE, &error) == 0;
        double oracle_nmse =
            oracle_ok ? logit_nmse(actual, oracle, KIPP_VOCAB_SIZE) : 1.0;
        int oracle_argmax = oracle_ok ? argmax(oracle, KIPP_VOCAB_SIZE) : -1;
        int pooled_argmax = argmax(actual, KIPP_VOCAB_SIZE);
        kipp_session_destroy(oracle_session);
        if (oracle_model != NULL) {
            (void)kipp_model_close(oracle_model, NULL);
        }
        free(oracle);
        fprintf(stderr,
                "POOLED-%s vs CPU oracle nmse=%.9g cpu_argmax=%d "
                "pooled_argmax=%d\n",
                label, oracle_nmse, oracle_argmax, pooled_argmax);
        if (!oracle_ok || oracle_nmse > 1.0e-4 ||
            oracle_argmax != pooled_argmax) {
            fprintf(stderr, "POOLED-%s diverges from the CPU oracle\n",
                    label);
            goto cleanup;
        }
    }

    /* (b) Serial sharing: destroy publishes; a new session adopts two full
     * blocks and reproduces the unshared logits bitwise. */
    kipp_session_destroy(session);
    session = NULL;
    if (kipp_session_create(pooled_model, POOL_SEQ, &session, &error) != 0 ||
        kipp_session_match_prefix(session, sequence, POOL_SEQ, &matched,
                                  &error) != 0) {
        fprintf(stderr, "pooled-%s prefix match failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (matched != 2 * KIPP_KV_BLOCK_TOKENS) {
        fprintf(stderr, "POOLED-%s expected 64 matched tokens, got %u\n", label,
                matched);
        goto cleanup;
    }
    if (kipp_session_eval(session, sequence + matched, POOL_SEQ - matched,
                          actual, KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "pooled-%s shared eval failed: %s\n", label, error.message);
        goto cleanup;
    }
    if (memcmp(actual, reference_full,
               KIPP_VOCAB_SIZE * sizeof(*actual)) != 0) {
        fprintf(stderr, "POOLED-%s shared-prefix logits differ\n", label);
        goto cleanup;
    }
    if (kipp_model_kv_pool_stats(pooled_model, &stats) != 0 ||
        stats.reused_blocks_total < 2) {
        fprintf(stderr, "POOLED-%s reuse counter did not grow\n", label);
        goto cleanup;
    }

    /* (e) Truncation into the private tail: roll the shared session back
     * mid-block and continue on a different suffix. */
    if (kipp_session_truncate(session, 70, &error) != 0 ||
        kipp_session_eval(session, alternate + 70, POOL_SEQ - 70, actual,
                          KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "pooled-%s truncate eval failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (memcmp(actual, reference_cut,
               KIPP_VOCAB_SIZE * sizeof(*actual)) != 0) {
        fprintf(stderr, "POOLED-%s truncated logits differ\n", label);
        goto cleanup;
    }
    kipp_session_destroy(session);
    session = NULL;

    /* (c) Batched mixed sharing: two adopters (one diverging) and one cold
     * session in a single kipp_eval_batch call. */
    if (kipp_session_create(pooled_model, POOL_SEQ, &session, &error) != 0 ||
        kipp_session_create(pooled_model, POOL_SEQ, &second, &error) != 0 ||
        kipp_session_create(pooled_model, POOL_SEQ, &third, &error) != 0 ||
        kipp_session_match_prefix(session, sequence, POOL_SEQ, &matched,
                                  &error) != 0 ||
        matched != 2 * KIPP_KV_BLOCK_TOKENS ||
        kipp_session_match_prefix(second, sequence, POOL_SEQ, &matched,
                                  &error) != 0 ||
        matched != 2 * KIPP_KV_BLOCK_TOKENS) {
        fprintf(stderr, "pooled-%s batch setup failed: %s\n", label, error.message);
        goto cleanup;
    }
    {
        kipp_batch_item batch[3] = {
            {session, sequence + 64, POOL_SEQ - 64, actual},
            {second, alternate + 64, POOL_SEQ - 64, actual_second},
            {third, sequence, POOL_SEQ, actual_third},
        };
        if (kipp_eval_batch(pooled_model, batch, 3, &error) != 0) {
            fprintf(stderr, "pooled-%s batch eval failed: %s\n", label,
                    error.message);
            goto cleanup;
        }
    }
    if (memcmp(actual, reference_full, KIPP_VOCAB_SIZE * sizeof(*actual)) !=
            0 ||
        memcmp(actual_second, reference_mixed,
               KIPP_VOCAB_SIZE * sizeof(*actual_second)) != 0 ||
        memcmp(actual_third, reference_full,
               KIPP_VOCAB_SIZE * sizeof(*actual_third)) != 0) {
        fprintf(stderr, "POOLED-%s batched logits differ\n", label);
        goto cleanup;
    }
    kipp_session_destroy(session);
    kipp_session_destroy(second);
    kipp_session_destroy(third);
    session = second = third = NULL;

    /* (d) Exhaustion: a 3-block pool refuses the fourth block cleanly and
     * the refused session's failure leaves its neighbor untouched. */
    if (kipp_model_open_pooled(model_path, backend, 3, &tiny_model,
                               &error) != 0 ||
        kipp_session_create(tiny_model, POOL_SEQ, &session, &error) != 0 ||
        kipp_session_create(tiny_model, POOL_SEQ, &second, &error) != 0) {
        fprintf(stderr, "pooled-%s tiny setup failed: %s\n", label, error.message);
        goto cleanup;
    }
    if (kipp_session_eval(session, sequence, 64, actual, KIPP_VOCAB_SIZE,
                          &error) != 0) {
        fprintf(stderr, "pooled-%s tiny eval failed: %s\n", label, error.message);
        goto cleanup;
    }
    if (kipp_session_eval(second, sequence, 64, actual_second,
                          KIPP_VOCAB_SIZE, &error) == 0 ||
        error.code != KIPP_ERROR_RANGE) {
        fprintf(stderr, "POOLED-%s exhaustion did not fail cleanly\n", label);
        goto cleanup;
    }
    if (kipp_session_eval(session, sequence + 64, 32, actual,
                          KIPP_VOCAB_SIZE, &error) != 0) {
        fprintf(stderr, "pooled-%s post-exhaustion eval failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (memcmp(actual, reference_full,
               KIPP_VOCAB_SIZE * sizeof(*actual)) != 0) {
        fprintf(stderr, "POOLED-%s neighbor corrupted by exhaustion\n", label);
        goto cleanup;
    }

    /* (f) Eviction: publish, then let new content reclaim the blocks; the
     * old prefix must simply stop matching (never alias). */
    kipp_session_destroy(session);
    kipp_session_destroy(second);
    session = second = NULL;
    if (kipp_session_create(tiny_model, POOL_SEQ, &session, &error) != 0 ||
        kipp_session_eval(session, alternate, 96, actual, KIPP_VOCAB_SIZE,
                          &error) != 0) {
        fprintf(stderr, "pooled-%s eviction fill failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    kipp_session_destroy(session);
    session = NULL;
    if (kipp_session_create(tiny_model, POOL_SEQ, &session, &error) != 0 ||
        kipp_session_match_prefix(session, sequence, POOL_SEQ, &matched,
                                  &error) != 0) {
        fprintf(stderr, "pooled-%s post-eviction match failed: %s\n", label,
                error.message);
        goto cleanup;
    }
    if (matched != 0) {
        fprintf(stderr, "POOLED-%s evicted prefix still matched %u\n", label,
                matched);
        goto cleanup;
    }

    fprintf(stderr,
            "POOLED-%s identity, sharing, batching, exhaustion, truncation, "
            "and eviction all bitwise-correct over %u-token sequences\n",
            label, (uint32_t)POOL_SEQ);
    result = 0;

cleanup:
    kipp_session_destroy(session);
    kipp_session_destroy(second);
    kipp_session_destroy(third);
    if (pooled_model != NULL) {
        (void)kipp_model_close(pooled_model, NULL);
    }
    if (tiny_model != NULL) {
        (void)kipp_model_close(tiny_model, NULL);
    }
    if (reference_model != NULL) {
        (void)kipp_model_close(reference_model, NULL);
    }
    free(tokens_path);
    free(seed);
    free(reference_full);
    free(reference_mixed);
    free(reference_cut);
    free(actual);
    free(actual_second);
    free(actual_third);
    return result;
}

/*
 * The conventional tolerance gate, run on the same multi-block sequence the
 * scramble gate uses: evaluate the synthesized 64-token prompt through an
 * identity-mapped session and compare its final logits against the stateless
 * forward pass (which never touches the KV cache) with the Phase-2 NMSE
 * bound and an argmax check. In an unfaulted build the two are bitwise-equal
 * (NMSE 0). This is the mutation study's model of "the standard reference
 * test": identity-mapped, tolerance-based, argmax-checked.
 */
static int run_fault_reference_test(const char *model_path,
                                    const char *vector_directory) {
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *seed = NULL;
    uint32_t *tokens = NULL;
    size_t token_bytes = 0;
    kipp_model *model = NULL;
    kipp_session *session = NULL;
    float *cached = NULL;
    float *reference = NULL;
    kipp_error error = {0};
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&seed, &token_bytes) != 0 ||
        token_bytes < sizeof(*seed) || token_bytes % sizeof(*seed)) {
        fprintf(stderr, "unable to read fault-reference token vector\n");
        goto cleanup;
    }
    size_t seed_count = token_bytes / sizeof(*seed);
    uint32_t n = 3 * KIPP_KV_BLOCK_TOKENS; /* match the scramble gate */
    tokens = malloc((size_t)n * sizeof(*tokens));
    cached = malloc(KIPP_VOCAB_SIZE * sizeof(*cached));
    reference = malloc(KIPP_VOCAB_SIZE * sizeof(*reference));
    if (tokens == NULL || cached == NULL || reference == NULL) {
        fprintf(stderr, "fault-reference allocation failed\n");
        goto cleanup;
    }
    for (uint32_t index = 0; index < n; ++index) {
        tokens[index] = seed[index % seed_count];
    }
    if (kipp_model_open_backend(model_path, KIPP_BACKEND_CPU, &model,
                                &error) != 0 ||
        kipp_session_create(model, n, &session, &error) != 0 ||
        kipp_session_eval(session, tokens, n, cached, KIPP_VOCAB_SIZE,
                          &error) != 0 ||
        kipp_model_eval(model, tokens, n, reference, KIPP_VOCAB_SIZE,
                        &error) != 0) {
        fprintf(stderr, "fault-reference eval failed: %s\n", error.message);
        goto cleanup;
    }
    double nmse = logit_nmse(cached, reference, KIPP_VOCAB_SIZE);
    int reference_argmax = argmax(reference, KIPP_VOCAB_SIZE);
    int cached_argmax = argmax(cached, KIPP_VOCAB_SIZE);
    fprintf(stderr,
            "FAULTREF tokens=%u nmse=%.9g reference_argmax=%d "
            "cached_argmax=%d\n",
            n, nmse, reference_argmax, cached_argmax);
    result = nmse <= 1.0e-6 && cached_argmax == reference_argmax ? 0 : -1;

cleanup:
    kipp_session_destroy(session);
    if (model != NULL) {
        (void)kipp_model_close(model, NULL);
    }
    free(tokens_path);
    free(seed);
    free(tokens);
    free(cached);
    free(reference);
    return result;
}

#ifdef KIPP_ENABLE_METAL
/* Metal kipp_session_eval_n rows must match a CPU single-position eval at
 * each position within the Metal-vs-CPU tolerance (1e-4 NMSE, same argmax). */
static int run_multilogit_metal_test(const char *model_path,
                                     const char *vector_directory) {
    char *tokens_path = join_path(vector_directory, "tokens.u32");
    uint32_t *tokens = NULL;
    size_t token_bytes = 0;
    size_t token_count;
    kipp_model *metal_model = NULL;
    kipp_model *cpu_model = NULL;
    kipp_session *metal_session = NULL;
    kipp_session *cpu_session = NULL;
    float *rows = NULL;
    float *reference = NULL;
    kipp_error error = {0};
    int result = -1;

    if (tokens_path == NULL ||
        read_file(tokens_path, (void **)&tokens, &token_bytes) != 0 ||
        token_bytes < 2 * sizeof(*tokens) || token_bytes % sizeof(*tokens)) {
        fprintf(stderr, "unable to read multi-logit token vector\n");
        goto cleanup;
    }
    token_count = token_bytes / sizeof(*tokens);
    uint32_t n = (uint32_t)token_count;
    rows = malloc((size_t)n * KIPP_VOCAB_SIZE * sizeof(*rows));
    reference = malloc(KIPP_VOCAB_SIZE * sizeof(*reference));
    if (rows == NULL || reference == NULL ||
        kipp_model_open_backend(model_path, KIPP_BACKEND_METAL, &metal_model,
                                &error) != 0 ||
        kipp_model_open_backend(model_path, KIPP_BACKEND_CPU, &cpu_model,
                                &error) != 0) {
        fprintf(stderr, "multi-logit metal setup failed: %s\n", error.message);
        goto cleanup;
    }
    if (kipp_session_create(metal_model, (uint32_t)token_count, &metal_session,
                            &error) != 0 ||
        kipp_session_eval_n(metal_session, tokens, token_count, rows, n,
                            &error) != 0 ||
        kipp_session_create(cpu_model, (uint32_t)token_count, &cpu_session,
                            &error) != 0) {
        fprintf(stderr, "multi-logit metal eval failed: %s\n", error.message);
        goto cleanup;
    }
    for (uint32_t position = 0; position < n; ++position) {
        if (kipp_session_eval(cpu_session, &tokens[position], 1, reference,
                              KIPP_VOCAB_SIZE, &error) != 0) {
            goto cleanup;
        }
        char label[40];
        (void)snprintf(label, sizeof(label), "multilogit-row-%u", position);
        if (phase2_compare(label, rows + (size_t)position * KIPP_VOCAB_SIZE,
                           reference, 1.0e-4) != 0) {
            goto cleanup;
        }
    }
    fprintf(stderr, "MULTILOGIT-METAL %u rows match CPU within 1e-4\n", n);
    result = 0;

cleanup:
    kipp_session_destroy(cpu_session);
    kipp_session_destroy(metal_session);
    if (cpu_model != NULL) {
        (void)kipp_model_close(cpu_model, NULL);
    }
    if (metal_model != NULL) {
        (void)kipp_model_close(metal_model, NULL);
    }
    free(tokens_path);
    free(tokens);
    free(rows);
    free(reference);
    return result;
}
#endif

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
    kipp_model_info model_info;
    if (kipp_model_get_info(model, &model_info) != 0) {
        fprintf(stderr, "Phase 2 model info failed\n");
        goto cleanup;
    }

    if (kipp_session_create(model, 0, &invalid, &error) == 0 ||
        error.code != KIPP_ERROR_RANGE || invalid != NULL ||
        kipp_session_create(model, model_info.context_length + 1, &invalid,
                            &error) == 0 ||
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
        info.cache_bytes != kipp_test_kv_cache_bytes(
                                model_info.block_count,
                                (uint32_t)token_count)) {
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
            (unsigned long long)kipp_test_kv_cache_bytes(
                model_info.block_count, (uint32_t)token_count),
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

/* Fill a 32-token block with a distinct, deterministic pattern. */
static void kv_fill_block(uint32_t *tokens, uint32_t base) {
    for (uint32_t index = 0; index < KIPP_KV_BLOCK_TOKENS; ++index) {
        tokens[index] = base + index;
    }
}

static void test_kv_pool_reuse(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(8);
    CHECK(pool != NULL);
    CHECK(kipp_kv_pool_free_count(pool) == 8);
    uint32_t a[KIPP_KV_BLOCK_TOKENS];
    uint32_t b[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(a, 100);
    kv_fill_block(b, 900);
    bool reused = true;
    uint32_t first = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    CHECK(first != KIPP_KV_INVALID_BLOCK);
    CHECK(reused == false);
    CHECK(kipp_kv_pool_free_count(pool) == 7);
    /* Identical tokens hit the cached block and bump its ref count. */
    uint32_t again = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    CHECK(again == first);
    CHECK(reused == true);
    CHECK(kipp_kv_pool_free_count(pool) == 7);
    /* Different tokens (verify-by-tokens) never alias, even on hash bucket
     * overlap. */
    uint32_t other = kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    CHECK(other != first);
    CHECK(reused == false);
    CHECK(kipp_kv_pool_free_count(pool) == 6);
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_refcount(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(4);
    uint32_t tokens[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(tokens, 10);
    bool reused;
    uint32_t id = kipp_kv_pool_acquire(pool, 0, tokens, 32, &reused);
    kipp_kv_pool_acquire(pool, 0, tokens, 32, &reused); /* ref == 2 */
    CHECK(reused == true);
    CHECK(kipp_kv_pool_free_count(pool) == 3);
    kipp_kv_pool_release(pool, id); /* ref == 1, still held */
    CHECK(kipp_kv_pool_free_count(pool) == 3);
    kipp_kv_pool_release(pool, id); /* ref == 0, now evictable */
    CHECK(kipp_kv_pool_free_count(pool) == 4);
    /* Over-release is a no-op, not an underflow. */
    kipp_kv_pool_release(pool, id);
    CHECK(kipp_kv_pool_free_count(pool) == 4);
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_revive(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(4);
    uint32_t tokens[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(tokens, 42);
    bool reused;
    uint32_t id = kipp_kv_pool_acquire(pool, 0, tokens, 32, &reused);
    kipp_kv_pool_release(pool, id); /* evictable but still indexed */
    CHECK(kipp_kv_pool_free_count(pool) == 4);
    /* A matching acquire revives the freed block rather than consuming a new
     * one. */
    uint32_t revived = kipp_kv_pool_acquire(pool, 0, tokens, 32, &reused);
    CHECK(revived == id);
    CHECK(reused == true);
    CHECK(kipp_kv_pool_free_count(pool) == 3);
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_evict(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(2);
    uint32_t a[KIPP_KV_BLOCK_TOKENS];
    uint32_t b[KIPP_KV_BLOCK_TOKENS];
    uint32_t c[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(a, 100);
    kv_fill_block(b, 200);
    kv_fill_block(c, 300);
    bool reused;
    uint32_t id_a = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    kipp_kv_pool_release(pool, id_a); /* released first -> LRU */
    uint32_t id_b = kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    kipp_kv_pool_release(pool, id_b);
    CHECK(kipp_kv_pool_free_count(pool) == 2);
    /* Fresh content under pressure evicts the least-recently-used block (A). */
    uint32_t id_c = kipp_kv_pool_acquire(pool, 0, c, 32, &reused);
    CHECK(reused == false);
    CHECK(id_c == id_a);
    /* B was not the LRU victim, so it is still revivable. */
    uint32_t revive_b = kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    CHECK(reused == true);
    CHECK(revive_b == id_b);
    /* A was evicted: its content is gone from the index. */
    kipp_kv_pool_release(pool, id_c);
    kipp_kv_pool_release(pool, revive_b);
    uint32_t reacquire_a = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    CHECK(reused == false);
    (void)reacquire_a;
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_partial_private(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(4);
    uint32_t tokens[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(tokens, 7);
    bool reused;
    /* Partial (< 32 token) blocks are never shared or revived. */
    uint32_t first = kipp_kv_pool_acquire(pool, 0, tokens, 20, &reused);
    CHECK(reused == false);
    uint32_t second = kipp_kv_pool_acquire(pool, 0, tokens, 20, &reused);
    CHECK(reused == false);
    CHECK(first != second);
    CHECK(kipp_kv_pool_free_count(pool) == 2);
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_prefix_match(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(16);
    uint32_t sequence[4 * KIPP_KV_BLOCK_TOKENS];
    for (uint32_t index = 0; index < 4 * KIPP_KV_BLOCK_TOKENS; ++index) {
        sequence[index] = index + 1;
    }
    /* Populate three chained full blocks and keep them referenced. */
    uint64_t parent = 0;
    uint32_t built[3];
    for (uint32_t block = 0; block < 3; ++block) {
        const uint32_t *block_tokens = sequence + block * KIPP_KV_BLOCK_TOKENS;
        bool reused;
        built[block] =
            kipp_kv_pool_acquire(pool, parent, block_tokens, 32, &reused);
        parent = kipp_kv_pool_hash(parent, block_tokens, 32);
    }

    uint32_t matched_blocks[4];
    uint32_t matched_tokens = 0;
    uint64_t out_parent = 0;
    /* With one extra token beyond the 3 cached blocks, all three match. */
    uint32_t matched = kipp_kv_pool_prefix_match(
        pool, sequence, 3 * KIPP_KV_BLOCK_TOKENS + 1, matched_blocks, 4,
        &matched_tokens, &out_parent);
    CHECK(matched == 3);
    CHECK(matched_tokens == 96);
    CHECK(matched_blocks[0] == built[0]);
    CHECK(matched_blocks[2] == built[2]);
    for (uint32_t block = 0; block < matched; ++block) {
        kipp_kv_pool_release(pool, matched_blocks[block]);
    }

    /* An exact multiple must leave a block short (the logits-producing cap). */
    matched = kipp_kv_pool_prefix_match(pool, sequence,
                                        3 * KIPP_KV_BLOCK_TOKENS, matched_blocks,
                                        4, &matched_tokens, &out_parent);
    CHECK(matched == 2);
    CHECK(matched_tokens == 64);
    for (uint32_t block = 0; block < matched; ++block) {
        kipp_kv_pool_release(pool, matched_blocks[block]);
    }

    /* A sequence shorter than one block matches nothing. */
    matched = kipp_kv_pool_prefix_match(pool, sequence, 32, matched_blocks, 4,
                                        &matched_tokens, &out_parent);
    CHECK(matched == 0);
    CHECK(matched_tokens == 0);

    /* A divergent first block short-circuits the walk. */
    uint32_t divergent[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(divergent, 5000);
    matched = kipp_kv_pool_prefix_match(pool, divergent, 33, matched_blocks, 4,
                                        &matched_tokens, &out_parent);
    CHECK(matched == 0);

    for (uint32_t block = 0; block < 3; ++block) {
        kipp_kv_pool_release(pool, built[block]);
    }
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_exhaustion(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(1);
    uint32_t a[KIPP_KV_BLOCK_TOKENS];
    uint32_t b[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(a, 1);
    kv_fill_block(b, 2);
    bool reused;
    uint32_t held = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    CHECK(held != KIPP_KV_INVALID_BLOCK);
    CHECK(kipp_kv_pool_free_count(pool) == 0);
    /* Nothing is evictable while the only block is referenced. */
    uint32_t denied = kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    CHECK(denied == KIPP_KV_INVALID_BLOCK);
    /* Freeing it restores capacity. */
    kipp_kv_pool_release(pool, held);
    uint32_t ok = kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    CHECK(ok != KIPP_KV_INVALID_BLOCK);
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_alloc_seal(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(4);
    uint32_t a[KIPP_KV_BLOCK_TOKENS];
    uint32_t ids[4];
    uint32_t matched_tokens = 0;
    uint64_t parent = 0;
    kv_fill_block(a, 7);

    /* alloc claims a private, unindexed block; nothing matches it yet. */
    uint32_t priv = kipp_kv_pool_alloc(pool);
    CHECK(priv != KIPP_KV_INVALID_BLOCK);
    CHECK(kipp_kv_pool_free_count(pool) == 3);
    CHECK(kipp_kv_pool_prefix_match(pool, a, 33, ids, 4, &matched_tokens,
                                    &parent) == 0);

    /* seal publishes it; a 33-token prefix now matches the one block. */
    CHECK(kipp_kv_pool_seal(pool, priv, 0, a) == 0);
    CHECK(kipp_kv_pool_prefix_match(pool, a, 33, ids, 4, &matched_tokens,
                                    &parent) == 1);
    CHECK(ids[0] == priv && matched_tokens == KIPP_KV_BLOCK_TOKENS);
    kipp_kv_pool_release(pool, ids[0]);

    /* Double seal is rejected; sealing a second block with identical content
     * leaves the index holding one copy (the original still matches). */
    CHECK(kipp_kv_pool_seal(pool, priv, 0, a) == -1);
    uint32_t dup = kipp_kv_pool_alloc(pool);
    CHECK(dup != KIPP_KV_INVALID_BLOCK);
    CHECK(kipp_kv_pool_seal(pool, dup, 0, a) == 0);
    CHECK(kipp_kv_pool_prefix_match(pool, a, 33, ids, 4, &matched_tokens,
                                    &parent) == 1);
    CHECK(ids[0] == priv);
    kipp_kv_pool_release(pool, ids[0]);

    /* Sealing an unreferenced block is rejected. */
    kipp_kv_pool_release(pool, dup);
    CHECK(kipp_kv_pool_seal(pool, dup, 0, a) == -1);
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_stats(void) {
    kipp_kv_pool *pool = kipp_kv_pool_create(2);
    kipp_kv_pool_stats stats;
    uint32_t a[KIPP_KV_BLOCK_TOKENS];
    uint32_t b[KIPP_KV_BLOCK_TOKENS];
    bool reused;
    kv_fill_block(a, 1);
    kv_fill_block(b, 2);

    kipp_kv_pool_get_stats(pool, &stats);
    CHECK(stats.total_blocks == 2 && stats.free_blocks == 2 &&
          stats.reused_blocks_total == 0 && stats.evicted_blocks_total == 0);

    uint32_t first = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    uint32_t again = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    CHECK(first == again && reused);
    kipp_kv_pool_get_stats(pool, &stats);
    CHECK(stats.free_blocks == 1 && stats.reused_blocks_total == 1);

    /* Release both refs, fill the pool with new content twice: the second
     * pass must evict the stale indexed blocks and count it. */
    kipp_kv_pool_release(pool, first);
    kipp_kv_pool_release(pool, again);
    uint32_t other = kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    CHECK(other != KIPP_KV_INVALID_BLOCK && !reused);
    kipp_kv_pool_release(pool, other);
    kv_fill_block(a, 3);
    kv_fill_block(b, 4);
    (void)kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    (void)kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    kipp_kv_pool_get_stats(pool, &stats);
    CHECK(stats.free_blocks == 0 && stats.evicted_blocks_total == 2);
    kipp_kv_pool_destroy(pool);
}

static void test_kv_pool_collision(void) {
    /* With every hash forced equal, token-unequal blocks must never alias:
     * the memcmp verify is what makes content addressing collision-proof. */
    kipp_kv_pool *pool = kipp_kv_pool_create(4);
    kipp_kv_pool_test_force_hash(pool, true);
    uint32_t a[KIPP_KV_BLOCK_TOKENS];
    uint32_t b[KIPP_KV_BLOCK_TOKENS];
    uint32_t ids[4];
    uint32_t matched_tokens = 0;
    bool reused;
    kv_fill_block(a, 1);
    kv_fill_block(b, 2);

    uint32_t first = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    uint32_t second = kipp_kv_pool_acquire(pool, 0, b, 32, &reused);
    CHECK(first != second && !reused);

    /* Token-equal still reuses despite the degenerate hash... */
    uint32_t again = kipp_kv_pool_acquire(pool, 0, a, 32, &reused);
    CHECK(again == first && reused);
    /* ...and prefix matching returns the right block, not a hash sibling. */
    CHECK(kipp_kv_pool_prefix_match(pool, b, 33, ids, 4, &matched_tokens,
                                    NULL) == 1);
    CHECK(ids[0] == second);
    kipp_kv_pool_release(pool, ids[0]);

    /* Sealed blocks obey the same rule. */
    uint32_t priv = kipp_kv_pool_alloc(pool);
    uint32_t c[KIPP_KV_BLOCK_TOKENS];
    kv_fill_block(c, 3);
    CHECK(kipp_kv_pool_seal(pool, priv, 0, c) == 0);
    CHECK(kipp_kv_pool_prefix_match(pool, c, 33, ids, 4, &matched_tokens,
                                    NULL) == 1);
    CHECK(ids[0] == priv);
    kipp_kv_pool_release(pool, ids[0]);
    kipp_kv_pool_destroy(pool);
}

int main(int argc, char **argv) {
    RUN(test_error_names);
    RUN(test_bf16);
    RUN(test_checked_arithmetic);
    RUN(test_kv_layout);
    RUN(test_rms_norm);
    RUN(test_matvec);
    RUN(test_quant_matvec);
    RUN(test_rope);
    RUN(test_softmax_and_swiglu);
    RUN(test_causal_gqa);
    RUN(test_pretokenizer);
    RUN(test_nfc_normalization);
    RUN(test_pretokenizer_fuzz);
    RUN(test_nfc_fuzz);
    RUN(test_gguf_reject_fuzz);
    RUN(test_json_parse_fuzz);
    RUN(test_http_header_fuzz);
    RUN(test_sampler);
    RUN(test_sample_ex);
    RUN(test_public_argument_checks);
    RUN(test_registry_rejection);
    RUN(test_prompt_lookup);
    RUN(test_spec_gate);
    RUN(test_chat_render);
    RUN(test_kv_pool_reuse);
    RUN(test_kv_pool_refcount);
    RUN(test_kv_pool_revive);
    RUN(test_kv_pool_evict);
    RUN(test_kv_pool_partial_private);
    RUN(test_kv_pool_prefix_match);
    RUN(test_kv_pool_exhaustion);
    RUN(test_kv_pool_alloc_seal);
    RUN(test_kv_pool_stats);
    RUN(test_kv_pool_collision);
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
    } else if (argc == 4 && strcmp(argv[1], "--multilogit") == 0) {
        ++tests_run;
        if (run_multilogit_test(argv[2], argv[3]) != 0) {
            ++failures;
            fprintf(stderr, "FAIL multilogit\n");
        } else {
            fprintf(stderr, "PASS multilogit\n");
        }
    } else if (argc == 4 && strcmp(argv[1], "--paged-cpu") == 0) {
        ++tests_run;
        if (run_paged_test(argv[2], argv[3], KIPP_BACKEND_CPU, "CPU") != 0) {
            ++failures;
            fprintf(stderr, "FAIL paged_cpu\n");
        } else {
            fprintf(stderr, "PASS paged_cpu\n");
        }
    } else if (argc == 4 && strcmp(argv[1], "--fault-reference") == 0) {
        ++tests_run;
        if (run_fault_reference_test(argv[2], argv[3]) != 0) {
            ++failures;
            fprintf(stderr, "FAIL fault_reference\n");
        } else {
            fprintf(stderr, "PASS fault_reference\n");
        }
    } else if (argc == 4 && strcmp(argv[1], "--pooled-cpu") == 0) {
        ++tests_run;
        if (run_pooled_test(argv[2], argv[3], KIPP_BACKEND_CPU, "CPU") != 0) {
            ++failures;
            fprintf(stderr, "FAIL pooled_cpu\n");
        } else {
            fprintf(stderr, "PASS pooled_cpu\n");
        }
#ifdef KIPP_ENABLE_METAL
    } else if (argc == 4 && strcmp(argv[1], "--pooled-metal") == 0) {
        ++tests_run;
        if (run_pooled_test(argv[2], argv[3], KIPP_BACKEND_METAL, "METAL") !=
            0) {
            ++failures;
            fprintf(stderr, "FAIL pooled_metal\n");
        } else {
            fprintf(stderr, "PASS pooled_metal\n");
        }
#endif
#ifdef KIPP_ENABLE_METAL
    } else if (argc == 4 && strcmp(argv[1], "--paged-metal") == 0) {
        ++tests_run;
        if (run_paged_test(argv[2], argv[3], KIPP_BACKEND_METAL, "METAL") != 0) {
            ++failures;
            fprintf(stderr, "FAIL paged_metal\n");
        } else {
            fprintf(stderr, "PASS paged_metal\n");
        }
#endif
#ifdef KIPP_ENABLE_METAL
    } else if (argc == 4 && strcmp(argv[1], "--multilogit-metal") == 0) {
        ++tests_run;
        if (run_multilogit_metal_test(argv[2], argv[3]) != 0) {
            ++failures;
            fprintf(stderr, "FAIL multilogit_metal\n");
        } else {
            fprintf(stderr, "PASS multilogit_metal\n");
        }
#endif
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
                "[--model|--phase2-model|--phase3-metal|--phase4-cuda|"
                "--multilogit|--paged-cpu "
                "MODEL.gguf "
                "VECTOR_DIRECTORY]\n",
                argv[0]);
        return 2;
    }
    fprintf(stderr, "%d test%s, %d failure%s\n", tests_run,
            tests_run == 1 ? "" : "s", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
