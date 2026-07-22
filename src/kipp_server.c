/*
 * Kipp API server: the deliberately small OpenAI Completions subset fixed
 * in docs/ARCHITECTURE.md.
 *
 *   GET  /healthz
 *   GET  /v1/models
 *   POST /v1/completions
 *
 * A single-threaded poll() loop serves concurrent loopback connections.
 * Admitted requests generate together: every scheduler step batches one
 * chunk of prefill or one decode token per active choice through
 * kipp_eval_batch, so the weights are read once per step across requests.
 * Unsupported request fields receive a clear client error instead of being
 * ignored. HTTP parsing, JSON handling, scheduling, and model execution
 * stay in separate layers, and the server never touches backend buffers or
 * KV memory directly.
 */

#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "kipp.h"
#include "kipp_chat.h"
#include "kipp_http.h"
#include "kipp_json.h"
#include "kipp_kv_pool.h" /* KIPP_KV_BLOCK_TOKENS for pool sizing */

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SERVER_HEADER_LIMIT (64u * 1024u)
#define SERVER_BODY_LIMIT (8u * 1024u * 1024u)
#define SERVER_DEFAULT_PORT 8080u
#define SERVER_DEFAULT_MAX_TOKENS 16u
#define SERVER_STOP_LIMIT 4u
#define SERVER_CHOICE_LIMIT 8u
#define SERVER_MAX_CONNECTIONS 64u
/*
 * Concurrent in-flight generations. Matched to KIPP_EVAL_BATCH_LIMIT so a
 * full batch of n=1 requests can decode together; total *choices* across
 * generations are still admission-capped at the batch limit, so the
 * scheduler's per-step item array can never overflow.
 */
#define SERVER_MAX_GENERATIONS 32u
#define SERVER_PREFILL_CHUNK 32u
#define SERVER_OUTPUT_LIMIT (4u * 1024u * 1024u)

static volatile sig_atomic_t server_running = 1;

/* Identity and limits of the one loaded model, set once before serving. */
static kipp_model_info server_info;

/* Process-lifetime serving counters, exposed at GET /metrics. */
static struct {
    uint64_t requests_total;
    uint64_t requests_failed_total;
    uint64_t prompt_tokens_total;
    uint64_t generation_tokens_total;
    uint64_t prefix_tokens_reused_total;
} server_metrics;

/*
 * Pool-backed cross-request prefix sharing (CPU and Metal backends). When
 * on, every choice gets a pooled session that adopts the longest published
 * prefix of its prompt at admission, and admission reserves worst-case pool
 * blocks so an admitted request can never hit exhaustion mid-generation.
 * The legacy single-slot session cache serves only non-pooled models.
 */
static bool server_pooled;
static kipp_model *server_model; /* for pool stats at GET /metrics */

static void handle_stop_signal(int signal_number) {
    (void)signal_number;
    server_running = 0;
}

/* ---------------------------------------------------------------- strings */

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    bool failed;
} string_builder;

static void sb_free(string_builder *builder) {
    free(builder->data);
    memset(builder, 0, sizeof(*builder));
}

static void sb_append_bytes(string_builder *builder, const char *bytes,
                            size_t count) {
    if (builder->failed) {
        return;
    }
    if (builder->length + count + 1 > builder->capacity) {
        size_t capacity =
            builder->capacity == 0 ? 256 : builder->capacity;
        while (capacity < builder->length + count + 1) {
            if (capacity > SIZE_MAX / 2) {
                builder->failed = true;
                return;
            }
            capacity *= 2;
        }
        char *resized = realloc(builder->data, capacity);
        if (resized == NULL) {
            builder->failed = true;
            return;
        }
        builder->data = resized;
        builder->capacity = capacity;
    }
    memcpy(builder->data + builder->length, bytes, count);
    builder->length += count;
    builder->data[builder->length] = '\0';
}

static void sb_append(string_builder *builder, const char *text) {
    sb_append_bytes(builder, text, strlen(text));
}

static void sb_append_format(string_builder *builder, const char *format,
                             ...) {
    char buffer[512];
    va_list arguments;
    va_start(arguments, format);
    int written = vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    if (written < 0 || (size_t)written >= sizeof(buffer)) {
        builder->failed = true;
        return;
    }
    sb_append_bytes(builder, buffer, (size_t)written);
}

/* Append text as a JSON string literal, including the quotes. */
static void sb_append_json_string(string_builder *builder, const char *text,
                                  size_t length) {
    sb_append(builder, "\"");
    for (size_t index = 0; index < length; ++index) {
        unsigned char byte = (unsigned char)text[index];
        switch (byte) {
        case '"':
            sb_append(builder, "\\\"");
            break;
        case '\\':
            sb_append(builder, "\\\\");
            break;
        case '\b':
            sb_append(builder, "\\b");
            break;
        case '\f':
            sb_append(builder, "\\f");
            break;
        case '\n':
            sb_append(builder, "\\n");
            break;
        case '\r':
            sb_append(builder, "\\r");
            break;
        case '\t':
            sb_append(builder, "\\t");
            break;
        default:
            if (byte < 0x20) {
                sb_append_format(builder, "\\u%04x", byte);
            } else {
                sb_append_bytes(builder, (const char *)&text[index], 1);
            }
            break;
        }
    }
    sb_append(builder, "\"");
}

/* ------------------------------------------------------------ completions */

#define SERVER_LOGIT_BIAS_LIMIT 64u

typedef struct {
    char *prompt;
    uint32_t max_tokens;
    uint32_t choice_count;
    float temperature;
    float top_p;
    uint32_t top_k;
    float min_p;
    float frequency_penalty;
    float presence_penalty;
    float repetition_penalty;
    uint32_t bias_tokens[SERVER_LOGIT_BIAS_LIMIT];
    float bias_values[SERVER_LOGIT_BIAS_LIMIT];
    size_t bias_count;
    uint64_t seed;
    bool stream;
    bool is_chat; /* chat completions: response uses message/delta shape */
    bool logprobs_enabled;
    uint32_t logprobs_top; /* number of top alternatives per token to report */
    bool include_usage;    /* stream_options.include_usage */
    char *stops[SERVER_STOP_LIMIT];
    size_t stop_count;
} completion_request;

static void completion_request_free(completion_request *request) {
    free(request->prompt);
    for (size_t index = 0; index < request->stop_count; ++index) {
        free(request->stops[index]);
    }
    memset(request, 0, sizeof(*request));
}

static bool json_number_as_u32(const kipp_json_value *value, uint32_t minimum,
                               uint32_t maximum, uint32_t *output) {
    if (value->type != KIPP_JSON_NUMBER || value->number != floor(value->number) ||
        value->number < (double)minimum || value->number > (double)maximum) {
        return false;
    }
    *output = (uint32_t)value->number;
    return true;
}

/*
 * Parse the sampling fields shared by text and chat completions. Returns 1
 * if the key was a recognized sampling field (handled), 0 if not this
 * parser's key (caller keeps checking), -1 on a validation error (message
 * written). Keeps both endpoints' sampling surface identical.
 */
static int parse_sampling_field(const char *key, const kipp_json_value *value,
                                completion_request *request, char *message,
                                size_t message_size) {
    if (strcmp(key, "top_k") == 0) {
        uint32_t top_k;
        if (!json_number_as_u32(value, 0, KIPP_VOCAB_SIZE, &top_k)) {
            (void)snprintf(message, message_size,
                           "top_k must be an integer between 0 and %u",
                           KIPP_VOCAB_SIZE);
            return -1;
        }
        request->top_k = top_k;
        return 1;
    }
    if (strcmp(key, "min_p") == 0) {
        if (value->type != KIPP_JSON_NUMBER || value->number < 0.0 ||
            value->number > 1.0) {
            (void)snprintf(message, message_size,
                           "min_p must be a number between 0 and 1");
            return -1;
        }
        request->min_p = (float)value->number;
        return 1;
    }
    if (strcmp(key, "frequency_penalty") == 0 ||
        strcmp(key, "presence_penalty") == 0) {
        if (value->type != KIPP_JSON_NUMBER || value->number < -2.0 ||
            value->number > 2.0) {
            (void)snprintf(message, message_size,
                           "%s must be a number between -2 and 2", key);
            return -1;
        }
        if (key[0] == 'f') {
            request->frequency_penalty = (float)value->number;
        } else {
            request->presence_penalty = (float)value->number;
        }
        return 1;
    }
    if (strcmp(key, "repetition_penalty") == 0) {
        if (value->type != KIPP_JSON_NUMBER || value->number <= 0.0 ||
            value->number > 4.0) {
            (void)snprintf(message, message_size,
                           "repetition_penalty must be greater than 0 and at "
                           "most 4");
            return -1;
        }
        request->repetition_penalty = (float)value->number;
        return 1;
    }
    if (strcmp(key, "logit_bias") == 0) {
        if (value->type != KIPP_JSON_OBJECT) {
            (void)snprintf(message, message_size,
                           "logit_bias must be an object of token biases");
            return -1;
        }
        if (value->count > SERVER_LOGIT_BIAS_LIMIT) {
            (void)snprintf(message, message_size,
                           "logit_bias accepts at most %u entries",
                           SERVER_LOGIT_BIAS_LIMIT);
            return -1;
        }
        for (size_t index = 0; index < value->count; ++index) {
            char *end = NULL;
            errno = 0;
            unsigned long token = strtoul(value->keys[index], &end, 10);
            if (errno != 0 || end == value->keys[index] || *end != '\0' ||
                token >= KIPP_VOCAB_SIZE) {
                (void)snprintf(message, message_size,
                               "logit_bias keys must be token ids in [0, %u)",
                               KIPP_VOCAB_SIZE);
                return -1;
            }
            const kipp_json_value *bias = &value->items[index];
            if (bias->type != KIPP_JSON_NUMBER || bias->number < -100.0 ||
                bias->number > 100.0) {
                (void)snprintf(message, message_size,
                               "logit_bias values must be between -100 and 100");
                return -1;
            }
            request->bias_tokens[request->bias_count] = (uint32_t)token;
            request->bias_values[request->bias_count] = (float)bias->number;
            ++request->bias_count;
        }
        return 1;
    }
    return 0;
}

/* Parse the shared stream_options object. Returns 0 or -1 (message written). */
static int parse_stream_options(const kipp_json_value *value,
                                completion_request *request, char *message,
                                size_t message_size) {
    if (value->type != KIPP_JSON_OBJECT) {
        (void)snprintf(message, message_size,
                       "stream_options must be an object");
        return -1;
    }
    for (size_t index = 0; index < value->count; ++index) {
        if (strcmp(value->keys[index], "include_usage") == 0) {
            if (value->items[index].type != KIPP_JSON_BOOL) {
                (void)snprintf(message, message_size,
                               "include_usage must be a boolean");
                return -1;
            }
            request->include_usage = value->items[index].boolean;
        } else {
            (void)snprintf(message, message_size,
                           "unsupported stream_options field \"%s\"",
                           value->keys[index]);
            return -1;
        }
    }
    return 0;
}

/*
 * Validate the request body against the supported field set. Returns 0 on
 * success; on failure writes a client-facing message and returns -1.
 */
static int parse_completion_request(const kipp_json_value *body,
                                    completion_request *request,
                                    char *message, size_t message_size) {
    memset(request, 0, sizeof(*request));
    request->max_tokens = SERVER_DEFAULT_MAX_TOKENS;
    request->choice_count = 1;
    request->temperature = 1.0f;
    request->top_p = 1.0f;
    request->repetition_penalty = 1.0f;
    request->seed = 0;

    if (body->type != KIPP_JSON_OBJECT) {
        (void)snprintf(message, message_size,
                       "request body must be a JSON object");
        return -1;
    }
    for (size_t index = 0; index < body->count; ++index) {
        const char *key = body->keys[index];
        const kipp_json_value *value = &body->items[index];
        int sampled = parse_sampling_field(key, value, request, message,
                                           message_size);
        if (sampled < 0) {
            return -1;
        }
        if (sampled > 0) {
            continue;
        }
        if (strcmp(key, "model") == 0) {
            if (value->type != KIPP_JSON_STRING ||
                strcmp(value->string, server_info.checkpoint_id) != 0) {
                (void)snprintf(message, message_size,
                               "model must be \"%s\"",
                               server_info.checkpoint_id);
                return -1;
            }
        } else if (strcmp(key, "prompt") == 0) {
            if (value->type != KIPP_JSON_STRING) {
                (void)snprintf(message, message_size,
                               "prompt must be a single string");
                return -1;
            }
            free(request->prompt);
            request->prompt = strdup(value->string);
            if (request->prompt == NULL) {
                (void)snprintf(message, message_size, "out of memory");
                return -1;
            }
        } else if (strcmp(key, "max_tokens") == 0) {
            if (!json_number_as_u32(value, 1, server_info.context_length,
                                    &request->max_tokens)) {
                (void)snprintf(message, message_size,
                               "max_tokens must be an integer between 1 "
                               "and %u",
                               server_info.context_length);
                return -1;
            }
        } else if (strcmp(key, "temperature") == 0) {
            if (value->type != KIPP_JSON_NUMBER || value->number < 0.0 ||
                value->number > 2.0) {
                (void)snprintf(message, message_size,
                               "temperature must be a number between 0 "
                               "and 2");
                return -1;
            }
            request->temperature = (float)value->number;
        } else if (strcmp(key, "top_p") == 0) {
            if (value->type != KIPP_JSON_NUMBER || value->number <= 0.0 ||
                value->number > 1.0) {
                (void)snprintf(message, message_size,
                               "top_p must be a number greater than 0 and "
                               "at most 1");
                return -1;
            }
            request->top_p = (float)value->number;
        } else if (strcmp(key, "seed") == 0) {
            uint32_t seed;
            if (!json_number_as_u32(value, 1, UINT32_MAX, &seed)) {
                (void)snprintf(message, message_size,
                               "seed must be an integer between 1 and %u",
                               UINT32_MAX);
                return -1;
            }
            request->seed = seed;
        } else if (strcmp(key, "n") == 0) {
            if (!json_number_as_u32(value, 1, SERVER_CHOICE_LIMIT,
                                    &request->choice_count)) {
                (void)snprintf(message, message_size,
                               "n must be an integer between 1 and %u",
                               SERVER_CHOICE_LIMIT);
                return -1;
            }
        } else if (strcmp(key, "stream") == 0) {
            if (value->type != KIPP_JSON_BOOL) {
                (void)snprintf(message, message_size,
                               "stream must be a boolean");
                return -1;
            }
            request->stream = value->boolean;
        } else if (strcmp(key, "stop") == 0) {
            const kipp_json_value *entries = value;
            size_t entry_count = 1;
            if (value->type == KIPP_JSON_ARRAY) {
                entries = value->items;
                entry_count = value->count;
            } else if (value->type != KIPP_JSON_STRING) {
                (void)snprintf(message, message_size,
                               "stop must be a string or an array of up to "
                               "%u strings",
                               SERVER_STOP_LIMIT);
                return -1;
            }
            if (entry_count > SERVER_STOP_LIMIT) {
                (void)snprintf(message, message_size,
                               "stop accepts at most %u strings",
                               SERVER_STOP_LIMIT);
                return -1;
            }
            for (size_t stop = 0; stop < entry_count; ++stop) {
                const kipp_json_value *entry =
                    value->type == KIPP_JSON_ARRAY ? &entries[stop] : value;
                if (entry->type != KIPP_JSON_STRING ||
                    entry->string[0] == '\0') {
                    (void)snprintf(message, message_size,
                                   "stop strings must be non-empty");
                    return -1;
                }
                request->stops[request->stop_count] = strdup(entry->string);
                if (request->stops[request->stop_count] == NULL) {
                    (void)snprintf(message, message_size, "out of memory");
                    return -1;
                }
                ++request->stop_count;
            }
        } else if (strcmp(key, "logprobs") == 0) {
            if (value->type == KIPP_JSON_NULL) {
                continue; /* explicit null disables logprobs */
            }
            uint32_t top;
            if (!json_number_as_u32(value, 0, 5, &top)) {
                (void)snprintf(message, message_size,
                               "logprobs must be an integer between 0 and 5");
                return -1;
            }
            request->logprobs_enabled = true;
            request->logprobs_top = top;
        } else if (strcmp(key, "stream_options") == 0) {
            if (parse_stream_options(value, request, message, message_size) !=
                0) {
                return -1;
            }
        } else {
            (void)snprintf(message, message_size,
                           "unsupported field \"%s\"", key);
            return -1;
        }
    }
    if (request->prompt == NULL) {
        (void)snprintf(message, message_size, "prompt is required");
        return -1;
    }
    return 0;
}

static kipp_chat_role chat_role_from_string(const char *role, bool *ok) {
    *ok = true;
    if (strcmp(role, "system") == 0) {
        return KIPP_ROLE_SYSTEM;
    }
    if (strcmp(role, "user") == 0) {
        return KIPP_ROLE_USER;
    }
    if (strcmp(role, "assistant") == 0) {
        return KIPP_ROLE_ASSISTANT;
    }
    *ok = false;
    return KIPP_ROLE_USER;
}

/*
 * Validate a Chat Completions body and lower it to a completion_request by
 * rendering the messages into a ChatML prompt. Shares every sampling field
 * with the text-completion parser; the only chat-specific inputs are the
 * `messages` array and optional `chat_template_kwargs.enable_thinking`.
 */
static int parse_chat_request(const kipp_json_value *body,
                              completion_request *request, char *message,
                              size_t message_size) {
    memset(request, 0, sizeof(*request));
    request->max_tokens = SERVER_DEFAULT_MAX_TOKENS;
    request->choice_count = 1;
    request->temperature = 1.0f;
    request->top_p = 1.0f;
    request->repetition_penalty = 1.0f;
    request->is_chat = true;

    if (body->type != KIPP_JSON_OBJECT) {
        (void)snprintf(message, message_size,
                       "request body must be a JSON object");
        return -1;
    }
    const kipp_json_value *messages = NULL;
    bool enable_thinking = true;
    bool saw_top_logprobs = false;
    for (size_t index = 0; index < body->count; ++index) {
        const char *key = body->keys[index];
        const kipp_json_value *value = &body->items[index];
        int sampled = parse_sampling_field(key, value, request, message,
                                           message_size);
        if (sampled < 0) {
            return -1;
        }
        if (sampled > 0) {
            continue;
        }
        if (strcmp(key, "messages") == 0) {
            if (value->type != KIPP_JSON_ARRAY || value->count == 0) {
                (void)snprintf(message, message_size,
                               "messages must be a non-empty array");
                return -1;
            }
            messages = value;
        } else if (strcmp(key, "model") == 0) {
            if (value->type != KIPP_JSON_STRING ||
                strcmp(value->string, server_info.checkpoint_id) != 0) {
                (void)snprintf(message, message_size, "model must be \"%s\"",
                               server_info.checkpoint_id);
                return -1;
            }
        } else if (strcmp(key, "max_tokens") == 0 ||
                   strcmp(key, "max_completion_tokens") == 0) {
            if (!json_number_as_u32(value, 1, server_info.context_length,
                                    &request->max_tokens)) {
                (void)snprintf(message, message_size,
                               "%s must be an integer between 1 and %u", key,
                               server_info.context_length);
                return -1;
            }
        } else if (strcmp(key, "temperature") == 0) {
            if (value->type != KIPP_JSON_NUMBER || value->number < 0.0 ||
                value->number > 2.0) {
                (void)snprintf(message, message_size,
                               "temperature must be a number between 0 and 2");
                return -1;
            }
            request->temperature = (float)value->number;
        } else if (strcmp(key, "top_p") == 0) {
            if (value->type != KIPP_JSON_NUMBER || value->number <= 0.0 ||
                value->number > 1.0) {
                (void)snprintf(message, message_size,
                               "top_p must be greater than 0 and at most 1");
                return -1;
            }
            request->top_p = (float)value->number;
        } else if (strcmp(key, "seed") == 0) {
            uint32_t seed;
            if (!json_number_as_u32(value, 1, UINT32_MAX, &seed)) {
                (void)snprintf(message, message_size,
                               "seed must be an integer between 1 and %u",
                               UINT32_MAX);
                return -1;
            }
            request->seed = seed;
        } else if (strcmp(key, "n") == 0) {
            if (!json_number_as_u32(value, 1, SERVER_CHOICE_LIMIT,
                                    &request->choice_count)) {
                (void)snprintf(message, message_size,
                               "n must be an integer between 1 and %u",
                               SERVER_CHOICE_LIMIT);
                return -1;
            }
        } else if (strcmp(key, "stream") == 0) {
            if (value->type != KIPP_JSON_BOOL) {
                (void)snprintf(message, message_size,
                               "stream must be a boolean");
                return -1;
            }
            request->stream = value->boolean;
        } else if (strcmp(key, "chat_template_kwargs") == 0) {
            if (value->type != KIPP_JSON_OBJECT) {
                (void)snprintf(message, message_size,
                               "chat_template_kwargs must be an object");
                return -1;
            }
            for (size_t inner = 0; inner < value->count; ++inner) {
                if (strcmp(value->keys[inner], "enable_thinking") == 0) {
                    if (value->items[inner].type != KIPP_JSON_BOOL) {
                        (void)snprintf(message, message_size,
                                       "enable_thinking must be a boolean");
                        return -1;
                    }
                    enable_thinking = value->items[inner].boolean;
                }
            }
        } else if (strcmp(key, "stop") == 0) {
            const kipp_json_value *entries = value;
            size_t entry_count = 1;
            if (value->type == KIPP_JSON_ARRAY) {
                entries = value->items;
                entry_count = value->count;
            } else if (value->type != KIPP_JSON_STRING) {
                (void)snprintf(message, message_size,
                               "stop must be a string or array of up to %u "
                               "strings",
                               SERVER_STOP_LIMIT);
                return -1;
            }
            if (entry_count > SERVER_STOP_LIMIT) {
                (void)snprintf(message, message_size,
                               "stop accepts at most %u strings",
                               SERVER_STOP_LIMIT);
                return -1;
            }
            for (size_t stop = 0; stop < entry_count; ++stop) {
                const kipp_json_value *entry =
                    value->type == KIPP_JSON_ARRAY ? &entries[stop] : value;
                if (entry->type != KIPP_JSON_STRING || entry->string[0] == '\0') {
                    (void)snprintf(message, message_size,
                                   "stop strings must be non-empty");
                    return -1;
                }
                request->stops[request->stop_count] = strdup(entry->string);
                if (request->stops[request->stop_count] == NULL) {
                    (void)snprintf(message, message_size, "out of memory");
                    return -1;
                }
                ++request->stop_count;
            }
        } else if (strcmp(key, "logprobs") == 0) {
            if (value->type == KIPP_JSON_NULL) {
                continue; /* explicit null disables logprobs */
            }
            if (value->type != KIPP_JSON_BOOL) {
                (void)snprintf(message, message_size,
                               "logprobs must be a boolean");
                return -1;
            }
            request->logprobs_enabled = value->boolean;
        } else if (strcmp(key, "top_logprobs") == 0) {
            uint32_t top;
            if (!json_number_as_u32(value, 0, 20, &top)) {
                (void)snprintf(message, message_size,
                               "top_logprobs must be an integer between 0 "
                               "and 20");
                return -1;
            }
            request->logprobs_top = top;
            saw_top_logprobs = true;
        } else if (strcmp(key, "stream_options") == 0) {
            if (parse_stream_options(value, request, message, message_size) !=
                0) {
                return -1;
            }
        } else {
            (void)snprintf(message, message_size, "unsupported field \"%s\"",
                           key);
            return -1;
        }
    }
    if (saw_top_logprobs && !request->logprobs_enabled) {
        (void)snprintf(message, message_size,
                       "top_logprobs requires logprobs to be true");
        return -1;
    }
    if (messages == NULL) {
        (void)snprintf(message, message_size, "messages is required");
        return -1;
    }

    kipp_chat_message *rendered = malloc(messages->count * sizeof(*rendered));
    if (rendered == NULL) {
        (void)snprintf(message, message_size, "out of memory");
        return -1;
    }
    for (size_t index = 0; index < messages->count; ++index) {
        const kipp_json_value *entry = &messages->items[index];
        const char *role = NULL;
        const char *content = NULL;
        if (entry->type != KIPP_JSON_OBJECT) {
            free(rendered);
            (void)snprintf(message, message_size,
                           "message %zu must be an object", index);
            return -1;
        }
        for (size_t field = 0; field < entry->count; ++field) {
            if (strcmp(entry->keys[field], "role") == 0 &&
                entry->items[field].type == KIPP_JSON_STRING) {
                role = entry->items[field].string;
            } else if (strcmp(entry->keys[field], "content") == 0 &&
                       entry->items[field].type == KIPP_JSON_STRING) {
                content = entry->items[field].string;
            }
        }
        bool role_ok = false;
        if (role == NULL || content == NULL) {
            free(rendered);
            (void)snprintf(message, message_size,
                           "message %zu needs string role and content", index);
            return -1;
        }
        rendered[index].role = chat_role_from_string(role, &role_ok);
        rendered[index].content = content;
        if (!role_ok) {
            free(rendered);
            (void)snprintf(message, message_size,
                           "message %zu has an unsupported role \"%s\"", index,
                           role);
            return -1;
        }
    }

    kipp_chat_options options = {server_info.variant, true, enable_thinking};
    kipp_error error = {0};
    int status =
        kipp_chat_render(rendered, messages->count, &options,
                         &request->prompt, &error);
    free(rendered);
    if (status != 0) {
        (void)snprintf(message, message_size, "%s", error.message);
        return -1;
    }
    return 0;
}

/*
 * Search generated text from search_start for the earliest stop string.
 * Earlier offsets were already checked when previous tokens were appended.
 * Returns the byte offset where the match starts, or generated_length when
 * none matched.
 */
static size_t find_stop_match(const completion_request *request,
                              const char *generated,
                              size_t generated_length, size_t search_start,
                              bool *found) {
    size_t earliest = generated_length;
    *found = false;
    for (size_t index = 0; index < request->stop_count; ++index) {
        const char *stop = request->stops[index];
        size_t stop_length = strlen(stop);
        if (stop_length > generated_length) {
            continue;
        }
        for (size_t offset = search_start;
             offset + stop_length <= generated_length; ++offset) {
            if (memcmp(generated + offset, stop, stop_length) == 0) {
                if (offset < earliest) {
                    earliest = offset;
                    *found = true;
                }
                break;
            }
        }
    }
    return earliest;
}

/* --------------------------------------------------- generation engine */

static uint64_t completion_counter;

/* Monotonic nanoseconds for the timings object (immune to wall-clock jumps). */
static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

/* Generation state for one of a request's n choices. */
#define SERVER_PENALTY_WINDOW 64u

/* One reported alternative for a generated position (top_logprobs entry). */
typedef struct {
    uint32_t token;
    float logprob;
    char *piece;      /* detokenized text, owned */
    size_t piece_len;
} logprob_alt;

/* Recorded logprobs for one generated token position. */
typedef struct {
    uint32_t token;
    float logprob;    /* log-softmax of the raw model logits */
    char *piece;      /* detokenized text, owned */
    size_t piece_len;
    size_t text_offset; /* byte offset of this token within the completion */
    logprob_alt *alts;
    uint32_t alt_count;
} logprob_step;

typedef struct {
    kipp_session *session;
    float *logits;
    string_builder text;
    size_t emitted;
    uint32_t completion_tokens;
    uint64_t rng;
    const char *finish_reason;
    uint32_t next_token;
    bool has_next;
    /* Most-recent generated tokens, oldest first, for penalty sampling. */
    uint32_t recent[SERVER_PENALTY_WINDOW];
    size_t recent_count;
    /* Per-position logprobs, recorded at sample time when requested. */
    logprob_step *logprobs;
    size_t logprob_count;
    size_t logprob_capacity;
    size_t streamed_logprobs; /* steps already flushed on the SSE stream */
} server_choice;

static void logprob_step_free(logprob_step *step) {
    free(step->piece);
    for (uint32_t index = 0; index < step->alt_count; ++index) {
        free(step->alts[index].piece);
    }
    free(step->alts);
}

static void choice_free_logprobs(server_choice *state) {
    for (size_t index = 0; index < state->logprob_count; ++index) {
        logprob_step_free(&state->logprobs[index]);
    }
    free(state->logprobs);
    state->logprobs = NULL;
    state->logprob_count = 0;
    state->logprob_capacity = 0;
    state->streamed_logprobs = 0;
}

/* Detokenize one token into an owned copy (kipp_text_free'd internally). */
static int detok_copy(kipp_model *model, uint32_t token, char **out_piece,
                      size_t *out_len) {
    char *piece = NULL;
    size_t length = 0;
    kipp_error error = {0};
    if (kipp_detokenize(model, &token, 1, &piece, &length, &error) != 0) {
        return -1;
    }
    char *copy = malloc(length + 1);
    if (copy == NULL) {
        kipp_text_free(piece);
        return -1;
    }
    memcpy(copy, piece, length);
    copy[length] = '\0';
    kipp_text_free(piece);
    *out_piece = copy;
    *out_len = length;
    return 0;
}

/*
 * Record log-probabilities for one generated position from the raw model
 * logits still held in state->logits (kipp_sample_ex samples off a copy, so
 * the buffer is untouched). Captures the chosen token plus the top_n most
 * likely alternatives. Returns 0, or -1 on allocation/detokenize failure.
 */
static int choice_capture_logprobs(server_choice *state, uint32_t chosen,
                                   uint32_t top_n, size_t text_offset,
                                   kipp_model *model) {
    if (state->logprob_count == state->logprob_capacity) {
        size_t next = state->logprob_capacity ? state->logprob_capacity * 2 : 16;
        logprob_step *grown =
            realloc(state->logprobs, next * sizeof(*grown));
        if (grown == NULL) {
            return -1;
        }
        state->logprobs = grown;
        state->logprob_capacity = next;
    }
    const float *logits = state->logits;

    /* Numerically-stable log-sum-exp over the full vocabulary. */
    float max_logit = logits[0];
    for (size_t index = 1; index < KIPP_VOCAB_SIZE; ++index) {
        if (logits[index] > max_logit) {
            max_logit = logits[index];
        }
    }
    double sum = 0.0;
    for (size_t index = 0; index < KIPP_VOCAB_SIZE; ++index) {
        sum += exp((double)logits[index] - (double)max_logit);
    }
    double lse = (double)max_logit + log(sum);

    logprob_step *step = &state->logprobs[state->logprob_count];
    memset(step, 0, sizeof(*step));
    step->token = chosen;
    step->logprob = (float)((double)logits[chosen] - lse);
    step->text_offset = text_offset;
    if (detok_copy(model, chosen, &step->piece, &step->piece_len) != 0) {
        return -1;
    }

    if (top_n > 0) {
        step->alts = calloc(top_n, sizeof(*step->alts));
        if (step->alts == NULL) {
            free(step->piece);
            return -1;
        }
        /* Partial selection: repeatedly pick the next-highest logit. */
        float threshold = INFINITY;
        uint32_t threshold_token = 0;
        for (uint32_t rank = 0; rank < top_n; ++rank) {
            float best = -INFINITY;
            uint32_t best_token = 0;
            bool found = false;
            for (size_t index = 0; index < KIPP_VOCAB_SIZE; ++index) {
                float value = logits[index];
                bool below = value < threshold ||
                             (value == threshold && index > threshold_token);
                if (below && (!found || value > best ||
                              (value == best && index < best_token))) {
                    best = value;
                    best_token = (uint32_t)index;
                    found = true;
                }
            }
            if (!found) {
                break;
            }
            logprob_alt *alt = &step->alts[step->alt_count];
            alt->token = best_token;
            alt->logprob = (float)((double)best - lse);
            if (detok_copy(model, best_token, &alt->piece, &alt->piece_len) !=
                0) {
                logprob_step_free(step);
                memset(step, 0, sizeof(*step));
                return -1;
            }
            ++step->alt_count;
            threshold = best;
            threshold_token = best_token;
        }
    }
    ++state->logprob_count;
    return 0;
}

/* Append a generated token to the penalty window, dropping the oldest. */
static void choice_remember_token(server_choice *state, uint32_t token) {
    if (state->recent_count < SERVER_PENALTY_WINDOW) {
        state->recent[state->recent_count++] = token;
        return;
    }
    memmove(state->recent, state->recent + 1,
            (SERVER_PENALTY_WINDOW - 1) * sizeof(*state->recent));
    state->recent[SERVER_PENALTY_WINDOW - 1] = token;
}

/*
 * One cached session and its evaluated token timeline for serial prefix
 * reuse. Only an idle single-choice request may adopt it; concurrent
 * requests use ephemeral sessions instead.
 */
typedef struct {
    kipp_session *session;
    uint32_t *tokens;
    size_t count;
    uint32_t capacity;
    bool busy;
} session_cache;

static session_cache cache;

static void session_cache_drop(void) {
    kipp_session_destroy(cache.session);
    free(cache.tokens);
    memset(&cache, 0, sizeof(cache));
}

static int session_cache_remember(uint32_t token) {
    if (cache.count == cache.capacity) {
        return -1;
    }
    cache.tokens[cache.count++] = token;
    return 0;
}

struct server_connection;

/* One admitted completion request, stepped by the shared batch loop. */
typedef struct {
    struct server_connection *conn; /* NULL once the client disconnects */
    completion_request request;
    kipp_tokens tokens;
    server_choice choices[SERVER_CHOICE_LIMIT];
    uint32_t choice_count;
    float *logits;
    size_t prefill_progress; /* prompt tokens already evaluated */
    bool sampling;           /* prefill finished; logits are live */
    bool uses_cache;
    uint64_t id;
    long long created;
    long long admitted_ns;     /* prompt clock start */
    long long prefill_done_ns; /* first sample begins here */
    size_t longest_stop;
    uint32_t steps;
    bool in_use;
} generation;

/* ------------------------------------------------------ connection layer */

typedef enum {
    CONNECTION_FREE = 0,
    CONNECTION_READING,
    CONNECTION_WAITING, /* complete request parsed; awaiting admission */
    CONNECTION_GENERATING,
    CONNECTION_DRAINING /* flush the output buffer, then close */
} connection_phase;

typedef struct server_connection {
    int fd;
    connection_phase phase;
    string_builder in;
    size_t header_end; /* 0 until the blank line is seen */
    size_t body_length;
    char method[8];
    char path[128];
    string_builder out;
    size_t out_sent;
    uint64_t arrival; /* admission FIFO order for waiting requests */
    generation *gen;
} server_connection;

static server_connection connections[SERVER_MAX_CONNECTIONS];
static generation generations[SERVER_MAX_GENERATIONS];
static uint64_t arrival_counter;

static void connection_close(server_connection *conn) {
    if (conn->fd >= 0) {
        (void)close(conn->fd);
    }
    sb_free(&conn->in);
    sb_free(&conn->out);
    if (conn->gen != NULL) {
        conn->gen->conn = NULL;
    }
    memset(conn, 0, sizeof(*conn));
    conn->fd = -1;
}

/* Append pre-built bytes to the connection's output buffer. */
static void connection_send(server_connection *conn, const char *bytes,
                            size_t count) {
    sb_append_bytes(&conn->out, bytes, count);
}

static void connection_respond(server_connection *conn, int status,
                               const char *reason, const char *body) {
    string_builder head = {0};
    sb_append_format(&head,
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: %zu\r\n"
                     "Connection: close\r\n"
                     "\r\n",
                     status, reason, strlen(body));
    if (!head.failed) {
        connection_send(conn, head.data, head.length);
        connection_send(conn, body, strlen(body));
    }
    sb_free(&head);
    conn->phase = CONNECTION_DRAINING;
}

static void connection_error(server_connection *conn, int status,
                             const char *reason, const char *type,
                             const char *format, ...) {
    char message[256];
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    string_builder body = {0};
    sb_append(&body, "{\"error\":{\"message\":");
    sb_append_json_string(&body, message, strlen(message));
    sb_append_format(&body, ",\"type\":\"%s\"}}", type);
    if (!body.failed) {
        connection_respond(conn, status, reason, body.data);
    } else {
        conn->phase = CONNECTION_DRAINING;
    }
    sb_free(&body);
}

/*
 * Like sb_append_json_string, but a single token's detokenized piece can be
 * a partial UTF-8 sequence (byte-fallback tokens split multi-byte glyphs).
 * Invalid sequences become U+FFFD so the JSON string stays valid Unicode;
 * the exact bytes are still reported in the accompanying `bytes` array.
 */
static void sb_append_json_token(string_builder *builder, const char *text,
                                 size_t length) {
    sb_append(builder, "\"");
    size_t index = 0;
    while (index < length) {
        unsigned char lead = (unsigned char)text[index];
        size_t seq = 0;
        if (lead < 0x80) {
            seq = 1;
        } else if ((lead & 0xE0) == 0xC0) {
            seq = 2;
        } else if ((lead & 0xF0) == 0xE0) {
            seq = 3;
        } else if ((lead & 0xF8) == 0xF0) {
            seq = 4;
        }
        bool valid = seq >= 1 && index + seq <= length;
        for (size_t k = 1; valid && k < seq; ++k) {
            if (((unsigned char)text[index + k] & 0xC0) != 0x80) {
                valid = false;
            }
        }
        if (!valid) {
            sb_append(builder, "\xEF\xBF\xBD"); /* U+FFFD */
            ++index;
            continue;
        }
        if (seq == 1) {
            switch (lead) {
            case '"':
                sb_append(builder, "\\\"");
                break;
            case '\\':
                sb_append(builder, "\\\\");
                break;
            case '\b':
                sb_append(builder, "\\b");
                break;
            case '\f':
                sb_append(builder, "\\f");
                break;
            case '\n':
                sb_append(builder, "\\n");
                break;
            case '\r':
                sb_append(builder, "\\r");
                break;
            case '\t':
                sb_append(builder, "\\t");
                break;
            default:
                if (lead < 0x20) {
                    sb_append_format(builder, "\\u%04x", lead);
                } else {
                    sb_append_bytes(builder, &text[index], 1);
                }
                break;
            }
        } else {
            sb_append_bytes(builder, &text[index], seq);
        }
        index += seq;
    }
    sb_append(builder, "\"");
}

/* JSON number for a logprob; keeps output valid if a value is non-finite. */
static void sb_append_logprob_number(string_builder *sb, float logprob) {
    if (isfinite(logprob)) {
        sb_append_format(sb, "%.10g", (double)logprob);
    } else {
        sb_append(sb, "-1e309"); /* client reads this as -Infinity */
    }
}

/* The token's raw UTF-8 bytes as a JSON array of integers. */
static void sb_append_bytes_array(string_builder *sb, const char *piece,
                                  size_t length) {
    sb_append(sb, "[");
    for (size_t index = 0; index < length; ++index) {
        sb_append_format(sb, "%s%u", index ? "," : "",
                         (unsigned)(unsigned char)piece[index]);
    }
    sb_append(sb, "]");
}

/* Chat logprobs: the `content` array entries for steps [from, to). */
static void sb_append_logprobs_chat(string_builder *sb,
                                    const logprob_step *steps, size_t from,
                                    size_t to) {
    for (size_t index = from; index < to; ++index) {
        const logprob_step *step = &steps[index];
        if (index > from) {
            sb_append(sb, ",");
        }
        sb_append(sb, "{\"token\":");
        sb_append_json_token(sb, step->piece, step->piece_len);
        sb_append(sb, ",\"logprob\":");
        sb_append_logprob_number(sb, step->logprob);
        sb_append(sb, ",\"bytes\":");
        sb_append_bytes_array(sb, step->piece, step->piece_len);
        sb_append(sb, ",\"top_logprobs\":[");
        for (uint32_t alt = 0; alt < step->alt_count; ++alt) {
            if (alt > 0) {
                sb_append(sb, ",");
            }
            sb_append(sb, "{\"token\":");
            sb_append_json_token(sb, step->alts[alt].piece,
                                 step->alts[alt].piece_len);
            sb_append(sb, ",\"logprob\":");
            sb_append_logprob_number(sb, step->alts[alt].logprob);
            sb_append(sb, ",\"bytes\":");
            sb_append_bytes_array(sb, step->alts[alt].piece,
                                  step->alts[alt].piece_len);
            sb_append(sb, "}");
        }
        sb_append(sb, "]}");
    }
}

/* Legacy completions logprobs object for steps [from, to). */
static void sb_append_logprobs_legacy(string_builder *sb,
                                      const logprob_step *steps, size_t from,
                                      size_t to) {
    sb_append(sb, "{\"tokens\":[");
    for (size_t index = from; index < to; ++index) {
        if (index > from) {
            sb_append(sb, ",");
        }
        sb_append_json_token(sb, steps[index].piece, steps[index].piece_len);
    }
    sb_append(sb, "],\"token_logprobs\":[");
    for (size_t index = from; index < to; ++index) {
        if (index > from) {
            sb_append(sb, ",");
        }
        sb_append_logprob_number(sb, steps[index].logprob);
    }
    sb_append(sb, "],\"top_logprobs\":[");
    for (size_t index = from; index < to; ++index) {
        const logprob_step *step = &steps[index];
        if (index > from) {
            sb_append(sb, ",");
        }
        sb_append(sb, "{");
        for (uint32_t alt = 0; alt < step->alt_count; ++alt) {
            if (alt > 0) {
                sb_append(sb, ",");
            }
            sb_append_json_token(sb, step->alts[alt].piece,
                                 step->alts[alt].piece_len);
            sb_append(sb, ":");
            sb_append_logprob_number(sb, step->alts[alt].logprob);
        }
        sb_append(sb, "}");
    }
    sb_append(sb, "],\"text_offset\":[");
    for (size_t index = from; index < to; ++index) {
        if (index > from) {
            sb_append(sb, ",");
        }
        sb_append_format(sb, "%zu", steps[index].text_offset);
    }
    sb_append(sb, "]}");
}

/* Emit the logprobs value for steps [from, to) per the endpoint's shape. */
static void sb_append_logprobs_value(string_builder *sb, bool is_chat,
                                     const logprob_step *steps, size_t from,
                                     size_t to) {
    if (is_chat) {
        sb_append(sb, "{\"content\":[");
        sb_append_logprobs_chat(sb, steps, from, to);
        sb_append(sb, "]}");
    } else {
        sb_append_logprobs_legacy(sb, steps, from, to);
    }
}

/* The llama.cpp-style timings object for a finished generation. */
static void sb_append_timings(string_builder *sb, long long admitted_ns,
                              long long prefill_done_ns, long long finish_ns,
                              size_t prompt_n, uint32_t predicted_n) {
    double prompt_ms = (double)(prefill_done_ns - admitted_ns) / 1e6;
    double predicted_ms = (double)(finish_ns - prefill_done_ns) / 1e6;
    if (prompt_ms < 0.0) {
        prompt_ms = 0.0;
    }
    if (predicted_ms < 0.0) {
        predicted_ms = 0.0;
    }
    double prompt_per_second =
        prompt_ms > 0.0 ? (double)prompt_n * 1000.0 / prompt_ms : 0.0;
    double predicted_per_second =
        predicted_ms > 0.0 ? (double)predicted_n * 1000.0 / predicted_ms : 0.0;
    sb_append_format(sb,
                     "{\"prompt_n\":%zu,\"prompt_ms\":%.3f,"
                     "\"prompt_per_second\":%.3f,\"predicted_n\":%u,"
                     "\"predicted_ms\":%.3f,\"predicted_per_second\":%.3f}",
                     prompt_n, prompt_ms, prompt_per_second, predicted_n,
                     predicted_ms, predicted_per_second);
}

/* Queue one SSE event on a generating connection. */
static void connection_send_sse(server_connection *conn, const char *payload,
                                size_t payload_length) {
    connection_send(conn, "data: ", 6);
    connection_send(conn, payload, payload_length);
    connection_send(conn, "\n\n", 2);
}

/*
 * Queue one streaming chunk. `logprobs_json`, when non-NULL, is the
 * pre-rendered value for this chunk's `logprobs` field; NULL emits null.
 */
static void connection_send_chunk(server_connection *conn, uint64_t id,
                                  long long created, uint32_t choice,
                                  const char *delta, size_t delta_length,
                                  const char *finish_reason, bool is_chat,
                                  const char *logprobs_json) {
    string_builder chunk = {0};
    sb_append_format(&chunk,
                     "{\"id\":\"cmpl-%llu\",\"object\":\"%s\",\"created\":%lld,"
                     "\"model\":\"%s\",\"choices\":[{\"%s\":",
                     (unsigned long long)id,
                     is_chat ? "chat.completion.chunk" : "text_completion",
                     created, server_info.checkpoint_id,
                     is_chat ? "delta" : "text");
    if (is_chat) {
        sb_append(&chunk, "{\"role\":\"assistant\",\"content\":");
        sb_append_json_string(&chunk, delta, delta_length);
        sb_append(&chunk, "}");
    } else {
        sb_append_json_string(&chunk, delta, delta_length);
    }
    sb_append_format(&chunk, ",\"index\":%u,\"logprobs\":%s,\"finish_reason\":",
                     choice, logprobs_json != NULL ? logprobs_json : "null");
    if (finish_reason != NULL) {
        sb_append_format(&chunk, "\"%s\"}]}", finish_reason);
    } else {
        sb_append(&chunk, "null}]}");
    }
    if (!chunk.failed) {
        connection_send_sse(conn, chunk.data, chunk.length);
    }
    sb_free(&chunk);
}

/*
 * The final usage-only chunk for stream_options.include_usage: an empty
 * choices array plus usage and timings, sent just before [DONE].
 */
static void connection_send_usage_chunk(server_connection *conn,
                                        generation *gen,
                                        uint32_t total_completion,
                                        long long finish_ns) {
    string_builder chunk = {0};
    sb_append_format(
        &chunk,
        "{\"id\":\"cmpl-%llu\",\"object\":\"%s\",\"created\":%lld,"
        "\"model\":\"%s\",\"choices\":[],\"usage\":{\"prompt_tokens\":%zu,"
        "\"completion_tokens\":%u,\"total_tokens\":%zu},\"timings\":",
        (unsigned long long)gen->id,
        gen->request.is_chat ? "chat.completion.chunk" : "text_completion",
        gen->created, server_info.checkpoint_id, gen->tokens.count,
        total_completion, gen->tokens.count + total_completion);
    sb_append_timings(&chunk, gen->admitted_ns, gen->prefill_done_ns, finish_ns,
                      gen->tokens.count, total_completion);
    sb_append(&chunk, "}");
    if (!chunk.failed) {
        connection_send_sse(conn, chunk.data, chunk.length);
    }
    sb_free(&chunk);
}

/* -------------------------------------------------- generation lifecycle */

/*
 * Release a generation's resources. The prefix cache survives cancellation
 * (its timeline only records successfully evaluated tokens); eval failures
 * drop it separately.
 */
static void generation_release(generation *gen) {
    for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
        sb_free(&gen->choices[choice].text);
        choice_free_logprobs(&gen->choices[choice]);
        if (!gen->uses_cache) {
            kipp_session_destroy(gen->choices[choice].session);
        }
    }
    if (gen->uses_cache) {
        cache.busy = false;
    }
    free(gen->logits);
    kipp_tokens_free(&gen->tokens);
    completion_request_free(&gen->request);
    if (gen->conn != NULL) {
        gen->conn->gen = NULL;
    }
    memset(gen, 0, sizeof(*gen));
}

static void generation_fail(generation *gen, const char *detail) {
    ++server_metrics.requests_failed_total;
    if (gen->uses_cache) {
        cache.busy = false;
        session_cache_drop();
    }
    server_connection *conn = gen->conn;
    if (conn != NULL) {
        if (gen->request.stream) {
            /* SSE already started; the truncated stream signals the error. */
            conn->phase = CONNECTION_DRAINING;
        } else {
            connection_error(conn, 500, "Internal Server Error",
                             "server_error", "generation failed: %s",
                             detail);
        }
    }
    generation_release(gen);
}

/* Build and queue the final response, then retire the generation. */
static void generation_finish(generation *gen) {
    server_connection *conn = gen->conn;
    long long finish_ns = now_ns();
    bool want_logprobs = gen->request.logprobs_enabled;
    for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
        if (gen->choices[choice].finish_reason == NULL) {
            gen->choices[choice].finish_reason = "length";
        }
    }
    if (conn == NULL) {
        generation_release(gen);
        return;
    }
    bool is_chat = gen->request.is_chat;
    if (gen->request.stream) {
        uint32_t total_completion = 0;
        for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
            server_choice *state = &gen->choices[choice];
            total_completion += state->completion_tokens;
            /* Any logprob steps not yet streamed ride the final chunks. */
            string_builder trailing = {0};
            const char *trailing_json = NULL;
            if (want_logprobs &&
                state->logprob_count > state->streamed_logprobs) {
                sb_append_logprobs_value(&trailing, is_chat, state->logprobs,
                                         state->streamed_logprobs,
                                         state->logprob_count);
                state->streamed_logprobs = state->logprob_count;
                trailing_json = trailing.failed ? NULL : trailing.data;
            }
            if (state->text.length > state->emitted) {
                connection_send_chunk(conn, gen->id, gen->created, choice,
                                      state->text.data + state->emitted,
                                      state->text.length - state->emitted,
                                      NULL, is_chat, trailing_json);
                trailing_json = NULL; /* attached to the content chunk */
            }
            connection_send_chunk(conn, gen->id, gen->created, choice, "", 0,
                                  state->finish_reason, is_chat, trailing_json);
            sb_free(&trailing);
        }
        if (gen->request.include_usage) {
            connection_send_usage_chunk(conn, gen, total_completion, finish_ns);
        }
        connection_send_sse(conn, "[DONE]", 6);
        conn->phase = CONNECTION_DRAINING;
    } else {
        string_builder response = {0};
        uint32_t total_completion = 0;
        sb_append_format(&response,
                         "{\"id\":\"cmpl-%llu\","
                         "\"object\":\"%s\","
                         "\"created\":%lld,\"model\":\"%s\","
                         "\"choices\":[",
                         (unsigned long long)gen->id,
                         is_chat ? "chat.completion" : "text_completion",
                         gen->created, server_info.checkpoint_id);
        for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
            server_choice *state = &gen->choices[choice];
            total_completion += state->completion_tokens;
            const char *body_text =
                state->text.data != NULL ? state->text.data : "";
            if (is_chat) {
                sb_append_format(&response,
                                 "%s{\"message\":{\"role\":\"assistant\","
                                 "\"content\":",
                                 choice == 0 ? "" : ",");
                sb_append_json_string(&response, body_text, state->text.length);
                sb_append(&response, "}");
            } else {
                sb_append_format(&response, "%s{\"text\":",
                                 choice == 0 ? "" : ",");
                sb_append_json_string(&response, body_text, state->text.length);
            }
            sb_append_format(&response, ",\"index\":%u,\"logprobs\":", choice);
            if (want_logprobs) {
                sb_append_logprobs_value(&response, is_chat, state->logprobs, 0,
                                         state->logprob_count);
            } else {
                sb_append(&response, "null");
            }
            sb_append_format(&response, ",\"finish_reason\":\"%s\"}",
                             state->finish_reason);
        }
        sb_append_format(&response,
                         "],\"usage\":{\"prompt_tokens\":%zu,"
                         "\"completion_tokens\":%u,"
                         "\"total_tokens\":%zu},\"timings\":",
                         gen->tokens.count, total_completion,
                         gen->tokens.count + total_completion);
        sb_append_timings(&response, gen->admitted_ns, gen->prefill_done_ns,
                          finish_ns, gen->tokens.count, total_completion);
        sb_append(&response, "}");
        if (response.failed) {
            connection_error(conn, 500, "Internal Server Error",
                             "server_error", "response allocation failed");
        } else {
            connection_respond(conn, 200, "OK", response.data);
        }
        sb_free(&response);
    }
    generation_release(gen);
}

static uint32_t active_batch_items(void) {
    uint32_t total = 0;
    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        if (generations[slot].in_use) {
            total += generations[slot].choice_count;
        }
    }
    return total;
}

/*
 * Try to admit a parsed completion request. Returns 0 on admission, 1 when
 * capacity is currently exhausted (leave the request waiting), -1 when the
 * request was rejected or failed (a response has been queued).
 */
static int generation_admit(kipp_model *model, server_connection *conn) {
    kipp_json_value body = {0};
    completion_request request = {0};
    char message[256];
    kipp_error error = {0};

    if (!kipp_json_parse(conn->in.data + conn->header_end, conn->body_length,
                    &body)) {
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "request body is not valid JSON");
        return -1;
    }
    bool is_chat = strcmp(conn->path, "/v1/chat/completions") == 0;
    int parse_status =
        is_chat ? parse_chat_request(&body, &request, message, sizeof(message))
                : parse_completion_request(&body, &request, message,
                                           sizeof(message));
    if (parse_status != 0) {
        kipp_json_free(&body);
        completion_request_free(&request);
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "%s", message);
        return -1;
    }
    kipp_json_free(&body);

    generation *gen = NULL;
    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        if (!generations[slot].in_use) {
            gen = &generations[slot];
            break;
        }
    }
    if (gen == NULL || active_batch_items() + request.choice_count >
                           KIPP_EVAL_BATCH_LIMIT) {
        completion_request_free(&request);
        return 1;
    }

    memset(gen, 0, sizeof(*gen));
    gen->request = request;
    gen->choice_count = request.choice_count;
    if (kipp_tokenize(model, gen->request.prompt, &gen->tokens, &error) !=
        0) {
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "prompt could not be tokenized: %s", error.message);
        goto reject;
    }
    if (gen->tokens.count == 0) {
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "prompt produced no tokens");
        goto reject;
    }
    if (gen->tokens.count + gen->request.max_tokens >
        server_info.context_length) {
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "prompt (%zu tokens) plus max_tokens (%u) exceeds "
                         "the %u-token context",
                         gen->tokens.count, gen->request.max_tokens,
                         server_info.context_length);
        goto reject;
    }
    gen->logits = malloc((size_t)gen->choice_count * KIPP_VOCAB_SIZE *
                         sizeof(*gen->logits));
    if (gen->logits == NULL) {
        connection_error(conn, 500, "Internal Server Error", "server_error",
                         "unable to allocate logits");
        goto reject;
    }

    size_t needed = gen->tokens.count + gen->request.max_tokens;
    if (server_pooled) {
        /*
         * Pooled admission: one pooled session per choice adopts the longest
         * published prefix, then worst-case block reservation decides whether
         * the whole request fits. Refusal leaves the connection WAITING, so
         * pool pressure delays admission instead of corrupting actives.
         */
        uint32_t matched = 0;
        bool session_failed = false;
        for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
            uint32_t choice_matched = 0;
            if (kipp_session_create(model, (uint32_t)needed,
                                    &gen->choices[choice].session,
                                    &error) != 0 ||
                kipp_session_match_prefix(gen->choices[choice].session,
                                          gen->tokens.data, gen->tokens.count,
                                          &choice_matched, &error) != 0 ||
                (choice > 0 && choice_matched != matched)) {
                session_failed = true;
            }
            if (session_failed) {
                for (uint32_t undo = 0; undo <= choice; ++undo) {
                    kipp_session_destroy(gen->choices[undo].session);
                    gen->choices[undo].session = NULL;
                }
                break;
            }
            matched = choice_matched;
        }
        if (session_failed) {
            connection_error(conn, 500, "Internal Server Error",
                             "server_error",
                             "unable to allocate pooled sessions: %s",
                             error.message);
            goto reject;
        }
        /*
         * Worst case, this request still needs its unmatched blocks and
         * every active choice still needs the tail of its own budget. The
         * scheduler is single-threaded, so admitting under this bound means
         * mid-generation allocation can never fail.
         */
        uint64_t request_blocks =
            (needed + KIPP_KV_BLOCK_TOKENS - 1) / KIPP_KV_BLOCK_TOKENS;
        uint64_t needed_blocks =
            (uint64_t)gen->choice_count *
            (request_blocks - matched / KIPP_KV_BLOCK_TOKENS);
        uint64_t outstanding_blocks = 0;
        for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
            generation *active = &generations[slot];
            if (!active->in_use) {
                continue;
            }
            uint64_t budget_blocks =
                (active->tokens.count + active->request.max_tokens +
                 KIPP_KV_BLOCK_TOKENS - 1) /
                KIPP_KV_BLOCK_TOKENS;
            for (uint32_t choice = 0; choice < active->choice_count;
                 ++choice) {
                kipp_session_info info;
                if (kipp_session_get_info(active->choices[choice].session,
                                          &info) == 0) {
                    uint64_t held_blocks =
                        ((uint64_t)info.length + KIPP_KV_BLOCK_TOKENS - 1) /
                        KIPP_KV_BLOCK_TOKENS;
                    outstanding_blocks += budget_blocks > held_blocks
                                              ? budget_blocks - held_blocks
                                              : 0;
                }
            }
        }
        kipp_kv_pool_stats_public stats;
        if (kipp_model_kv_pool_stats(model, &stats) != 0 ||
            stats.free_blocks < needed_blocks + outstanding_blocks) {
            for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
                kipp_session_destroy(gen->choices[choice].session);
                gen->choices[choice].session = NULL;
            }
            free(gen->logits);
            kipp_tokens_free(&gen->tokens);
            completion_request_free(&gen->request);
            memset(gen, 0, sizeof(*gen));
            return 1; /* pool pressure: stay WAITING, retry via FIFO */
        }
        gen->prefill_progress = matched;
        server_metrics.prefix_tokens_reused_total += matched;
        goto admitted;
    }
    if (gen->choice_count == 1 && !cache.busy && cache.session != NULL &&
        cache.capacity >= needed) {
        /* Roll the cached session back to the shared prompt prefix. */
        size_t shared = 0;
        while (shared < cache.count && shared < gen->tokens.count &&
               cache.tokens[shared] == gen->tokens.data[shared]) {
            ++shared;
        }
        if (shared == gen->tokens.count) {
            --shared; /* re-evaluate the last token for its logits */
        }
        if (kipp_session_truncate(cache.session, (uint32_t)shared, &error) ==
            0) {
            cache.count = shared;
            gen->uses_cache = true;
            gen->prefill_progress = shared;
            gen->choices[0].session = cache.session;
        } else {
            session_cache_drop();
        }
    }
    if (!gen->uses_cache && gen->choice_count == 1 && !cache.busy) {
        /* Build a fresh cache entry, rounded up for future reuse. */
        session_cache_drop();
        uint32_t capacity = (uint32_t)((needed + 1023) / 1024 * 1024);
        if (capacity > server_info.context_length) {
            capacity = server_info.context_length;
        }
        cache.tokens = malloc((size_t)capacity * sizeof(*cache.tokens));
        if (cache.tokens != NULL &&
            kipp_session_create(model, capacity, &cache.session, &error) ==
                0) {
            cache.capacity = capacity;
            cache.count = 0;
            gen->uses_cache = true;
            gen->choices[0].session = cache.session;
        } else {
            session_cache_drop();
        }
    }
    if (gen->uses_cache) {
        cache.busy = true;
    } else {
        for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
            if (kipp_session_create(model,
                                    (uint32_t)gen->tokens.count +
                                        gen->request.max_tokens,
                                    &gen->choices[choice].session,
                                    &error) != 0) {
                for (uint32_t undo = 0; undo < choice; ++undo) {
                    kipp_session_destroy(gen->choices[undo].session);
                    gen->choices[undo].session = NULL;
                }
                connection_error(conn, 500, "Internal Server Error",
                                 "server_error",
                                 "unable to allocate choice sessions: %s",
                                 error.message);
                goto reject;
            }
        }
    }

admitted:
    gen->id = ++completion_counter;
    gen->created = (long long)time(NULL);
    gen->admitted_ns = now_ns();
    uint64_t seed_base = gen->request.seed != 0
                             ? gen->request.seed
                             : (uint64_t)time(NULL) * 2654435761u + gen->id;
    for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
        server_choice *state = &gen->choices[choice];
        state->logits = gen->logits + (size_t)choice * KIPP_VOCAB_SIZE;
        state->rng = seed_base + choice;
        if (state->rng == 0) {
            state->rng = 1;
        }
    }
    for (size_t index = 0; index < gen->request.stop_count; ++index) {
        size_t stop_length = strlen(gen->request.stops[index]);
        if (stop_length > gen->longest_stop) {
            gen->longest_stop = stop_length;
        }
    }
    if (gen->request.stream) {
        static const char stream_headers[] =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "\r\n";
        connection_send(conn, stream_headers, sizeof(stream_headers) - 1);
    }
    gen->conn = conn;
    gen->in_use = true;
    ++server_metrics.requests_total;
    server_metrics.prompt_tokens_total += gen->tokens.count;
    conn->gen = gen;
    conn->phase = CONNECTION_GENERATING;
    return 0;

reject:
    free(gen->logits);
    kipp_tokens_free(&gen->tokens);
    /* completion_request_free is safe on the moved struct. */
    completion_request_free(&gen->request);
    memset(gen, 0, sizeof(*gen));
    return -1;
}

/*
 * Sampling half of one scheduler step: every generation with live logits
 * samples one token per unfinished choice, extends its text, applies stop
 * conditions, and queues stream chunks.
 */
static void step_sample(kipp_model *model) {
    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        generation *gen = &generations[slot];
        if (!gen->in_use || !gen->sampling) {
            continue;
        }
        kipp_error error = {0};
        bool any_active = false;
        for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
            server_choice *state = &gen->choices[choice];
            if (state->finish_reason != NULL) {
                continue;
            }
            uint32_t token;
            kipp_sample_params sp = {0};
            sp.temperature = gen->request.temperature;
            sp.top_p = gen->request.top_p;
            sp.top_k = gen->request.top_k;
            sp.min_p = gen->request.min_p;
            sp.frequency_penalty = gen->request.frequency_penalty;
            sp.presence_penalty = gen->request.presence_penalty;
            sp.repetition_penalty = gen->request.repetition_penalty;
            sp.recent_tokens = state->recent;
            sp.recent_count = state->recent_count;
            sp.logit_bias_tokens = gen->request.bias_count > 0
                                       ? gen->request.bias_tokens
                                       : NULL;
            sp.logit_bias_values = gen->request.bias_count > 0
                                       ? gen->request.bias_values
                                       : NULL;
            sp.logit_bias_count = gen->request.bias_count;
            if (kipp_sample_ex(state->logits, KIPP_VOCAB_SIZE, &sp,
                               &state->rng, &token, &error) != 0) {
                generation_fail(gen, error.message);
                break;
            }
            choice_remember_token(state, token);
            if (kipp_model_is_stop_token(model, token)) {
                state->finish_reason = "stop";
                continue;
            }
            char *piece = NULL;
            size_t piece_length = 0;
            if (kipp_detokenize(model, &token, 1, &piece, &piece_length,
                                &error) != 0) {
                generation_fail(gen, error.message);
                break;
            }
            sb_append_bytes(&state->text, piece, piece_length);
            kipp_text_free(piece);
            ++state->completion_tokens;
            ++server_metrics.generation_tokens_total;
            if (state->text.failed) {
                generation_fail(gen, "completion text allocation failed");
                break;
            }
            size_t search_start = 0;
            size_t checked = piece_length + gen->longest_stop - 1;
            if (gen->longest_stop > 0 && state->text.length > checked) {
                search_start = state->text.length - checked;
            }
            bool stop_found = false;
            size_t stop_offset =
                find_stop_match(&gen->request, state->text.data,
                                state->text.length, search_start,
                                &stop_found);
            if (stop_found) {
                state->text.length = stop_offset;
                state->text.data[state->text.length] = '\0';
                state->finish_reason = "stop";
            } else if (gen->steps + 1 == gen->request.max_tokens) {
                state->finish_reason = "length";
            } else {
                state->next_token = token;
                state->has_next = true;
                any_active = true;
            }
            /*
             * Record this token's logprobs from the raw logits while they
             * are still live. Skipped when a stop string truncated the
             * token's text away, so entries stay aligned with the output.
             */
            if (gen->request.logprobs_enabled && !stop_found) {
                size_t offset = state->text.length >= piece_length
                                    ? state->text.length - piece_length
                                    : 0;
                if (choice_capture_logprobs(state, token,
                                            gen->request.logprobs_top, offset,
                                            model) != 0) {
                    generation_fail(gen, "logprobs allocation failed");
                    break;
                }
            }
            if (gen->request.stream && gen->conn != NULL) {
                size_t safe = state->text.length;
                if (state->finish_reason == NULL && gen->longest_stop > 0) {
                    safe = state->text.length > gen->longest_stop - 1
                               ? state->text.length - (gen->longest_stop - 1)
                               : 0;
                }
                if (safe > state->emitted) {
                    string_builder lp = {0};
                    const char *lp_json = NULL;
                    if (gen->request.logprobs_enabled) {
                        size_t upto = state->streamed_logprobs;
                        while (upto < state->logprob_count &&
                               state->logprobs[upto].text_offset +
                                       state->logprobs[upto].piece_len <=
                                   safe) {
                            ++upto;
                        }
                        if (upto > state->streamed_logprobs) {
                            sb_append_logprobs_value(
                                &lp, gen->request.is_chat, state->logprobs,
                                state->streamed_logprobs, upto);
                            state->streamed_logprobs = upto;
                            lp_json = lp.failed ? NULL : lp.data;
                        }
                    }
                    connection_send_chunk(gen->conn, gen->id, gen->created,
                                          choice,
                                          state->text.data + state->emitted,
                                          safe - state->emitted, NULL,
                                          gen->request.is_chat, lp_json);
                    state->emitted = safe;
                    sb_free(&lp);
                }
            }
        }
        if (!gen->in_use) {
            continue; /* failed above */
        }
        ++gen->steps;
        if (!any_active || gen->steps >= gen->request.max_tokens) {
            generation_finish(gen);
        }
    }
}

/*
 * Evaluation half of one scheduler step: chunked prefill for admissions
 * still working through their prompt, plus one decode token per active
 * choice, all in a single batched backend call.
 */
static void step_eval(kipp_model *model) {
    kipp_batch_item items[KIPP_EVAL_BATCH_LIMIT];
    generation *owner[KIPP_EVAL_BATCH_LIMIT];
    size_t item_count = 0;
    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        generation *gen = &generations[slot];
        if (!gen->in_use) {
            continue;
        }
        if (!gen->sampling) {
            size_t remaining = gen->tokens.count - gen->prefill_progress;
            size_t chunk = remaining < SERVER_PREFILL_CHUNK
                               ? remaining
                               : SERVER_PREFILL_CHUNK;
            for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
                items[item_count] = (kipp_batch_item){
                    gen->choices[choice].session,
                    gen->tokens.data + gen->prefill_progress, chunk,
                    gen->choices[choice].logits};
                owner[item_count++] = gen;
            }
        } else {
            for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
                server_choice *state = &gen->choices[choice];
                if (state->finish_reason == NULL && state->has_next) {
                    items[item_count] = (kipp_batch_item){
                        state->session, &state->next_token, 1,
                        state->logits};
                    owner[item_count++] = gen;
                }
            }
        }
    }
    if (item_count == 0) {
        return;
    }
    kipp_error error = {0};
    if (kipp_eval_batch(model, items, item_count, &error) != 0) {
        /* The batch failed as a unit; fail every involved generation. */
        for (size_t index = 0; index < item_count; ++index) {
            if (owner[index]->in_use) {
                generation_fail(owner[index], error.message);
            }
        }
        return;
    }
    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        generation *gen = &generations[slot];
        if (!gen->in_use) {
            continue;
        }
        if (!gen->sampling) {
            size_t remaining = gen->tokens.count - gen->prefill_progress;
            size_t chunk = remaining < SERVER_PREFILL_CHUNK
                               ? remaining
                               : SERVER_PREFILL_CHUNK;
            if (gen->uses_cache) {
                memcpy(cache.tokens + cache.count,
                       gen->tokens.data + gen->prefill_progress,
                       chunk * sizeof(*cache.tokens));
                cache.count += chunk;
            }
            gen->prefill_progress += chunk;
            if (gen->prefill_progress == gen->tokens.count) {
                gen->sampling = true;
                gen->prefill_done_ns = now_ns();
            }
        } else {
            for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
                server_choice *state = &gen->choices[choice];
                if (state->finish_reason == NULL && state->has_next) {
                    if (gen->uses_cache &&
                        session_cache_remember(state->next_token) != 0) {
                        generation_fail(gen, "session timeline overflow");
                        break;
                    }
                    state->has_next = false;
                }
            }
        }
    }
}

static bool any_generation_active(void) {
    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        if (generations[slot].in_use) {
            return true;
        }
    }
    return false;
}

/* ------------------------------------------------------------ event loop */

/* Serve GET /metrics as a Prometheus text exposition. */
static void connection_serve_metrics(server_connection *conn) {
    uint32_t running = 0;
    uint32_t running_choices = 0;
    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        if (generations[slot].in_use) {
            ++running;
            running_choices += generations[slot].choice_count;
        }
    }
    uint32_t waiting = 0;
    for (uint32_t slot = 0; slot < SERVER_MAX_CONNECTIONS; ++slot) {
        if (connections[slot].fd >= 0 &&
            connections[slot].phase == CONNECTION_WAITING) {
            ++waiting;
        }
    }
    string_builder body = {0};
    static const struct {
        const char *name;
        const char *help;
        const char *type;
    } counters[] = {
        {"kipp_requests_total", "Completion requests admitted", "counter"},
        {"kipp_requests_failed_total", "Requests that failed mid-generation",
         "counter"},
        {"kipp_prompt_tokens_total", "Prompt tokens admitted", "counter"},
        {"kipp_generation_tokens_total", "Tokens generated", "counter"},
    };
    const uint64_t values[] = {
        server_metrics.requests_total, server_metrics.requests_failed_total,
        server_metrics.prompt_tokens_total,
        server_metrics.generation_tokens_total};
    for (size_t index = 0; index < sizeof(values) / sizeof(values[0]);
         ++index) {
        sb_append_format(&body, "# HELP %s %s\n# TYPE %s %s\n%s %llu\n",
                         counters[index].name, counters[index].help,
                         counters[index].name, counters[index].type,
                         counters[index].name,
                         (unsigned long long)values[index]);
    }
    sb_append_format(
        &body,
        "# HELP kipp_requests_running Active generations\n"
        "# TYPE kipp_requests_running gauge\nkipp_requests_running %u\n"
        "# HELP kipp_batch_choices_running Active choices in the batch\n"
        "# TYPE kipp_batch_choices_running gauge\n"
        "kipp_batch_choices_running %u\n"
        "# HELP kipp_requests_waiting Parsed requests awaiting admission\n"
        "# TYPE kipp_requests_waiting gauge\nkipp_requests_waiting %u\n",
        running, running_choices, waiting);
    kipp_kv_pool_stats_public pool_stats;
    if (server_pooled &&
        kipp_model_kv_pool_stats(server_model, &pool_stats) == 0) {
        /* One append per metric: sb_append_format's scratch is small. */
        sb_append_format(&body,
                         "# HELP kipp_kv_pool_blocks_total KV pool blocks\n"
                         "# TYPE kipp_kv_pool_blocks_total gauge\n"
                         "kipp_kv_pool_blocks_total %u\n",
                         pool_stats.total_blocks);
        sb_append_format(&body,
                         "# HELP kipp_kv_pool_blocks_free Unreferenced KV "
                         "pool blocks\n"
                         "# TYPE kipp_kv_pool_blocks_free gauge\n"
                         "kipp_kv_pool_blocks_free %u\n",
                         pool_stats.free_blocks);
        sb_append_format(&body,
                         "# HELP kipp_kv_pool_reused_blocks_total Pool "
                         "blocks adopted from the content index\n"
                         "# TYPE kipp_kv_pool_reused_blocks_total counter\n"
                         "kipp_kv_pool_reused_blocks_total %llu\n",
                         (unsigned long long)pool_stats.reused_blocks_total);
        sb_append_format(&body,
                         "# HELP kipp_kv_pool_evicted_blocks_total "
                         "Published blocks reclaimed for new content\n"
                         "# TYPE kipp_kv_pool_evicted_blocks_total counter\n"
                         "kipp_kv_pool_evicted_blocks_total %llu\n",
                         (unsigned long long)pool_stats.evicted_blocks_total);
        sb_append_format(
            &body,
            "# HELP kipp_prefix_tokens_reused_total Prompt tokens adopted "
            "from the pool at admission\n"
            "# TYPE kipp_prefix_tokens_reused_total counter\n"
            "kipp_prefix_tokens_reused_total %llu\n",
            (unsigned long long)server_metrics.prefix_tokens_reused_total);
    }
    if (body.failed) {
        conn->phase = CONNECTION_DRAINING;
        sb_free(&body);
        return;
    }
    string_builder head = {0};
    sb_append_format(&head,
                     "HTTP/1.1 200 OK\r\n"
                     "Content-Type: text/plain; version=0.0.4\r\n"
                     "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                     body.length);
    if (!head.failed) {
        connection_send(conn, head.data, head.length);
        connection_send(conn, body.data, body.length);
    }
    conn->phase = CONNECTION_DRAINING;
    sb_free(&head);
    sb_free(&body);
}

/* Route a fully-read request; completions move to WAITING for admission. */
static void connection_route(server_connection *conn) {
    if (strcmp(conn->method, "GET") == 0 &&
        strcmp(conn->path, "/metrics") == 0) {
        connection_serve_metrics(conn);
    } else if (strcmp(conn->method, "GET") == 0 &&
        strcmp(conn->path, "/healthz") == 0) {
        connection_respond(conn, 200, "OK", "{\"status\":\"ok\"}");
    } else if (strcmp(conn->method, "GET") == 0 &&
               strcmp(conn->path, "/v1/models") == 0) {
        string_builder body = {0};
        sb_append_format(&body,
                         "{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
                         "\"object\":\"model\",\"owned_by\":\"kipp\"}]}",
                         server_info.checkpoint_id);
        if (!body.failed) {
            connection_respond(conn, 200, "OK", body.data);
        } else {
            conn->phase = CONNECTION_DRAINING;
        }
        sb_free(&body);
    } else if (strcmp(conn->method, "POST") == 0 &&
               (strcmp(conn->path, "/v1/completions") == 0 ||
                strcmp(conn->path, "/v1/chat/completions") == 0)) {
        conn->phase = CONNECTION_WAITING;
        conn->arrival = ++arrival_counter;
    } else if (strcmp(conn->path, "/healthz") == 0 ||
               strcmp(conn->path, "/v1/models") == 0 ||
               strcmp(conn->path, "/v1/completions") == 0 ||
               strcmp(conn->path, "/v1/chat/completions") == 0) {
        connection_error(conn, 405, "Method Not Allowed",
                         "invalid_request_error",
                         "method %s is not supported on %s", conn->method,
                         conn->path);
    } else {
        connection_error(conn, 404, "Not Found", "invalid_request_error",
                         "unknown path %s", conn->path);
    }
}

/* Consume readable bytes; returns -1 when the connection must close. */
static int connection_read(server_connection *conn) {
    char chunk[8192];
    for (;;) {
        ssize_t received = recv(conn->fd, chunk, sizeof(chunk), 0);
        if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            return -1;
        }
        if (received == 0) {
            /* Orderly close. Mid-generation this cancels the request. */
            return conn->phase == CONNECTION_DRAINING && conn->out.length > 0
                       ? 0
                       : -1;
        }
        if (conn->phase != CONNECTION_READING) {
            continue; /* discard extra client bytes */
        }
        if (memchr(chunk, '\0', (size_t)received) != NULL) {
            return -1;
        }
        sb_append_bytes(&conn->in, chunk, (size_t)received);
        if (conn->in.failed ||
            conn->in.length > SERVER_HEADER_LIMIT + SERVER_BODY_LIMIT) {
            return -1;
        }
    }
    if (conn->phase != CONNECTION_READING) {
        return 0;
    }
    if (conn->header_end == 0 && conn->in.data != NULL) {
        char *split = strstr(conn->in.data, "\r\n\r\n");
        if (split == NULL) {
            if (conn->in.length > SERVER_HEADER_LIMIT) {
                return -1;
            }
            return 0;
        }
        conn->header_end = (size_t)(split - conn->in.data) + 4;
        *split = '\0';
        if (sscanf(conn->in.data, "%7s %127s", conn->method, conn->path) !=
            2) {
            *split = '\r';
            return -1;
        }
        const char *length_header =
            kipp_http_header_value(conn->in.data, "Content-Length");
        *split = '\r';
        if (length_header != NULL) {
            char *end = NULL;
            errno = 0;
            unsigned long long parsed = strtoull(length_header, &end, 10);
            if (errno != 0 || end == length_header ||
                parsed > SERVER_BODY_LIMIT) {
                return -1;
            }
            conn->body_length = (size_t)parsed;
        }
    }
    if (conn->header_end != 0 &&
        conn->in.length >= conn->header_end + conn->body_length) {
        connection_route(conn);
    }
    return 0;
}

/* Drain pending output; returns -1 when the connection must close. */
static int connection_write(server_connection *conn) {
    while (conn->out_sent < conn->out.length) {
        ssize_t written = send(conn->fd, conn->out.data + conn->out_sent,
                               conn->out.length - conn->out_sent, 0);
        if (written < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return 0;
            }
            return -1;
        }
        conn->out_sent += (size_t)written;
    }
    if (conn->out_sent == conn->out.length &&
        conn->phase == CONNECTION_DRAINING) {
        return -1; /* fully flushed; close */
    }
    return 0;
}

static void serve(kipp_model *model, int listener) {
    for (uint32_t slot = 0; slot < SERVER_MAX_CONNECTIONS; ++slot) {
        connections[slot].fd = -1;
    }
    struct pollfd fds[1 + SERVER_MAX_CONNECTIONS];
    while (server_running) {
        nfds_t count = 0;
        fds[count].fd = listener;
        fds[count].events = POLLIN;
        ++count;
        int poll_map[1 + SERVER_MAX_CONNECTIONS];
        for (uint32_t slot = 0; slot < SERVER_MAX_CONNECTIONS; ++slot) {
            server_connection *conn = &connections[slot];
            if (conn->fd < 0) {
                continue;
            }
            short events = POLLIN;
            if (conn->out_sent < conn->out.length ||
                conn->phase == CONNECTION_DRAINING) {
                events |= POLLOUT;
            }
            fds[count].fd = conn->fd;
            fds[count].events = events;
            poll_map[count] = (int)slot;
            ++count;
        }
        int timeout = any_generation_active() ? 0 : -1;
        int ready = poll(fds, count, timeout);
        if (ready < 0 && errno != EINTR) {
            fprintf(stderr, "kipp-server: poll: %s\n", strerror(errno));
            break;
        }

        if (ready > 0 && (fds[0].revents & POLLIN) != 0) {
            for (;;) {
                int accepted = accept(listener, NULL, NULL);
                if (accepted < 0) {
                    break;
                }
                (void)fcntl(accepted, F_SETFL,
                            fcntl(accepted, F_GETFL, 0) | O_NONBLOCK);
                server_connection *conn = NULL;
                for (uint32_t slot = 0; slot < SERVER_MAX_CONNECTIONS;
                     ++slot) {
                    if (connections[slot].fd < 0) {
                        conn = &connections[slot];
                        break;
                    }
                }
                if (conn == NULL) {
                    (void)close(accepted);
                    continue;
                }
                memset(conn, 0, sizeof(*conn));
                conn->fd = accepted;
                conn->phase = CONNECTION_READING;
            }
        }
        if (ready > 0) {
            for (nfds_t index = 1; index < count; ++index) {
                server_connection *conn = &connections[poll_map[index]];
                if (conn->fd < 0 || fds[index].revents == 0) {
                    continue;
                }
                if ((fds[index].revents & (POLLERR | POLLNVAL)) != 0) {
                    connection_close(conn);
                    continue;
                }
                if ((fds[index].revents & (POLLIN | POLLHUP)) != 0 &&
                    connection_read(conn) != 0) {
                    connection_close(conn);
                    continue;
                }
                if ((fds[index].revents & POLLOUT) != 0 &&
                    connection_write(conn) != 0) {
                    connection_close(conn);
                    continue;
                }
                if (conn->out.length - conn->out_sent >
                    SERVER_OUTPUT_LIMIT) {
                    connection_close(conn); /* unresponsive client */
                }
            }
        }

        /* Admit waiting requests in arrival order while capacity allows. */
        for (;;) {
            server_connection *next = NULL;
            for (uint32_t slot = 0; slot < SERVER_MAX_CONNECTIONS; ++slot) {
                server_connection *conn = &connections[slot];
                if (conn->fd >= 0 && conn->phase == CONNECTION_WAITING &&
                    (next == NULL || conn->arrival < next->arrival)) {
                    next = conn;
                }
            }
            if (next == NULL) {
                break;
            }
            int admitted = generation_admit(model, next);
            if (admitted == 1) {
                break; /* capacity exhausted; keep waiting */
            }
        }

        if (any_generation_active()) {
            step_sample(model);
            step_eval(model);
            /* Push freshly queued output before the next step. */
            for (uint32_t slot = 0; slot < SERVER_MAX_CONNECTIONS; ++slot) {
                server_connection *conn = &connections[slot];
                if (conn->fd >= 0 && conn->out_sent < conn->out.length &&
                    connection_write(conn) != 0) {
                    connection_close(conn);
                }
            }
        }
    }

    for (uint32_t slot = 0; slot < SERVER_MAX_GENERATIONS; ++slot) {
        if (generations[slot].in_use) {
            generation_release(&generations[slot]);
        }
    }
    for (uint32_t slot = 0; slot < SERVER_MAX_CONNECTIONS; ++slot) {
        if (connections[slot].fd >= 0) {
            connection_close(&connections[slot]);
        }
    }
}

/* ------------------------------------------------------------------ main */

static void server_usage(const char *program) {
    fprintf(stderr,
            "Usage: %s --model MODEL.gguf [--backend cpu|metal|cuda] "
            "[--port N] [--kv-pool-mib N]\n"
            "Serves GET /healthz, GET /metrics, GET /v1/models, "
            "POST /v1/completions, and POST /v1/chat/completions "
            "on 127.0.0.1.\n"
            "Concurrent requests decode together through batched "
            "evaluation.\n"
            "--kv-pool-mib sizes the shared KV pool for cross-request "
            "prefix sharing\n"
            "(CPU/Metal; default sizes it to the checkpoint's context "
            "length; 0 disables).\n",
            program);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    kipp_backend_kind backend = KIPP_BACKEND_CPU;
    unsigned long port = SERVER_DEFAULT_PORT;
    unsigned long pool_mib = ULONG_MAX; /* ULONG_MAX = auto-size */
    kipp_model *model = NULL;
    kipp_error error = {0};
    int listener = -1;
    int exit_code = 1;

    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--model") == 0 && index + 1 < argc) {
            model_path = argv[++index];
        } else if (strcmp(argv[index], "--backend") == 0 &&
                   index + 1 < argc) {
            const char *name = argv[++index];
            if (strcmp(name, "cpu") == 0) {
                backend = KIPP_BACKEND_CPU;
            } else if (strcmp(name, "metal") == 0) {
                backend = KIPP_BACKEND_METAL;
            } else if (strcmp(name, "cuda") == 0) {
                backend = KIPP_BACKEND_CUDA;
            } else {
                server_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[index], "--port") == 0 && index + 1 < argc) {
            char *end = NULL;
            errno = 0;
            port = strtoul(argv[++index], &end, 10);
            if (errno != 0 || end == argv[index] || *end != '\0' ||
                port == 0 || port > 65535) {
                server_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[index], "--kv-pool-mib") == 0 &&
                   index + 1 < argc) {
            char *end = NULL;
            errno = 0;
            pool_mib = strtoul(argv[++index], &end, 10);
            if (errno != 0 || end == argv[index] || *end != '\0' ||
                pool_mib == ULONG_MAX) {
                server_usage(argv[0]);
                return 1;
            }
        } else {
            server_usage(argv[0]);
            return 1;
        }
    }
    if (model_path == NULL) {
        server_usage(argv[0]);
        return 1;
    }

    (void)signal(SIGPIPE, SIG_IGN);
    /* No SA_RESTART: a stop signal must interrupt poll() so the serve
     * loop can observe server_running and exit cleanly. */
    struct sigaction stop_action = {0};
    stop_action.sa_handler = handle_stop_signal;
    (void)sigaction(SIGINT, &stop_action, NULL);
    (void)sigaction(SIGTERM, &stop_action, NULL);

    if (kipp_model_open_backend(model_path, backend, &model, &error) != 0) {
        fprintf(stderr, "kipp-server: %s: %s\n",
                kipp_error_code_name(error.code), error.message);
        return 1;
    }
    if (kipp_model_get_info(model, &server_info) != 0) {
        fprintf(stderr, "kipp-server: unable to read model info\n");
        (void)kipp_model_close(model, NULL);
        return 1;
    }
    /*
     * Pool-backed prefix sharing is the CPU/Metal default: reopen the model
     * pooled, sized to the checkpoint's context length (the worst single
     * session today) unless --kv-pool-mib overrides it. 0 disables; CUDA
     * stays on the legacy per-session path.
     */
    if (pool_mib != 0 && backend != KIPP_BACKEND_CUDA) {
        uint64_t per_token_bytes = (uint64_t)server_info.block_count * 2u *
                                   KIPP_ATTENTION_HEAD_COUNT_KV *
                                   KIPP_ATTENTION_HEAD_DIM * sizeof(uint16_t);
        uint64_t per_block_bytes = per_token_bytes * KIPP_KV_BLOCK_TOKENS;
        uint64_t pool_blocks;
        if (pool_mib == ULONG_MAX) {
            pool_blocks = (server_info.context_length +
                           KIPP_KV_BLOCK_TOKENS - 1) /
                          KIPP_KV_BLOCK_TOKENS;
        } else {
            pool_blocks = pool_mib * 1024u * 1024u / per_block_bytes;
        }
        if (pool_blocks == 0 || pool_blocks > UINT32_MAX) {
            fprintf(stderr,
                    "kipp-server: --kv-pool-mib %lu holds no whole %u-token "
                    "block (one block is %llu MiB)\n",
                    pool_mib, KIPP_KV_BLOCK_TOKENS,
                    (unsigned long long)(per_block_bytes / (1024u * 1024u)));
            (void)kipp_model_close(model, NULL);
            return 1;
        }
        kipp_model *pooled_model = NULL;
        if (kipp_model_close(model, &error) != 0) {
            fprintf(stderr, "kipp-server: close: %s\n", error.message);
            return 1;
        }
        model = NULL;
        if (kipp_model_open_pooled(model_path, backend,
                                   (uint32_t)pool_blocks, &pooled_model,
                                   &error) != 0) {
            fprintf(stderr, "kipp-server: pooled open: %s: %s\n",
                    kipp_error_code_name(error.code), error.message);
            return 1;
        }
        model = pooled_model;
        server_pooled = true;
        fprintf(stderr,
                "kipp-server: KV pool %llu blocks (%llu MiB) for "
                "cross-request prefix sharing\n",
                (unsigned long long)pool_blocks,
                (unsigned long long)(pool_blocks * per_block_bytes /
                                     (1024u * 1024u)));
    }
    server_model = model;

    listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        fprintf(stderr, "kipp-server: socket: %s\n", strerror(errno));
        goto cleanup;
    }
    int enable = 1;
    (void)setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &enable,
                     sizeof(enable));
    (void)fcntl(listener, F_SETFL,
                fcntl(listener, F_GETFL, 0) | O_NONBLOCK);
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t)port);
    if (bind(listener, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(listener, 64) != 0) {
        fprintf(stderr, "kipp-server: bind 127.0.0.1:%lu: %s\n", port,
                strerror(errno));
        goto cleanup;
    }
    fprintf(stderr,
            "kipp-server %s: serving %s on http://127.0.0.1:%lu "
            "(backend %s)\n",
            KIPP_VERSION, server_info.checkpoint_id, port,
            kipp_backend_name(backend));

    serve(model, listener);
    exit_code = 0;

cleanup:
    if (listener >= 0) {
        (void)close(listener);
    }
    session_cache_drop();
    if (kipp_model_close(model, &error) != 0) {
        fprintf(stderr, "kipp-server: close: %s\n", error.message);
        exit_code = 1;
    }
    return exit_code;
}
