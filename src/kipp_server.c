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

#include <arpa/inet.h>
#include <errno.h>
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

#define SERVER_MODEL_ID "qwen3-4b-base"
#define SERVER_HEADER_LIMIT (64u * 1024u)
#define SERVER_BODY_LIMIT (8u * 1024u * 1024u)
#define SERVER_DEFAULT_PORT 8080u
#define SERVER_DEFAULT_MAX_TOKENS 16u
#define SERVER_STOP_LIMIT 4u
#define SERVER_CHOICE_LIMIT 8u
#define SERVER_JSON_DEPTH_LIMIT 16u
#define SERVER_MAX_CONNECTIONS 64u
#define SERVER_MAX_GENERATIONS 8u
#define SERVER_PREFILL_CHUNK 32u
#define SERVER_OUTPUT_LIMIT (4u * 1024u * 1024u)

static volatile sig_atomic_t server_running = 1;

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

/* ------------------------------------------------------------------- JSON */

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type;

typedef struct json_value json_value;

struct json_value {
    json_type type;
    bool boolean;
    double number;
    char *string;
    char **keys;
    json_value *items;
    size_t count;
};

static void json_free_value(json_value *value) {
    if (value == NULL) {
        return;
    }
    free(value->string);
    for (size_t index = 0; index < value->count; ++index) {
        if (value->keys != NULL) {
            free(value->keys[index]);
        }
        json_free_value(&value->items[index]);
    }
    free(value->keys);
    free(value->items);
    memset(value, 0, sizeof(*value));
}

typedef struct {
    const char *text;
    size_t length;
    size_t offset;
} json_cursor;

static void json_skip_space(json_cursor *cursor) {
    while (cursor->offset < cursor->length) {
        char c = cursor->text[cursor->offset];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            break;
        }
        ++cursor->offset;
    }
}

static bool json_literal(json_cursor *cursor, const char *literal) {
    size_t length = strlen(literal);
    if (cursor->length - cursor->offset < length ||
        memcmp(cursor->text + cursor->offset, literal, length) != 0) {
        return false;
    }
    cursor->offset += length;
    return true;
}

static size_t json_utf8_encode(char *output, uint32_t codepoint) {
    if (codepoint < 0x80) {
        output[0] = (char)codepoint;
        return 1;
    }
    if (codepoint < 0x800) {
        output[0] = (char)(0xc0 | (codepoint >> 6));
        output[1] = (char)(0x80 | (codepoint & 0x3f));
        return 2;
    }
    if (codepoint < 0x10000) {
        output[0] = (char)(0xe0 | (codepoint >> 12));
        output[1] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        output[2] = (char)(0x80 | (codepoint & 0x3f));
        return 3;
    }
    output[0] = (char)(0xf0 | (codepoint >> 18));
    output[1] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
    output[2] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
    output[3] = (char)(0x80 | (codepoint & 0x3f));
    return 4;
}

static bool json_parse_hex4(json_cursor *cursor, uint32_t *value) {
    if (cursor->length - cursor->offset < 4) {
        return false;
    }
    *value = 0;
    for (int digit = 0; digit < 4; ++digit) {
        char c = cursor->text[cursor->offset + (size_t)digit];
        uint32_t nibble;
        if (c >= '0' && c <= '9') {
            nibble = (uint32_t)(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            nibble = (uint32_t)(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            nibble = (uint32_t)(c - 'A' + 10);
        } else {
            return false;
        }
        *value = (*value << 4) | nibble;
    }
    cursor->offset += 4;
    return true;
}

static bool json_parse_string(json_cursor *cursor, char **output) {
    if (cursor->text[cursor->offset] != '"') {
        return false;
    }
    ++cursor->offset;
    string_builder builder = {0};
    while (cursor->offset < cursor->length) {
        char c = cursor->text[cursor->offset];
        if (c == '"') {
            ++cursor->offset;
            if (builder.failed) {
                sb_free(&builder);
                return false;
            }
            *output = builder.data != NULL ? builder.data : calloc(1, 1);
            return *output != NULL;
        }
        if ((unsigned char)c < 0x20) {
            break;
        }
        if (c != '\\') {
            sb_append_bytes(&builder, &cursor->text[cursor->offset], 1);
            ++cursor->offset;
            continue;
        }
        ++cursor->offset;
        if (cursor->offset >= cursor->length) {
            break;
        }
        char escape = cursor->text[cursor->offset];
        ++cursor->offset;
        char encoded[4];
        uint32_t codepoint;
        switch (escape) {
        case '"':
        case '\\':
        case '/':
            sb_append_bytes(&builder, &escape, 1);
            continue;
        case 'b':
            sb_append(&builder, "\b");
            continue;
        case 'f':
            sb_append(&builder, "\f");
            continue;
        case 'n':
            sb_append(&builder, "\n");
            continue;
        case 'r':
            sb_append(&builder, "\r");
            continue;
        case 't':
            sb_append(&builder, "\t");
            continue;
        case 'u':
            if (!json_parse_hex4(cursor, &codepoint)) {
                sb_free(&builder);
                return false;
            }
            if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                uint32_t low;
                if (!json_literal(cursor, "\\u") ||
                    !json_parse_hex4(cursor, &low) || low < 0xdc00 ||
                    low > 0xdfff) {
                    sb_free(&builder);
                    return false;
                }
                codepoint = 0x10000 +
                            ((codepoint - 0xd800) << 10) + (low - 0xdc00);
            } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                sb_free(&builder);
                return false;
            }
            sb_append_bytes(&builder, encoded,
                            json_utf8_encode(encoded, codepoint));
            continue;
        default:
            sb_free(&builder);
            return false;
        }
    }
    sb_free(&builder);
    return false;
}

static bool json_parse_value(json_cursor *cursor, json_value *value,
                             unsigned depth);

static bool json_parse_collection(json_cursor *cursor, json_value *value,
                                  bool is_object, unsigned depth) {
    char open = is_object ? '{' : '[';
    char close = is_object ? '}' : ']';
    if (cursor->text[cursor->offset] != open) {
        return false;
    }
    ++cursor->offset;
    value->type = is_object ? JSON_OBJECT : JSON_ARRAY;
    json_skip_space(cursor);
    if (cursor->offset < cursor->length &&
        cursor->text[cursor->offset] == close) {
        ++cursor->offset;
        return true;
    }
    size_t capacity = 0;
    while (cursor->offset < cursor->length) {
        if (value->count == capacity) {
            size_t new_capacity = capacity == 0 ? 4 : capacity * 2;
            json_value *items =
                realloc(value->items, new_capacity * sizeof(*items));
            if (items == NULL) {
                return false;
            }
            memset(items + capacity, 0,
                   (new_capacity - capacity) * sizeof(*items));
            value->items = items;
            if (is_object) {
                char **keys =
                    realloc(value->keys, new_capacity * sizeof(*keys));
                if (keys == NULL) {
                    return false;
                }
                memset(keys + capacity, 0,
                       (new_capacity - capacity) * sizeof(*keys));
                value->keys = keys;
            }
            capacity = new_capacity;
        }
        json_skip_space(cursor);
        if (is_object) {
            if (cursor->offset >= cursor->length ||
                !json_parse_string(cursor, &value->keys[value->count])) {
                return false;
            }
            json_skip_space(cursor);
            if (cursor->offset >= cursor->length ||
                cursor->text[cursor->offset] != ':') {
                return false;
            }
            ++cursor->offset;
        }
        json_skip_space(cursor);
        if (cursor->offset >= cursor->length ||
            !json_parse_value(cursor, &value->items[value->count],
                              depth + 1)) {
            return false;
        }
        ++value->count;
        json_skip_space(cursor);
        if (cursor->offset >= cursor->length) {
            return false;
        }
        if (cursor->text[cursor->offset] == ',') {
            ++cursor->offset;
            continue;
        }
        if (cursor->text[cursor->offset] == close) {
            ++cursor->offset;
            return true;
        }
        return false;
    }
    return false;
}

static bool json_parse_value(json_cursor *cursor, json_value *value,
                             unsigned depth) {
    if (depth > SERVER_JSON_DEPTH_LIMIT) {
        return false;
    }
    json_skip_space(cursor);
    if (cursor->offset >= cursor->length) {
        return false;
    }
    char c = cursor->text[cursor->offset];
    if (c == '{' || c == '[') {
        return json_parse_collection(cursor, value, c == '{', depth);
    }
    if (c == '"') {
        value->type = JSON_STRING;
        return json_parse_string(cursor, &value->string);
    }
    if (json_literal(cursor, "true")) {
        value->type = JSON_BOOL;
        value->boolean = true;
        return true;
    }
    if (json_literal(cursor, "false")) {
        value->type = JSON_BOOL;
        value->boolean = false;
        return true;
    }
    if (json_literal(cursor, "null")) {
        value->type = JSON_NULL;
        return true;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char buffer[64];
        size_t length = 0;
        while (cursor->offset < cursor->length &&
               length + 1 < sizeof(buffer)) {
            char digit = cursor->text[cursor->offset];
            if (digit != '-' && digit != '+' && digit != '.' &&
                digit != 'e' && digit != 'E' &&
                (digit < '0' || digit > '9')) {
                break;
            }
            buffer[length++] = digit;
            ++cursor->offset;
        }
        buffer[length] = '\0';
        char *end = NULL;
        errno = 0;
        value->number = strtod(buffer, &end);
        if (errno != 0 || end != buffer + length || length == 0 ||
            !isfinite(value->number)) {
            return false;
        }
        value->type = JSON_NUMBER;
        return true;
    }
    return false;
}

static bool json_parse(const char *text, size_t length, json_value *value) {
    json_cursor cursor = {text, length, 0};
    memset(value, 0, sizeof(*value));
    if (!json_parse_value(&cursor, value, 0)) {
        json_free_value(value);
        return false;
    }
    json_skip_space(&cursor);
    if (cursor.offset != length) {
        json_free_value(value);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------ completions */

typedef struct {
    char *prompt;
    uint32_t max_tokens;
    uint32_t choice_count;
    float temperature;
    float top_p;
    uint64_t seed;
    bool stream;
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

static bool json_number_as_u32(const json_value *value, uint32_t minimum,
                               uint32_t maximum, uint32_t *output) {
    if (value->type != JSON_NUMBER || value->number != floor(value->number) ||
        value->number < (double)minimum || value->number > (double)maximum) {
        return false;
    }
    *output = (uint32_t)value->number;
    return true;
}

/*
 * Validate the request body against the supported field set. Returns 0 on
 * success; on failure writes a client-facing message and returns -1.
 */
static int parse_completion_request(const json_value *body,
                                    completion_request *request,
                                    char *message, size_t message_size) {
    memset(request, 0, sizeof(*request));
    request->max_tokens = SERVER_DEFAULT_MAX_TOKENS;
    request->choice_count = 1;
    request->temperature = 1.0f;
    request->top_p = 1.0f;
    request->seed = 0;

    if (body->type != JSON_OBJECT) {
        (void)snprintf(message, message_size,
                       "request body must be a JSON object");
        return -1;
    }
    for (size_t index = 0; index < body->count; ++index) {
        const char *key = body->keys[index];
        const json_value *value = &body->items[index];
        if (strcmp(key, "model") == 0) {
            if (value->type != JSON_STRING ||
                strcmp(value->string, SERVER_MODEL_ID) != 0) {
                (void)snprintf(message, message_size,
                               "model must be \"%s\"", SERVER_MODEL_ID);
                return -1;
            }
        } else if (strcmp(key, "prompt") == 0) {
            if (value->type != JSON_STRING) {
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
            if (!json_number_as_u32(value, 1, KIPP_CONTEXT_LENGTH,
                                    &request->max_tokens)) {
                (void)snprintf(message, message_size,
                               "max_tokens must be an integer between 1 "
                               "and %u",
                               KIPP_CONTEXT_LENGTH);
                return -1;
            }
        } else if (strcmp(key, "temperature") == 0) {
            if (value->type != JSON_NUMBER || value->number < 0.0 ||
                value->number > 2.0) {
                (void)snprintf(message, message_size,
                               "temperature must be a number between 0 "
                               "and 2");
                return -1;
            }
            request->temperature = (float)value->number;
        } else if (strcmp(key, "top_p") == 0) {
            if (value->type != JSON_NUMBER || value->number <= 0.0 ||
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
            if (value->type != JSON_BOOL) {
                (void)snprintf(message, message_size,
                               "stream must be a boolean");
                return -1;
            }
            request->stream = value->boolean;
        } else if (strcmp(key, "stop") == 0) {
            const json_value *entries = value;
            size_t entry_count = 1;
            if (value->type == JSON_ARRAY) {
                entries = value->items;
                entry_count = value->count;
            } else if (value->type != JSON_STRING) {
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
                const json_value *entry =
                    value->type == JSON_ARRAY ? &entries[stop] : value;
                if (entry->type != JSON_STRING ||
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

/* Generation state for one of a request's n choices. */
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
} server_choice;

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

/* Queue one SSE event on a generating connection. */
static void connection_send_sse(server_connection *conn, const char *payload,
                                size_t payload_length) {
    connection_send(conn, "data: ", 6);
    connection_send(conn, payload, payload_length);
    connection_send(conn, "\n\n", 2);
}

static void connection_send_chunk(server_connection *conn, uint64_t id,
                                  long long created, uint32_t choice,
                                  const char *delta, size_t delta_length,
                                  const char *finish_reason) {
    string_builder chunk = {0};
    sb_append_format(&chunk,
                     "{\"id\":\"cmpl-%llu\",\"object\":\"text_completion\","
                     "\"created\":%lld,\"model\":\"%s\",\"choices\":[{"
                     "\"text\":",
                     (unsigned long long)id, created, SERVER_MODEL_ID);
    sb_append_json_string(&chunk, delta, delta_length);
    if (finish_reason != NULL) {
        sb_append_format(&chunk,
                         ",\"index\":%u,\"logprobs\":null,"
                         "\"finish_reason\":\"%s\"}]}",
                         choice, finish_reason);
    } else {
        sb_append_format(&chunk,
                         ",\"index\":%u,\"logprobs\":null,"
                         "\"finish_reason\":null}]}",
                         choice);
    }
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
    for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
        if (gen->choices[choice].finish_reason == NULL) {
            gen->choices[choice].finish_reason = "length";
        }
    }
    if (conn == NULL) {
        generation_release(gen);
        return;
    }
    if (gen->request.stream) {
        for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
            server_choice *state = &gen->choices[choice];
            if (state->text.length > state->emitted) {
                connection_send_chunk(conn, gen->id, gen->created, choice,
                                      state->text.data + state->emitted,
                                      state->text.length - state->emitted,
                                      NULL);
            }
            connection_send_chunk(conn, gen->id, gen->created, choice, "", 0,
                                  state->finish_reason);
        }
        connection_send_sse(conn, "[DONE]", 6);
        conn->phase = CONNECTION_DRAINING;
    } else {
        string_builder response = {0};
        uint32_t total_completion = 0;
        sb_append_format(&response,
                         "{\"id\":\"cmpl-%llu\","
                         "\"object\":\"text_completion\","
                         "\"created\":%lld,\"model\":\"%s\","
                         "\"choices\":[",
                         (unsigned long long)gen->id, gen->created,
                         SERVER_MODEL_ID);
        for (uint32_t choice = 0; choice < gen->choice_count; ++choice) {
            server_choice *state = &gen->choices[choice];
            total_completion += state->completion_tokens;
            sb_append_format(&response, "%s{\"text\":",
                             choice == 0 ? "" : ",");
            sb_append_json_string(&response,
                                  state->text.data != NULL ? state->text.data
                                                           : "",
                                  state->text.length);
            sb_append_format(&response,
                             ",\"index\":%u,\"logprobs\":null,"
                             "\"finish_reason\":\"%s\"}",
                             choice, state->finish_reason);
        }
        sb_append_format(&response,
                         "],\"usage\":{\"prompt_tokens\":%zu,"
                         "\"completion_tokens\":%u,"
                         "\"total_tokens\":%zu}}",
                         gen->tokens.count, total_completion,
                         gen->tokens.count + total_completion);
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
    json_value body = {0};
    completion_request request = {0};
    char message[256];
    kipp_error error = {0};

    if (!json_parse(conn->in.data + conn->header_end, conn->body_length,
                    &body)) {
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "request body is not valid JSON");
        return -1;
    }
    if (parse_completion_request(&body, &request, message,
                                 sizeof(message)) != 0) {
        json_free_value(&body);
        completion_request_free(&request);
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "%s", message);
        return -1;
    }
    json_free_value(&body);

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
    if (gen->tokens.count + gen->request.max_tokens > KIPP_CONTEXT_LENGTH) {
        connection_error(conn, 400, "Bad Request", "invalid_request_error",
                         "prompt (%zu tokens) plus max_tokens (%u) exceeds "
                         "the %u-token context",
                         gen->tokens.count, gen->request.max_tokens,
                         KIPP_CONTEXT_LENGTH);
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
        if (capacity > KIPP_CONTEXT_LENGTH) {
            capacity = KIPP_CONTEXT_LENGTH;
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

    gen->id = ++completion_counter;
    gen->created = (long long)time(NULL);
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
            if (kipp_sample(state->logits, KIPP_VOCAB_SIZE,
                            gen->request.temperature, gen->request.top_p,
                            &state->rng, &token, &error) != 0) {
                generation_fail(gen, error.message);
                break;
            }
            if (token == KIPP_EOS_TOKEN_ID) {
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
            if (gen->request.stream && gen->conn != NULL) {
                size_t safe = state->text.length;
                if (state->finish_reason == NULL && gen->longest_stop > 0) {
                    safe = state->text.length > gen->longest_stop - 1
                               ? state->text.length - (gen->longest_stop - 1)
                               : 0;
                }
                if (safe > state->emitted) {
                    connection_send_chunk(gen->conn, gen->id, gen->created,
                                          choice,
                                          state->text.data + state->emitted,
                                          safe - state->emitted, NULL);
                    state->emitted = safe;
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

static const char *find_header_value(const char *headers, const char *name) {
    size_t name_length = strlen(name);
    const char *line = headers;
    while (line != NULL && *line != '\0') {
        if (strncasecmp(line, name, name_length) == 0 &&
            line[name_length] == ':') {
            const char *value = line + name_length + 1;
            while (*value == ' ' || *value == '\t') {
                ++value;
            }
            return value;
        }
        line = strstr(line, "\r\n");
        if (line != NULL) {
            line += 2;
        }
    }
    return NULL;
}

/* Route a fully-read request; completions move to WAITING for admission. */
static void connection_route(server_connection *conn) {
    if (strcmp(conn->method, "GET") == 0 &&
        strcmp(conn->path, "/healthz") == 0) {
        connection_respond(conn, 200, "OK", "{\"status\":\"ok\"}");
    } else if (strcmp(conn->method, "GET") == 0 &&
               strcmp(conn->path, "/v1/models") == 0) {
        connection_respond(conn, 200, "OK",
                           "{\"object\":\"list\",\"data\":[{\"id\":\""
                           SERVER_MODEL_ID "\",\"object\":\"model\","
                           "\"owned_by\":\"kipp\"}]}");
    } else if (strcmp(conn->method, "POST") == 0 &&
               strcmp(conn->path, "/v1/completions") == 0) {
        conn->phase = CONNECTION_WAITING;
        conn->arrival = ++arrival_counter;
    } else if (strcmp(conn->path, "/healthz") == 0 ||
               strcmp(conn->path, "/v1/models") == 0 ||
               strcmp(conn->path, "/v1/completions") == 0) {
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
            find_header_value(conn->in.data, "Content-Length");
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
            "[--port N]\n"
            "Serves GET /healthz, GET /v1/models, and POST /v1/completions "
            "on 127.0.0.1.\n"
            "Concurrent requests decode together through batched "
            "evaluation.\n",
            program);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    kipp_backend_kind backend = KIPP_BACKEND_CPU;
    unsigned long port = SERVER_DEFAULT_PORT;
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
            KIPP_VERSION, SERVER_MODEL_ID, port, kipp_backend_name(backend));

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
