#include "kipp_json.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* A minimal growable byte buffer, local to this module. */
typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    bool failed;
} json_builder;

static void json_builder_free(json_builder *builder) {
    free(builder->data);
    memset(builder, 0, sizeof(*builder));
}

static void json_builder_append(json_builder *builder, const char *bytes,
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

void kipp_json_free(kipp_json_value *value) {
    if (value == NULL) {
        return;
    }
    free(value->string);
    for (size_t index = 0; index < value->count; ++index) {
        if (value->keys != NULL) {
            free(value->keys[index]);
        }
        kipp_json_free(&value->items[index]);
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
    json_builder builder = {0};
    while (cursor->offset < cursor->length) {
        char c = cursor->text[cursor->offset];
        if (c == '"') {
            ++cursor->offset;
            if (builder.failed) {
                json_builder_free(&builder);
                return false;
            }
            *output = builder.data != NULL ? builder.data : calloc(1, 1);
            return *output != NULL;
        }
        if ((unsigned char)c < 0x20) {
            break;
        }
        if (c != '\\') {
            json_builder_append(&builder, &cursor->text[cursor->offset], 1);
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
            json_builder_append(&builder, &escape, 1);
            continue;
        case 'b':
            json_builder_append(&builder, "\b", 1);
            continue;
        case 'f':
            json_builder_append(&builder, "\f", 1);
            continue;
        case 'n':
            json_builder_append(&builder, "\n", 1);
            continue;
        case 'r':
            json_builder_append(&builder, "\r", 1);
            continue;
        case 't':
            json_builder_append(&builder, "\t", 1);
            continue;
        case 'u':
            if (!json_parse_hex4(cursor, &codepoint)) {
                json_builder_free(&builder);
                return false;
            }
            if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                uint32_t low;
                if (!json_literal(cursor, "\\u") ||
                    !json_parse_hex4(cursor, &low) || low < 0xdc00 ||
                    low > 0xdfff) {
                    json_builder_free(&builder);
                    return false;
                }
                codepoint = 0x10000 +
                            ((codepoint - 0xd800) << 10) + (low - 0xdc00);
            } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                json_builder_free(&builder);
                return false;
            }
            json_builder_append(&builder, encoded,
                                json_utf8_encode(encoded, codepoint));
            continue;
        default:
            json_builder_free(&builder);
            return false;
        }
    }
    json_builder_free(&builder);
    return false;
}

static bool json_parse_value(json_cursor *cursor, kipp_json_value *value,
                             unsigned depth);

static bool json_parse_collection(json_cursor *cursor, kipp_json_value *value,
                                  bool is_object, unsigned depth) {
    char open = is_object ? '{' : '[';
    char close = is_object ? '}' : ']';
    if (cursor->text[cursor->offset] != open) {
        return false;
    }
    ++cursor->offset;
    value->type = is_object ? KIPP_JSON_OBJECT : KIPP_JSON_ARRAY;
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
            kipp_json_value *items =
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
                goto fail_slot;
            }
            json_skip_space(cursor);
            if (cursor->offset >= cursor->length ||
                cursor->text[cursor->offset] != ':') {
                goto fail_slot;
            }
            ++cursor->offset;
        }
        json_skip_space(cursor);
        if (cursor->offset >= cursor->length ||
            !json_parse_value(cursor, &value->items[value->count],
                              depth + 1)) {
            goto fail_slot;
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

    /*
     * Failure with an in-flight slot: value->count does not yet cover it,
     * so the caller's recursive free would never reach the parsed key or
     * the failed child's partial subtree; release them here.
     */
fail_slot:
    if (is_object && value->keys != NULL &&
        value->keys[value->count] != NULL) {
        free(value->keys[value->count]);
        value->keys[value->count] = NULL;
    }
    kipp_json_free(&value->items[value->count]);
    return false;
}

static bool json_parse_value(json_cursor *cursor, kipp_json_value *value,
                             unsigned depth) {
    if (depth > KIPP_JSON_DEPTH_LIMIT) {
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
        value->type = KIPP_JSON_STRING;
        return json_parse_string(cursor, &value->string);
    }
    if (json_literal(cursor, "true")) {
        value->type = KIPP_JSON_BOOL;
        value->boolean = true;
        return true;
    }
    if (json_literal(cursor, "false")) {
        value->type = KIPP_JSON_BOOL;
        value->boolean = false;
        return true;
    }
    if (json_literal(cursor, "null")) {
        value->type = KIPP_JSON_NULL;
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
        value->type = KIPP_JSON_NUMBER;
        return true;
    }
    return false;
}

bool kipp_json_parse(const char *text, size_t length,
                     kipp_json_value *value) {
    json_cursor cursor = {text, length, 0};
    memset(value, 0, sizeof(*value));
    if (!json_parse_value(&cursor, value, 0)) {
        kipp_json_free(value);
        return false;
    }
    json_skip_space(&cursor);
    if (cursor.offset != length) {
        kipp_json_free(value);
        return false;
    }
    return true;
}
