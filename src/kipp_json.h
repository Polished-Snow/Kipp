#ifndef KIPP_JSON_H
#define KIPP_JSON_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Minimal strict JSON parser for the server's request bodies. Parses a
 * complete document (no trailing bytes), rejects invalid UTF-16 escapes,
 * non-finite numbers, and nesting deeper than KIPP_JSON_DEPTH_LIMIT. The
 * parser allocates; release any successfully parsed value with
 * kipp_json_free. Kept separate from the HTTP and serving layers so it can
 * be unit-tested and fuzzed in isolation.
 */

#define KIPP_JSON_DEPTH_LIMIT 16u

typedef enum {
    KIPP_JSON_NULL,
    KIPP_JSON_BOOL,
    KIPP_JSON_NUMBER,
    KIPP_JSON_STRING,
    KIPP_JSON_ARRAY,
    KIPP_JSON_OBJECT
} kipp_json_type;

typedef struct kipp_json_value kipp_json_value;

struct kipp_json_value {
    kipp_json_type type;
    bool boolean;
    double number;
    char *string;
    char **keys; /* object member names; NULL for arrays */
    kipp_json_value *items;
    size_t count;
};

/*
 * Parse `length` bytes as one JSON document. Returns true and fills *value
 * on success; returns false and leaves *value zeroed on any error.
 */
bool kipp_json_parse(const char *text, size_t length, kipp_json_value *value);
void kipp_json_free(kipp_json_value *value);

#endif
