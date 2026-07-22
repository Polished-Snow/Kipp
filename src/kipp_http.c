#include "kipp_http.h"

#include <stddef.h>
#include <string.h>
#include <strings.h>

const char *kipp_http_header_value(const char *headers, const char *name) {
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
