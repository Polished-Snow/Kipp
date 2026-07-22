#ifndef KIPP_HTTP_H
#define KIPP_HTTP_H

/*
 * Minimal HTTP header helpers, separated from the serving layer so parsing
 * of untrusted request bytes can be unit-tested and fuzzed in isolation.
 */

/*
 * Find the value of a header (case-insensitive name match) in a
 * NUL-terminated header block whose lines are separated by CRLF. Returns a
 * pointer into `headers` at the first non-blank value byte, or NULL when
 * the header is absent.
 */
const char *kipp_http_header_value(const char *headers, const char *name);

#endif
