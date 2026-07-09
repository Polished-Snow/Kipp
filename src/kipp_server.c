/*
 * Kipp API server.
 *
 * This file will expose the native engine through deliberately small
 * OpenAI- and Anthropic-compatible HTTP interfaces. It must not introduce a
 * Python runtime into the inference path.
 */

const char *kipp_server_scaffold_status(void) {
    return "Kipp API server is not implemented";
}
