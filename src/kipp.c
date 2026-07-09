/*
 * Kipp core inference engine.
 *
 * This file will own model loading, tensor operations, the readable reference
 * forward pass, and dispatch into isolated hardware backends once a target
 * model family is selected.
 */

const char *kipp_scaffold_status(void) {
    return "Kipp inference is not implemented";
}
