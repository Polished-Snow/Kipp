#ifndef KIPP_METAL_H
#define KIPP_METAL_H

#include "kipp_backend.h"

const kipp_backend_ops *kipp_metal_backend_operations(void);

/*
 * Run model-free Metal kernel comparisons. Returns zero only when a real Metal
 * device compiled and executed every operator test successfully.
 */
int kipp_metal_run_operator_tests(kipp_error *error);
const char *kipp_metal_device_name(void);

/* Test hook: reverse a Metal backend session's KV page table before eval. */
int kipp_metal_test_scramble_session(void *backendSession);

/*
 * Test hook: cap the decode split-K count on a Metal backend model.
 * 1 forces the legacy single-split path regardless of context length;
 * 0 restores automatic position-derived splitting.
 */
int kipp_metal_test_set_ksplit_cap(void *backendModel, uint32_t cap);

#endif
