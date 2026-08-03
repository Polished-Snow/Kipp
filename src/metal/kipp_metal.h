#ifndef KIPP_METAL_H
#define KIPP_METAL_H

#include "kipp_backend.h"

const kipp_backend_ops *kipp_metal_backend_operations(void);

/*
 * Run model-free Metal kernel comparisons. Returns zero only when a real Metal
 * device compiled and executed every operator test successfully.
 */
int kipp_metal_run_operator_tests(kipp_error *error);

/*
 * Isolated projection-matmul micro-benchmark on the live pipelines: every
 * Qwen3-4B projection shape, per weight scheme, with rotating vs reused
 * weight buffers to separate traffic-bound from compute-bound behaviour.
 * A measurement instrument, not a correctness gate.
 */
int kipp_metal_run_mm_bench(kipp_error *error);

/*
 * Isolated prefill-attention micro-benchmark: replays the chunked-prefill
 * dispatch shape over a synthetic paged cache at several context lengths,
 * rotating vs reusing layer slices to separate DRAM-bound from
 * issue/latency-bound behaviour. A measurement instrument, not a gate.
 */
int kipp_metal_run_fa_bench(kipp_error *error);
const char *kipp_metal_device_name(void);

/* Test hook: reverse a Metal backend session's KV page table before eval. */
int kipp_metal_test_scramble_session(void *backendSession);

/*
 * Test hook: cap the decode split-K count on a Metal backend model.
 * 1 forces the legacy single-split path regardless of context length;
 * 0 restores automatic position-derived splitting.
 */
int kipp_metal_test_set_ksplit_cap(void *backendModel, uint32_t cap);

/*
 * Test hook: pin a Metal backend model to the vector/streaming kernels even
 * when a round is wide enough for the simdgroup-matrix path. The vector path is
 * the one anchored to the CPU oracle by the short-sequence gates, so forcing it
 * turns "do the matrix kernels agree with the reference?" into a GPU-only
 * comparison at any length. 1 forces vectors; 0 restores automatic selection.
 */
int kipp_metal_test_force_vector(void *backendModel, int force);

#endif
