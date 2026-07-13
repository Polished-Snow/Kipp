#ifndef KIPP_CUDA_H
#define KIPP_CUDA_H

#include "kipp_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

const kipp_backend_ops *kipp_cuda_backend_operations(void);

/*
 * Run model-free CUDA kernel comparisons. Returns zero only when a real CUDA
 * device executed every operator test successfully.
 */
int kipp_cuda_run_operator_tests(kipp_error *error);
const char *kipp_cuda_device_name(void);

#ifdef __cplusplus
}
#endif

#endif
