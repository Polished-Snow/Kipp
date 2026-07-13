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

#endif
