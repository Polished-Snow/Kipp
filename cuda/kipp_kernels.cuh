#ifndef KIPP_CUDA_KERNELS_CUH
#define KIPP_CUDA_KERNELS_CUH

#include <cuda_runtime_api.h>

#include <stdint.h>

cudaError_t kipp_cuda_launch_bf16_roundtrip(
    const float *input, uint16_t *bits, float *output, uint32_t count,
    cudaStream_t stream);
cudaError_t kipp_cuda_launch_embed(const uint16_t *embedding, float *output,
                                   uint32_t token, cudaStream_t stream);
cudaError_t kipp_cuda_launch_rms_norm(
    const float *input, const uint16_t *weight, float *output, uint32_t length,
    float epsilon, cudaStream_t stream);
cudaError_t kipp_cuda_launch_head_norm(float *states, const uint16_t *weight,
                                       uint32_t head_count, float epsilon,
                                       cudaStream_t stream);
cudaError_t kipp_cuda_launch_matvec(const uint16_t *weight,
                                    const float *input, float *output,
                                    uint32_t rows, uint32_t columns,
                                    cudaStream_t stream);
cudaError_t kipp_cuda_launch_rope(float *states, uint32_t head_count,
                                  uint32_t position, float theta,
                                  cudaStream_t stream);
cudaError_t kipp_cuda_launch_kv_write(
    const float *key, const float *value, uint16_t *key_cache,
    uint16_t *value_cache, uint32_t layer, uint32_t position,
    uint32_t capacity, cudaStream_t stream);
cudaError_t kipp_cuda_launch_cached_gqa(
    const float *query, const uint16_t *key_cache,
    const uint16_t *value_cache, float *scores, float *output, uint32_t layer,
    uint32_t position, uint32_t capacity, cudaStream_t stream);
cudaError_t kipp_cuda_launch_residual(float *residual, const float *addition,
                                      uint32_t count, cudaStream_t stream);
cudaError_t kipp_cuda_launch_swiglu(float *gate, const float *up,
                                    uint32_t count, cudaStream_t stream);

#endif
