#include "kipp_kernels.cuh"

#include "kipp.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace {

constexpr uint32_t kThreads = 256u;

__device__ __forceinline__ float bf16_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ __forceinline__ uint16_t float_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

__global__ void bf16_roundtrip_kernel(const float *input, uint16_t *bits,
                                      float *output, uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        uint16_t rounded = float_to_bf16(input[index]);
        bits[index] = rounded;
        output[index] = bf16_to_float(rounded);
    }
}

__global__ void embed_kernel(const uint16_t *embedding, float *output,
                             uint32_t token, uint32_t length) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        size_t source = static_cast<size_t>(token) * length + index;
        output[index] = bf16_to_float(embedding[source]);
    }
}

__global__ void rms_norm_kernel(const float *input, const uint16_t *weight,
                                float *output, uint32_t length,
                                float epsilon) {
    __shared__ float reduction[kThreads];
    uint32_t thread = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t index = thread; index < length; index += blockDim.x) {
        sum += input[index] * input[index];
    }
    reduction[thread] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }
    float scale = rsqrtf(reduction[0] / static_cast<float>(length) + epsilon);
    for (uint32_t index = thread; index < length; index += blockDim.x) {
        output[index] = input[index] * scale * bf16_to_float(weight[index]);
    }
}

__global__ void head_norm_kernel(float *states, const uint16_t *weight,
                                 uint32_t head_count, float epsilon) {
    __shared__ float reduction[KIPP_ATTENTION_HEAD_DIM];
    uint32_t head = blockIdx.x;
    uint32_t dimension = threadIdx.x;
    if (head >= head_count || dimension >= KIPP_ATTENTION_HEAD_DIM) {
        return;
    }
    float *values =
        states + static_cast<size_t>(head) * KIPP_ATTENTION_HEAD_DIM;
    float value = values[dimension];
    reduction[dimension] = value * value;
    __syncthreads();
    for (uint32_t stride = KIPP_ATTENTION_HEAD_DIM / 2; stride != 0;
         stride /= 2) {
        if (dimension < stride) {
            reduction[dimension] += reduction[dimension + stride];
        }
        __syncthreads();
    }
    float scale =
        rsqrtf(reduction[0] / static_cast<float>(KIPP_ATTENTION_HEAD_DIM) +
               epsilon);
    values[dimension] = value * scale * bf16_to_float(weight[dimension]);
}

__global__ void matvec_kernel(const uint16_t *weight, const float *input,
                              float *output, uint32_t rows,
                              uint32_t columns) {
    __shared__ float reduction[kThreads];
    uint32_t row = blockIdx.x;
    uint32_t thread = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const uint16_t *row_weights =
        weight + static_cast<size_t>(row) * columns;
    float sum = 0.0f;
    for (uint32_t column = thread; column < columns; column += blockDim.x) {
        sum += bf16_to_float(row_weights[column]) * input[column];
    }
    reduction[thread] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0) {
        output[row] = reduction[0];
    }
}

__global__ void rope_kernel(float *states, uint32_t head_count,
                            uint32_t position, float theta) {
    uint32_t head = blockIdx.x;
    uint32_t index = threadIdx.x;
    constexpr uint32_t half = KIPP_ATTENTION_HEAD_DIM / 2;
    if (head >= head_count || index >= half) {
        return;
    }
    float *values =
        states + static_cast<size_t>(head) * KIPP_ATTENTION_HEAD_DIM;
    float frequency =
        powf(theta, -(2.0f * static_cast<float>(index)) /
                        static_cast<float>(KIPP_ATTENTION_HEAD_DIM));
    float angle = static_cast<float>(position) * frequency;
    float cosine = cosf(angle);
    float sine = sinf(angle);
    float first = values[index];
    float second = values[index + half];
    values[index] = first * cosine - second * sine;
    values[index + half] = second * cosine + first * sine;
}

__global__ void kv_write_kernel(const float *key, const float *value,
                                uint16_t *key_cache, uint16_t *value_cache,
                                uint32_t layer, uint32_t position,
                                uint32_t capacity) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t values_per_token =
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    if (index < values_per_token) {
        size_t cache_index =
            (static_cast<size_t>(layer) * capacity + position) *
                values_per_token +
            index;
        key_cache[cache_index] = float_to_bf16(key[index]);
        value_cache[cache_index] = float_to_bf16(value[index]);
    }
}

__global__ void cached_gqa_kernel(
    const float *query, const uint16_t *key_cache,
    const uint16_t *value_cache, float *scores, float *output, uint32_t layer,
    uint32_t position, uint32_t capacity, uint32_t query_head_count) {
    __shared__ float reduction[KIPP_ATTENTION_HEAD_DIM];
    uint32_t query_head = blockIdx.x;
    uint32_t dimension = threadIdx.x;
    if (query_head >= query_head_count ||
        dimension >= KIPP_ATTENTION_HEAD_DIM) {
        return;
    }

    constexpr uint32_t values_per_token =
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    uint32_t queries_per_kv =
        query_head_count / KIPP_ATTENTION_HEAD_COUNT_KV;
    uint32_t kv_head = query_head / queries_per_kv;
    const float *query_values =
        query + static_cast<size_t>(query_head) * KIPP_ATTENTION_HEAD_DIM;
    float *head_scores =
        scores + static_cast<size_t>(query_head) * capacity;
    float maximum = -3.402823466e+38F;
    float scale = rsqrtf(static_cast<float>(KIPP_ATTENTION_HEAD_DIM));

    for (uint32_t source = 0; source <= position; ++source) {
        size_t base =
            (static_cast<size_t>(layer) * capacity + source) *
                values_per_token +
            static_cast<size_t>(kv_head) * KIPP_ATTENTION_HEAD_DIM;
        reduction[dimension] =
            query_values[dimension] *
            bf16_to_float(key_cache[base + dimension]);
        __syncthreads();
        for (uint32_t stride = KIPP_ATTENTION_HEAD_DIM / 2; stride != 0;
             stride /= 2) {
            if (dimension < stride) {
                reduction[dimension] += reduction[dimension + stride];
            }
            __syncthreads();
        }
        if (dimension == 0) {
            float score = reduction[0] * scale;
            head_scores[source] = score;
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }

    if (dimension == 0) {
        float denominator = 0.0f;
        for (uint32_t source = 0; source <= position; ++source) {
            float value = expf(head_scores[source] - maximum);
            head_scores[source] = value;
            denominator += value;
        }
        float inverse = 1.0f / denominator;
        for (uint32_t source = 0; source <= position; ++source) {
            head_scores[source] *= inverse;
        }
    }
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t source = 0; source <= position; ++source) {
        size_t base =
            (static_cast<size_t>(layer) * capacity + source) *
                values_per_token +
            static_cast<size_t>(kv_head) * KIPP_ATTENTION_HEAD_DIM;
        sum += head_scores[source] *
               bf16_to_float(value_cache[base + dimension]);
    }
    output[static_cast<size_t>(query_head) * KIPP_ATTENTION_HEAD_DIM +
           dimension] = sum;
}

__global__ void residual_kernel(float *residual, const float *addition,
                                uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        residual[index] += addition[index];
    }
}

__global__ void swiglu_kernel(float *gate, const float *up, uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = gate[index];
        gate[index] = (value / (1.0f + expf(-value))) * up[index];
    }
}

uint32_t block_count(uint32_t count) {
    return (count + kThreads - 1u) / kThreads;
}

}  // namespace

cudaError_t kipp_cuda_launch_bf16_roundtrip(
    const float *input, uint16_t *bits, float *output, uint32_t count,
    cudaStream_t stream) {
    bf16_roundtrip_kernel<<<block_count(count), kThreads, 0, stream>>>(
        input, bits, output, count);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_embed(const uint16_t *embedding, float *output,
                                   uint32_t token, uint32_t length,
                                   cudaStream_t stream) {
    embed_kernel<<<block_count(length), kThreads, 0, stream>>>(
        embedding, output, token, length);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_rms_norm(
    const float *input, const uint16_t *weight, float *output, uint32_t length,
    float epsilon, cudaStream_t stream) {
    rms_norm_kernel<<<1, kThreads, 0, stream>>>(input, weight, output, length,
                                               epsilon);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_head_norm(float *states, const uint16_t *weight,
                                       uint32_t head_count, float epsilon,
                                       cudaStream_t stream) {
    head_norm_kernel<<<head_count, KIPP_ATTENTION_HEAD_DIM, 0, stream>>>(
        states, weight, head_count, epsilon);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_matvec(const uint16_t *weight,
                                    const float *input, float *output,
                                    uint32_t rows, uint32_t columns,
                                    cudaStream_t stream) {
    matvec_kernel<<<rows, kThreads, 0, stream>>>(weight, input, output, rows,
                                                columns);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_rope(float *states, uint32_t head_count,
                                  uint32_t position, float theta,
                                  cudaStream_t stream) {
    rope_kernel<<<head_count, KIPP_ATTENTION_HEAD_DIM / 2, 0, stream>>>(
        states, head_count, position, theta);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_kv_write(
    const float *key, const float *value, uint16_t *key_cache,
    uint16_t *value_cache, uint32_t layer, uint32_t position,
    uint32_t capacity, cudaStream_t stream) {
    constexpr uint32_t count =
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    kv_write_kernel<<<block_count(count), kThreads, 0, stream>>>(
        key, value, key_cache, value_cache, layer, position, capacity);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_cached_gqa(
    const float *query, const uint16_t *key_cache,
    const uint16_t *value_cache, float *scores, float *output, uint32_t layer,
    uint32_t position, uint32_t capacity, uint32_t query_head_count,
    cudaStream_t stream) {
    cached_gqa_kernel<<<query_head_count, KIPP_ATTENTION_HEAD_DIM, 0,
                        stream>>>(query, key_cache, value_cache, scores, output,
                                  layer, position, capacity,
                                  query_head_count);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_residual(float *residual, const float *addition,
                                      uint32_t count, cudaStream_t stream) {
    residual_kernel<<<block_count(count), kThreads, 0, stream>>>(
        residual, addition, count);
    return cudaGetLastError();
}

cudaError_t kipp_cuda_launch_swiglu(float *gate, const float *up,
                                    uint32_t count, cudaStream_t stream) {
    swiglu_kernel<<<block_count(count), kThreads, 0, stream>>>(gate, up, count);
    return cudaGetLastError();
}
