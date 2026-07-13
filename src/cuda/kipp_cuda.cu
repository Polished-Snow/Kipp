#include "cuda/kipp_cuda.h"

#include "../../cuda/kipp_kernels.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIPP_CUDA_ORACLE_CAPACITY 256u
#define KIPP_CUDA_RMS_EPSILON 1.0e-6f
#define KIPP_CUDA_ROPE_THETA 1000000.0f

typedef struct cuda_backend_model cuda_backend_model;

typedef struct {
    cuda_backend_model *model;
    uint32_t capacity;
    uint32_t length;
    cudaStream_t stream;
    uint16_t *key_cache;
    uint16_t *value_cache;
    float *x;
    float *normalized;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *projection;
    float *gate;
    float *up;
    float *scores;
    float *logits;
} cuda_backend_session;

struct cuda_backend_model {
    int device;
    const kipp_model_view *view;
    uintptr_t host_span_start;
    size_t weight_bytes;
    uint8_t *weights;
    cuda_backend_session *oracle_session;
};

static int cuda_fail(kipp_error *error, kipp_error_code code,
                     const char *format, ...) {
    if (error != NULL) {
        va_list arguments;
        error->code = code;
        va_start(arguments, format);
        (void)vsnprintf(error->message, sizeof(error->message), format,
                        arguments);
        va_end(arguments);
    }
    return -1;
}

static void cuda_clear_error(kipp_error *error) {
    if (error != NULL) {
        error->code = KIPP_OK;
        error->message[0] = '\0';
    }
}

static kipp_error_code cuda_error_code(cudaError_t status) {
    switch (status) {
    case cudaErrorMemoryAllocation:
        return KIPP_ERROR_MEMORY;
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorInitializationError:
    case cudaErrorInvalidDevice:
        return KIPP_ERROR_UNSUPPORTED;
    default:
        return KIPP_ERROR_INTERNAL;
    }
}

static int cuda_fail_status(kipp_error *error, const char *operation,
                            cudaError_t status) {
    return cuda_fail(error, cuda_error_code(status), "CUDA %s failed: %s",
                     operation, cudaGetErrorString(status));
}

template <typename T>
static int cuda_allocate(T **pointer, size_t count, const char *name,
                         kipp_error *error) {
    *pointer = NULL;
    if (count > SIZE_MAX / sizeof(T)) {
        return cuda_fail(error, KIPP_ERROR_MEMORY,
                         "CUDA %s allocation size overflows", name);
    }
    cudaError_t status =
        cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(T));
    if (status != cudaSuccess) {
        return cuda_fail_status(error, name, status);
    }
    return 0;
}

static void cuda_session_delete(cuda_backend_session *session) {
    if (session == NULL) {
        return;
    }
    if (session->model != NULL) {
        (void)cudaSetDevice(session->model->device);
    }
    if (session->stream != NULL) {
        (void)cudaStreamSynchronize(session->stream);
    }
    (void)cudaFree(session->key_cache);
    (void)cudaFree(session->value_cache);
    (void)cudaFree(session->x);
    (void)cudaFree(session->normalized);
    (void)cudaFree(session->query);
    (void)cudaFree(session->key);
    (void)cudaFree(session->value);
    (void)cudaFree(session->attention);
    (void)cudaFree(session->projection);
    (void)cudaFree(session->gate);
    (void)cudaFree(session->up);
    (void)cudaFree(session->scores);
    (void)cudaFree(session->logits);
    if (session->stream != NULL) {
        (void)cudaStreamDestroy(session->stream);
    }
    free(session);
}

static uint64_t cuda_session_device_bytes(uint32_t capacity) {
    uint64_t slab_elements =
        static_cast<uint64_t>(KIPP_BLOCK_COUNT) * capacity *
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    uint64_t score_elements =
        static_cast<uint64_t>(KIPP_ATTENTION_HEAD_COUNT) * capacity;
    uint64_t float_elements =
        3u * KIPP_EMBEDDING_LENGTH +
        2u * KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM +
        2u * KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM +
        2u * KIPP_FEED_FORWARD_LENGTH + score_elements + KIPP_VOCAB_SIZE;
    return 2u * slab_elements * sizeof(uint16_t) +
           float_elements * sizeof(float);
}

static cuda_backend_session *
cuda_session_new(cuda_backend_model *model, uint32_t capacity,
                 kipp_error *error) {
    if (capacity == 0 || capacity > KIPP_CONTEXT_LENGTH) {
        cuda_fail(error, KIPP_ERROR_RANGE,
                  "CUDA session capacity must be between 1 and %u",
                  KIPP_CONTEXT_LENGTH);
        return NULL;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (memory_status != cudaSuccess) {
        cuda_fail_status(error, "session memory capacity query", memory_status);
        return NULL;
    }
    uint64_t required_bytes = cuda_session_device_bytes(capacity);
    if (required_bytes > free_bytes) {
        cuda_fail(error, KIPP_ERROR_MEMORY,
                  "CUDA session requires %llu device bytes but only %zu of "
                  "%zu bytes are free",
                  static_cast<unsigned long long>(required_bytes), free_bytes,
                  total_bytes);
        return NULL;
    }
    size_t slab_elements =
        static_cast<size_t>(KIPP_BLOCK_COUNT) * capacity *
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    size_t score_elements =
        static_cast<size_t>(KIPP_ATTENTION_HEAD_COUNT) * capacity;

    cuda_backend_session *session =
        static_cast<cuda_backend_session *>(calloc(1, sizeof(*session)));
    if (session == NULL) {
        cuda_fail(error, KIPP_ERROR_MEMORY,
                  "unable to allocate CUDA session metadata");
        return NULL;
    }
    session->model = model;
    session->capacity = capacity;

    cudaError_t status =
        cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        cuda_fail_status(error, "stream creation", status);
        cuda_session_delete(session);
        return NULL;
    }

    if (cuda_allocate(&session->key_cache, slab_elements, "key cache", error) !=
            0 ||
        cuda_allocate(&session->value_cache, slab_elements, "value cache",
                      error) != 0 ||
        cuda_allocate(&session->x, KIPP_EMBEDDING_LENGTH, "residual scratch",
                      error) != 0 ||
        cuda_allocate(&session->normalized, KIPP_EMBEDDING_LENGTH,
                      "normalization scratch", error) != 0 ||
        cuda_allocate(&session->query,
                      KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM,
                      "query scratch", error) != 0 ||
        cuda_allocate(&session->key,
                      KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
                      "key scratch", error) != 0 ||
        cuda_allocate(&session->value,
                      KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
                      "value scratch", error) != 0 ||
        cuda_allocate(&session->attention,
                      KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM,
                      "attention scratch", error) != 0 ||
        cuda_allocate(&session->projection, KIPP_EMBEDDING_LENGTH,
                      "projection scratch", error) != 0 ||
        cuda_allocate(&session->gate, KIPP_FEED_FORWARD_LENGTH, "gate scratch",
                      error) != 0 ||
        cuda_allocate(&session->up, KIPP_FEED_FORWARD_LENGTH, "up scratch",
                      error) != 0 ||
        cuda_allocate(&session->scores, score_elements, "attention scores",
                      error) != 0 ||
        cuda_allocate(&session->logits, KIPP_VOCAB_SIZE, "logits", error) !=
            0) {
        cuda_session_delete(session);
        return NULL;
    }
    return session;
}

static int cuda_include_tensor(const kipp_model_view *view,
                               const kipp_tensor_view *tensor,
                               uintptr_t mapping_start, uintptr_t mapping_end,
                               uintptr_t *span_start, uintptr_t *span_end,
                               kipp_error *error) {
    if (tensor->data == NULL || tensor->type != KIPP_TENSOR_BF16 ||
        tensor->byte_count == 0 || tensor->byte_count > SIZE_MAX) {
        return cuda_fail(error, KIPP_ERROR_FORMAT,
                         "CUDA tensor %s has invalid storage metadata",
                         tensor->name != NULL ? tensor->name : "(unnamed)");
    }
    uintptr_t tensor_start = reinterpret_cast<uintptr_t>(tensor->data);
    if (tensor_start < mapping_start || tensor_start >= mapping_end ||
        tensor->byte_count > mapping_end - tensor_start) {
        return cuda_fail(error, KIPP_ERROR_FORMAT,
                         "CUDA tensor %s lies outside the model mapping",
                         tensor->name != NULL ? tensor->name : "(unnamed)");
    }
    uintptr_t tensor_end =
        tensor_start + static_cast<uintptr_t>(tensor->byte_count);
    if (tensor_start < *span_start) {
        *span_start = tensor_start;
    }
    if (tensor_end > *span_end) {
        *span_end = tensor_end;
    }
    (void)view;
    return 0;
}

static int cuda_model_tensor_span(const kipp_model_view *view,
                                  uintptr_t *span_start, size_t *span_bytes,
                                  kipp_error *error) {
    if (view->mapping == NULL || view->mapping_size == 0 ||
        view->mapping_size > SIZE_MAX) {
        return cuda_fail(error, KIPP_ERROR_FORMAT,
                         "CUDA model mapping is invalid");
    }
    uintptr_t mapping_start = reinterpret_cast<uintptr_t>(view->mapping);
    if (view->mapping_size > UINTPTR_MAX - mapping_start) {
        return cuda_fail(error, KIPP_ERROR_MEMORY,
                         "CUDA model mapping address overflows");
    }
    uintptr_t mapping_end =
        mapping_start + static_cast<uintptr_t>(view->mapping_size);
    uintptr_t first = mapping_end;
    uintptr_t last = mapping_start;

#define INCLUDE_TENSOR(tensor)                                                   \
    do {                                                                         \
        if (cuda_include_tensor(view, &(tensor), mapping_start, mapping_end,     \
                                &first, &last, error) != 0) {                     \
            return -1;                                                           \
        }                                                                        \
    } while (0)

    INCLUDE_TENSOR(view->weights.token_embedding);
    for (uint32_t index = 0; index < KIPP_BLOCK_COUNT; ++index) {
        const kipp_qwen3_layer_weights *layer = &view->weights.layers[index];
        INCLUDE_TENSOR(layer->attention_norm);
        INCLUDE_TENSOR(layer->attention_q);
        INCLUDE_TENSOR(layer->attention_k);
        INCLUDE_TENSOR(layer->attention_v);
        INCLUDE_TENSOR(layer->attention_output);
        INCLUDE_TENSOR(layer->attention_q_norm);
        INCLUDE_TENSOR(layer->attention_k_norm);
        INCLUDE_TENSOR(layer->feed_forward_norm);
        INCLUDE_TENSOR(layer->feed_forward_gate);
        INCLUDE_TENSOR(layer->feed_forward_up);
        INCLUDE_TENSOR(layer->feed_forward_down);
    }
    INCLUDE_TENSOR(view->weights.output_norm);
#undef INCLUDE_TENSOR

    if (first >= last || last - first > SIZE_MAX) {
        return cuda_fail(error, KIPP_ERROR_MEMORY,
                         "CUDA tensor span size overflows");
    }
    *span_start = first;
    *span_bytes = static_cast<size_t>(last - first);
    return 0;
}

static const uint16_t *
cuda_tensor(const cuda_backend_model *model,
            const kipp_tensor_view *tensor) {
    size_t offset =
        static_cast<size_t>(reinterpret_cast<uintptr_t>(tensor->data) -
                            model->host_span_start);
    return reinterpret_cast<const uint16_t *>(model->weights + offset);
}

static int cuda_check_launch(cudaError_t status, const char *operation,
                             kipp_error *error) {
    if (status != cudaSuccess) {
        return cuda_fail_status(error, operation, status);
    }
    return 0;
}

static int cuda_encode_token(cuda_backend_model *model,
                             cuda_backend_session *session, uint32_t token,
                             uint32_t position, bool write_logits,
                             float *host_logits, kipp_error *error) {
    cudaStream_t stream = session->stream;
    const kipp_qwen3_weights *weights = &model->view->weights;
    if (cuda_check_launch(
            kipp_cuda_launch_embed(cuda_tensor(model, &weights->token_embedding),
                                   session->x, token, stream),
            "embedding launch", error) != 0) {
        return -1;
    }

    for (uint32_t layer_index = 0; layer_index < KIPP_BLOCK_COUNT;
         ++layer_index) {
        const kipp_qwen3_layer_weights *layer =
            &weights->layers[layer_index];
        if (cuda_check_launch(
                kipp_cuda_launch_rms_norm(
                    session->x, cuda_tensor(model, &layer->attention_norm),
                    session->normalized, KIPP_EMBEDDING_LENGTH,
                    KIPP_CUDA_RMS_EPSILON, stream),
                "attention RMSNorm launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &layer->attention_q),
                    session->normalized, session->query,
                    KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM,
                    KIPP_EMBEDDING_LENGTH, stream),
                "query projection launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &layer->attention_k),
                    session->normalized, session->key,
                    KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
                    KIPP_EMBEDDING_LENGTH, stream),
                "key projection launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &layer->attention_v),
                    session->normalized, session->value,
                    KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
                    KIPP_EMBEDDING_LENGTH, stream),
                "value projection launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_head_norm(
                    session->query,
                    cuda_tensor(model, &layer->attention_q_norm),
                    KIPP_ATTENTION_HEAD_COUNT, KIPP_CUDA_RMS_EPSILON, stream),
                "query head norm launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_head_norm(
                    session->key,
                    cuda_tensor(model, &layer->attention_k_norm),
                    KIPP_ATTENTION_HEAD_COUNT_KV, KIPP_CUDA_RMS_EPSILON,
                    stream),
                "key head norm launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_rope(session->query,
                                      KIPP_ATTENTION_HEAD_COUNT, position,
                                      KIPP_CUDA_ROPE_THETA, stream),
                "query RoPE launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_rope(session->key,
                                      KIPP_ATTENTION_HEAD_COUNT_KV, position,
                                      KIPP_CUDA_ROPE_THETA, stream),
                "key RoPE launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_kv_write(
                    session->key, session->value, session->key_cache,
                    session->value_cache, layer_index, position,
                    session->capacity, stream),
                "KV write launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_cached_gqa(
                    session->query, session->key_cache, session->value_cache,
                    session->scores, session->attention, layer_index, position,
                    session->capacity, stream),
                "cached GQA launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &layer->attention_output),
                    session->attention, session->projection,
                    KIPP_EMBEDDING_LENGTH,
                    KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM,
                    stream),
                "attention output launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_residual(
                    session->x, session->projection, KIPP_EMBEDDING_LENGTH,
                    stream),
                "attention residual launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_rms_norm(
                    session->x,
                    cuda_tensor(model, &layer->feed_forward_norm),
                    session->normalized, KIPP_EMBEDDING_LENGTH,
                    KIPP_CUDA_RMS_EPSILON, stream),
                "feed-forward RMSNorm launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &layer->feed_forward_gate),
                    session->normalized, session->gate,
                    KIPP_FEED_FORWARD_LENGTH, KIPP_EMBEDDING_LENGTH, stream),
                "gate projection launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &layer->feed_forward_up),
                    session->normalized, session->up,
                    KIPP_FEED_FORWARD_LENGTH, KIPP_EMBEDDING_LENGTH, stream),
                "up projection launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_swiglu(session->gate, session->up,
                                        KIPP_FEED_FORWARD_LENGTH, stream),
                "SwiGLU launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &layer->feed_forward_down),
                    session->gate, session->projection,
                    KIPP_EMBEDDING_LENGTH, KIPP_FEED_FORWARD_LENGTH, stream),
                "down projection launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_residual(
                    session->x, session->projection, KIPP_EMBEDDING_LENGTH,
                    stream),
                "feed-forward residual launch", error) != 0) {
            return -1;
        }
    }

    if (write_logits) {
        if (cuda_check_launch(
                kipp_cuda_launch_rms_norm(
                    session->x, cuda_tensor(model, &weights->output_norm),
                    session->normalized, KIPP_EMBEDDING_LENGTH,
                    KIPP_CUDA_RMS_EPSILON, stream),
                "output RMSNorm launch", error) != 0 ||
            cuda_check_launch(
                kipp_cuda_launch_matvec(
                    cuda_tensor(model, &weights->token_embedding),
                    session->normalized, session->logits, KIPP_VOCAB_SIZE,
                    KIPP_EMBEDDING_LENGTH, stream),
                "logit projection launch", error) != 0) {
            return -1;
        }
        cudaError_t status =
            cudaMemcpyAsync(host_logits, session->logits,
                            KIPP_VOCAB_SIZE * sizeof(*host_logits),
                            cudaMemcpyDeviceToHost, stream);
        if (status != cudaSuccess) {
            return cuda_fail_status(error, "logit download", status);
        }
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            return cuda_fail_status(error, "evaluation", status);
        }
    }
    return 0;
}

static int cuda_model_create(const kipp_model_view *view, void **backend_model,
                             kipp_error *error) {
    cuda_clear_error(error);
    if (view == NULL || backend_model == NULL) {
        return cuda_fail(error, KIPP_ERROR_ARGUMENT,
                         "CUDA model arguments are required");
    }
    *backend_model = NULL;

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return cuda_fail_status(error, "device selection", status);
    }
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return cuda_fail_status(error, "device query", status);
    }

    uintptr_t span_start = 0;
    size_t span_bytes = 0;
    if (cuda_model_tensor_span(view, &span_start, &span_bytes, error) != 0) {
        return -1;
    }

    cuda_backend_model *model =
        static_cast<cuda_backend_model *>(calloc(1, sizeof(*model)));
    if (model == NULL) {
        return cuda_fail(error, KIPP_ERROR_MEMORY,
                         "unable to allocate CUDA model metadata");
    }
    model->device = device;
    model->view = view;
    model->host_span_start = span_start;
    model->weight_bytes = span_bytes;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        cuda_fail_status(error, "memory capacity query", status);
        free(model);
        return -1;
    }
    uint64_t required_bytes =
        static_cast<uint64_t>(span_bytes) +
        cuda_session_device_bytes(KIPP_CUDA_ORACLE_CAPACITY);
    if (required_bytes > free_bytes) {
        cuda_fail(error, KIPP_ERROR_MEMORY,
                  "CUDA model requires %llu device bytes but only %zu of %zu "
                  "bytes are free",
                  static_cast<unsigned long long>(required_bytes), free_bytes,
                  total_bytes);
        free(model);
        return -1;
    }

    status = cudaMalloc(reinterpret_cast<void **>(&model->weights), span_bytes);
    if (status != cudaSuccess) {
        cuda_fail_status(error, "resident weight allocation", status);
        free(model);
        return -1;
    }
    status = cudaMemcpy(model->weights,
                        reinterpret_cast<const void *>(span_start), span_bytes,
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cuda_fail_status(error, "resident weight upload", status);
        (void)cudaFree(model->weights);
        free(model);
        return -1;
    }

    model->oracle_session =
        cuda_session_new(model, KIPP_CUDA_ORACLE_CAPACITY, error);
    if (model->oracle_session == NULL) {
        (void)cudaFree(model->weights);
        free(model);
        return -1;
    }
    *backend_model = model;
    (void)properties;
    return 0;
}

static void cuda_model_destroy(void *backend_model) {
    cuda_backend_model *model =
        static_cast<cuda_backend_model *>(backend_model);
    if (model == NULL) {
        return;
    }
    (void)cudaSetDevice(model->device);
    cuda_session_delete(model->oracle_session);
    (void)cudaFree(model->weights);
    free(model);
}

static int cuda_session_create(void *backend_model, uint32_t capacity,
                               void **backend_session, kipp_error *error) {
    cuda_clear_error(error);
    if (backend_model == NULL || backend_session == NULL) {
        return cuda_fail(error, KIPP_ERROR_ARGUMENT,
                         "CUDA session arguments are required");
    }
    *backend_session = NULL;
    cuda_backend_model *model =
        static_cast<cuda_backend_model *>(backend_model);
    cudaError_t status = cudaSetDevice(model->device);
    if (status != cudaSuccess) {
        return cuda_fail_status(error, "session device selection", status);
    }
    cuda_backend_session *session = cuda_session_new(model, capacity, error);
    if (session == NULL) {
        return -1;
    }
    *backend_session = session;
    return 0;
}

static void cuda_session_destroy(void *backend_session) {
    cuda_session_delete(
        static_cast<cuda_backend_session *>(backend_session));
}

static int cuda_session_reset(void *backend_session, kipp_error *error) {
    cuda_clear_error(error);
    if (backend_session == NULL) {
        return cuda_fail(error, KIPP_ERROR_ARGUMENT,
                         "CUDA session is required");
    }
    cuda_backend_session *session =
        static_cast<cuda_backend_session *>(backend_session);
    session->length = 0;
    return 0;
}

static int cuda_eval(void *backend_model, kipp_eval_item *items,
                     size_t item_count, kipp_error *error) {
    cuda_clear_error(error);
    if (backend_model == NULL || items == NULL || item_count == 0) {
        return cuda_fail(error, KIPP_ERROR_ARGUMENT,
                         "CUDA evaluation items are required");
    }
    cuda_backend_model *model =
        static_cast<cuda_backend_model *>(backend_model);
    cudaError_t status = cudaSetDevice(model->device);
    if (status != cudaSuccess) {
        return cuda_fail_status(error, "evaluation device selection", status);
    }

    for (size_t item_index = 0; item_index < item_count; ++item_index) {
        kipp_eval_item *item = &items[item_index];
        if (item->tokens == NULL || item->token_count == 0 ||
            item->logits == NULL) {
            return cuda_fail(error, KIPP_ERROR_ARGUMENT,
                             "CUDA evaluation item is incomplete");
        }
        for (uint32_t token_index = 0; token_index < item->token_count;
             ++token_index) {
            if (item->tokens[token_index] >= KIPP_VOCAB_SIZE) {
                return cuda_fail(error, KIPP_ERROR_RANGE,
                                 "CUDA token %u is out of range", token_index);
            }
        }

        cuda_backend_session *session;
        if (item->session == NULL) {
            if (item->start_position != 0 ||
                item->token_count > KIPP_CUDA_ORACLE_CAPACITY) {
                return cuda_fail(
                    error, KIPP_ERROR_RANGE,
                    "stateless CUDA evaluation must start at zero and fit %u "
                    "tokens",
                    KIPP_CUDA_ORACLE_CAPACITY);
            }
            session = model->oracle_session;
            session->length = 0;
        } else {
            session =
                static_cast<cuda_backend_session *>(item->session);
            if (session->model != model) {
                return cuda_fail(error, KIPP_ERROR_ARGUMENT,
                                 "CUDA session belongs to another model");
            }
        }
        if (item->start_position != session->length) {
            return cuda_fail(
                error, KIPP_ERROR_ARGUMENT,
                "CUDA start position %u does not match session length %u",
                item->start_position, session->length);
        }
        if (item->token_count > session->capacity - session->length) {
            return cuda_fail(error, KIPP_ERROR_RANGE,
                             "CUDA session append exceeds capacity %u",
                             session->capacity);
        }

        uint32_t initial_length = session->length;
        for (uint32_t token_index = 0; token_index < item->token_count;
             ++token_index) {
            bool final_token = token_index + 1 == item->token_count;
            if (cuda_encode_token(
                    model, session, item->tokens[token_index],
                    item->start_position + token_index, final_token,
                    final_token ? item->logits : NULL, error) != 0) {
                session->length = initial_length;
                (void)cudaStreamSynchronize(session->stream);
                return -1;
            }
            ++session->length;
        }
    }
    return 0;
}

extern "C" const kipp_backend_ops *kipp_cuda_backend_operations(void) {
    static const kipp_backend_ops operations = {
        cuda_model_create,   cuda_model_destroy, cuda_session_create,
        cuda_session_destroy, cuda_session_reset, cuda_eval,
    };
    return &operations;
}

static uint16_t cuda_test_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

static int cuda_expect_near(const char *name, const float *actual,
                            const float *expected, size_t count,
                            float tolerance, kipp_error *error) {
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(actual[index]) ||
            fabsf(actual[index] - expected[index]) > tolerance) {
            return cuda_fail(
                error, KIPP_ERROR_INTERNAL,
                "CUDA operator %s mismatch at %zu: %.9g != %.9g", name,
                index, actual[index], expected[index]);
        }
    }
    return 0;
}

template <typename T>
class cuda_test_buffer {
  public:
    cuda_test_buffer() : data_(NULL) {}
    ~cuda_test_buffer() { (void)cudaFree(data_); }
    cuda_test_buffer(const cuda_test_buffer &) = delete;
    cuda_test_buffer &operator=(const cuda_test_buffer &) = delete;

    int allocate(size_t count, const char *name, kipp_error *error) {
        return cuda_allocate(&data_, count, name, error);
    }

    int upload(const T *source, size_t count, const char *name,
               kipp_error *error) {
        cudaError_t status =
            cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice);
        return status == cudaSuccess ? 0
                                     : cuda_fail_status(error, name, status);
    }

    int download(T *destination, size_t count, cudaStream_t stream,
                 const char *name, kipp_error *error) {
        cudaError_t status =
            cudaMemcpyAsync(destination, data_, count * sizeof(T),
                            cudaMemcpyDeviceToHost, stream);
        return status == cudaSuccess ? 0
                                     : cuda_fail_status(error, name, status);
    }

    T *get() { return data_; }
    const T *get() const { return data_; }

  private:
    T *data_;
};

static int cuda_test_sync(cudaStream_t stream, kipp_error *error) {
    cudaError_t status = cudaStreamSynchronize(stream);
    return status == cudaSuccess
               ? 0
               : cuda_fail_status(error, "operator test execution", status);
}

static int cuda_test_bf16(cudaStream_t stream, kipp_error *error) {
    const float input[] = {1.0f, -2.0f, 1.00390625f};
    uint16_t expected_bits[3];
    float expected[3];
    for (size_t index = 0; index < 3; ++index) {
        expected_bits[index] = cuda_test_float_to_bf16(input[index]);
        uint32_t bits = static_cast<uint32_t>(expected_bits[index]) << 16;
        memcpy(&expected[index], &bits, sizeof(bits));
    }
    uint16_t actual_bits[3] = {0};
    float actual[3] = {0.0f};
    cuda_test_buffer<float> device_input;
    cuda_test_buffer<uint16_t> device_bits;
    cuda_test_buffer<float> device_output;
    if (device_input.allocate(3, "BF16 test input", error) != 0 ||
        device_bits.allocate(3, "BF16 test bits", error) != 0 ||
        device_output.allocate(3, "BF16 test output", error) != 0 ||
        device_input.upload(input, 3, "BF16 test input upload", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_bf16_roundtrip(
                              device_input.get(), device_bits.get(),
                              device_output.get(), 3, stream),
                          "BF16 test launch", error) != 0 ||
        device_bits.download(actual_bits, 3, stream,
                             "BF16 test bits download", error) != 0 ||
        device_output.download(actual, 3, stream, "BF16 test output download",
                               error) != 0 ||
        cuda_test_sync(stream, error) != 0) {
        return -1;
    }
    if (memcmp(actual_bits, expected_bits, sizeof(actual_bits)) != 0) {
        return cuda_fail(error, KIPP_ERROR_INTERNAL,
                         "CUDA BF16 bit conversion mismatch");
    }
    return cuda_expect_near("bf16", actual, expected, 3, 0.0f, error);
}

static int cuda_test_embedding(cudaStream_t stream, kipp_error *error) {
    const size_t count = 2u * KIPP_EMBEDDING_LENGTH;
    uint16_t *embedding =
        static_cast<uint16_t *>(calloc(count, sizeof(*embedding)));
    float *actual = static_cast<float *>(
        malloc(KIPP_EMBEDDING_LENGTH * sizeof(*actual)));
    float *expected = static_cast<float *>(
        malloc(KIPP_EMBEDDING_LENGTH * sizeof(*expected)));
    if (embedding == NULL || actual == NULL || expected == NULL) {
        free(embedding);
        free(actual);
        free(expected);
        return cuda_fail(error, KIPP_ERROR_MEMORY,
                         "CUDA embedding test allocation failed");
    }
    for (uint32_t index = 0; index < KIPP_EMBEDDING_LENGTH; ++index) {
        embedding[KIPP_EMBEDDING_LENGTH + index] =
            cuda_test_float_to_bf16(2.0f);
        expected[index] = 2.0f;
    }
    cuda_test_buffer<uint16_t> device_embedding;
    cuda_test_buffer<float> device_output;
    int result = 0;
    if (device_embedding.allocate(count, "embedding test weights", error) !=
            0 ||
        device_output.allocate(KIPP_EMBEDDING_LENGTH, "embedding test output",
                               error) != 0 ||
        device_embedding.upload(embedding, count, "embedding test upload",
                                error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_embed(
                              device_embedding.get(), device_output.get(), 1,
                              stream),
                          "embedding test launch", error) != 0 ||
        device_output.download(actual, KIPP_EMBEDDING_LENGTH, stream,
                               "embedding test download", error) != 0 ||
        cuda_test_sync(stream, error) != 0 ||
        cuda_expect_near("embedding", actual, expected,
                         KIPP_EMBEDDING_LENGTH, 0.0f, error) != 0) {
        result = -1;
    }
    free(embedding);
    free(actual);
    free(expected);
    return result;
}

static int cuda_test_norm(cudaStream_t stream, kipp_error *error) {
    const float input[] = {3.0f, 4.0f};
    const uint16_t weight[] = {cuda_test_float_to_bf16(1.0f),
                               cuda_test_float_to_bf16(1.0f)};
    float scale = 1.0f / sqrtf(12.5f);
    const float expected[] = {3.0f * scale, 4.0f * scale};
    float actual[2] = {0.0f};
    cuda_test_buffer<float> device_input;
    cuda_test_buffer<uint16_t> device_weight;
    cuda_test_buffer<float> device_output;
    if (device_input.allocate(2, "RMSNorm test input", error) != 0 ||
        device_weight.allocate(2, "RMSNorm test weight", error) != 0 ||
        device_output.allocate(2, "RMSNorm test output", error) != 0 ||
        device_input.upload(input, 2, "RMSNorm input upload", error) != 0 ||
        device_weight.upload(weight, 2, "RMSNorm weight upload", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_rms_norm(
                              device_input.get(), device_weight.get(),
                              device_output.get(), 2, 0.0f, stream),
                          "RMSNorm test launch", error) != 0 ||
        device_output.download(actual, 2, stream, "RMSNorm test download",
                               error) != 0 ||
        cuda_test_sync(stream, error) != 0) {
        return -1;
    }
    return cuda_expect_near("rms_norm", actual, expected, 2, 1.0e-6f, error);
}

static int cuda_test_matvec(cudaStream_t stream, kipp_error *error) {
    const uint16_t weight[] = {
        cuda_test_float_to_bf16(1.0f),  cuda_test_float_to_bf16(2.0f),
        cuda_test_float_to_bf16(3.0f),  cuda_test_float_to_bf16(-1.0f),
        cuda_test_float_to_bf16(0.5f),  cuda_test_float_to_bf16(0.0f),
    };
    const float input[] = {2.0f, -1.0f, 0.5f};
    const float expected[] = {1.5f, -2.5f};
    float actual[2] = {0.0f};
    cuda_test_buffer<uint16_t> device_weight;
    cuda_test_buffer<float> device_input;
    cuda_test_buffer<float> device_output;
    if (device_weight.allocate(6, "matvec test weight", error) != 0 ||
        device_input.allocate(3, "matvec test input", error) != 0 ||
        device_output.allocate(2, "matvec test output", error) != 0 ||
        device_weight.upload(weight, 6, "matvec weight upload", error) != 0 ||
        device_input.upload(input, 3, "matvec input upload", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_matvec(
                              device_weight.get(), device_input.get(),
                              device_output.get(), 2, 3, stream),
                          "matvec test launch", error) != 0 ||
        device_output.download(actual, 2, stream, "matvec test download",
                               error) != 0 ||
        cuda_test_sync(stream, error) != 0) {
        return -1;
    }
    return cuda_expect_near("matvec", actual, expected, 2, 1.0e-6f, error);
}

static int cuda_test_head_norm(cudaStream_t stream, kipp_error *error) {
    float states[KIPP_ATTENTION_HEAD_DIM];
    uint16_t weight[KIPP_ATTENTION_HEAD_DIM];
    float actual[KIPP_ATTENTION_HEAD_DIM];
    float expected[KIPP_ATTENTION_HEAD_DIM];
    float normalized =
        1.0f / sqrtf(1.0f + KIPP_CUDA_RMS_EPSILON);
    for (uint32_t index = 0; index < KIPP_ATTENTION_HEAD_DIM; ++index) {
        states[index] = 1.0f;
        weight[index] = cuda_test_float_to_bf16(1.0f);
        expected[index] = normalized;
    }
    cuda_test_buffer<float> device_states;
    cuda_test_buffer<uint16_t> device_weight;
    if (device_states.allocate(KIPP_ATTENTION_HEAD_DIM,
                               "head norm test states", error) != 0 ||
        device_weight.allocate(KIPP_ATTENTION_HEAD_DIM,
                               "head norm test weight", error) != 0 ||
        device_states.upload(states, KIPP_ATTENTION_HEAD_DIM,
                             "head norm state upload", error) != 0 ||
        device_weight.upload(weight, KIPP_ATTENTION_HEAD_DIM,
                             "head norm weight upload", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_head_norm(
                              device_states.get(), device_weight.get(), 1,
                              KIPP_CUDA_RMS_EPSILON, stream),
                          "head norm test launch", error) != 0 ||
        device_states.download(actual, KIPP_ATTENTION_HEAD_DIM, stream,
                               "head norm test download", error) != 0 ||
        cuda_test_sync(stream, error) != 0) {
        return -1;
    }
    return cuda_expect_near("head_norm", actual, expected,
                            KIPP_ATTENTION_HEAD_DIM, 1.0e-6f, error);
}

static int cuda_test_rope(cudaStream_t stream, kipp_error *error) {
    float states[KIPP_ATTENTION_HEAD_DIM] = {0.0f};
    float actual[KIPP_ATTENTION_HEAD_DIM];
    float expected[KIPP_ATTENTION_HEAD_DIM] = {0.0f};
    states[0] = 1.0f;
    states[KIPP_ATTENTION_HEAD_DIM / 2] = 3.0f;
    expected[0] = cosf(1.0f) - 3.0f * sinf(1.0f);
    expected[KIPP_ATTENTION_HEAD_DIM / 2] =
        3.0f * cosf(1.0f) + sinf(1.0f);
    cuda_test_buffer<float> device_states;
    if (device_states.allocate(KIPP_ATTENTION_HEAD_DIM, "RoPE test states",
                               error) != 0 ||
        device_states.upload(states, KIPP_ATTENTION_HEAD_DIM,
                             "RoPE state upload", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_rope(
                              device_states.get(), 1, 1, 10000.0f, stream),
                          "RoPE test launch", error) != 0 ||
        device_states.download(actual, KIPP_ATTENTION_HEAD_DIM, stream,
                               "RoPE test download", error) != 0 ||
        cuda_test_sync(stream, error) != 0) {
        return -1;
    }
    return cuda_expect_near("rope", actual, expected,
                            KIPP_ATTENTION_HEAD_DIM, 2.0e-5f, error);
}

static int cuda_test_elementwise(cudaStream_t stream, kipp_error *error) {
    const float residual[] = {1.0f, 2.0f};
    const float addition[] = {3.0f, 4.0f};
    const float gate[] = {1.0f, 0.0f};
    const float up[] = {2.0f, 3.0f};
    const float expected_residual[] = {4.0f, 6.0f};
    const float expected_gate[] = {1.4621172f, 0.0f};
    float actual_residual[2] = {0.0f};
    float actual_gate[2] = {0.0f};
    cuda_test_buffer<float> device_residual;
    cuda_test_buffer<float> device_addition;
    cuda_test_buffer<float> device_gate;
    cuda_test_buffer<float> device_up;
    if (device_residual.allocate(2, "residual test values", error) != 0 ||
        device_addition.allocate(2, "residual test addition", error) != 0 ||
        device_gate.allocate(2, "SwiGLU test gate", error) != 0 ||
        device_up.allocate(2, "SwiGLU test up", error) != 0 ||
        device_residual.upload(residual, 2, "residual test upload", error) !=
            0 ||
        device_addition.upload(addition, 2, "addition test upload", error) !=
            0 ||
        device_gate.upload(gate, 2, "gate test upload", error) != 0 ||
        device_up.upload(up, 2, "up test upload", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_residual(
                              device_residual.get(), device_addition.get(), 2,
                              stream),
                          "residual test launch", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_swiglu(
                              device_gate.get(), device_up.get(), 2, stream),
                          "SwiGLU test launch", error) != 0 ||
        device_residual.download(actual_residual, 2, stream,
                                 "residual test download", error) != 0 ||
        device_gate.download(actual_gate, 2, stream, "SwiGLU test download",
                             error) != 0 ||
        cuda_test_sync(stream, error) != 0 ||
        cuda_expect_near("residual", actual_residual, expected_residual, 2,
                         1.0e-6f, error) != 0) {
        return -1;
    }
    return cuda_expect_near("swiglu", actual_gate, expected_gate, 2, 1.0e-6f,
                            error);
}

static int cuda_test_kv_gqa(cudaStream_t stream, kipp_error *error) {
    constexpr uint32_t capacity = 2;
    constexpr size_t kv_values =
        capacity * KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    constexpr size_t state_values =
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM;
    constexpr size_t query_values =
        KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM;
    float key[state_values] = {0.0f};
    float value[state_values] = {0.0f};
    float query[query_values] = {0.0f};
    float actual[query_values];
    for (uint32_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT_KV; ++head) {
        value[static_cast<size_t>(head) * KIPP_ATTENTION_HEAD_DIM] = 1.0f;
    }

    cuda_test_buffer<float> device_key;
    cuda_test_buffer<float> device_value;
    cuda_test_buffer<float> device_query;
    cuda_test_buffer<uint16_t> device_key_cache;
    cuda_test_buffer<uint16_t> device_value_cache;
    cuda_test_buffer<float> device_scores;
    cuda_test_buffer<float> device_output;
    if (device_key.allocate(state_values, "KV test key", error) != 0 ||
        device_value.allocate(state_values, "KV test value", error) != 0 ||
        device_query.allocate(query_values, "GQA test query", error) != 0 ||
        device_key_cache.allocate(kv_values, "GQA test key cache", error) !=
            0 ||
        device_value_cache.allocate(kv_values, "GQA test value cache", error) !=
            0 ||
        device_scores.allocate(KIPP_ATTENTION_HEAD_COUNT * capacity,
                               "GQA test scores", error) != 0 ||
        device_output.allocate(query_values, "GQA test output", error) != 0 ||
        device_key.upload(key, state_values, "KV key upload", error) != 0 ||
        device_value.upload(value, state_values, "KV value upload", error) !=
            0 ||
        device_query.upload(query, query_values, "GQA query upload", error) !=
            0 ||
        cuda_check_launch(kipp_cuda_launch_kv_write(
                              device_key.get(), device_value.get(),
                              device_key_cache.get(), device_value_cache.get(),
                              0, 0, capacity, stream),
                          "first KV test launch", error) != 0 ||
        cuda_test_sync(stream, error) != 0) {
        return -1;
    }

    for (uint32_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT_KV; ++head) {
        value[static_cast<size_t>(head) * KIPP_ATTENTION_HEAD_DIM] = 3.0f;
    }
    if (device_value.upload(value, state_values, "second KV value upload",
                            error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_kv_write(
                              device_key.get(), device_value.get(),
                              device_key_cache.get(), device_value_cache.get(),
                              0, 1, capacity, stream),
                          "second KV test launch", error) != 0 ||
        cuda_check_launch(kipp_cuda_launch_cached_gqa(
                              device_query.get(), device_key_cache.get(),
                              device_value_cache.get(), device_scores.get(),
                              device_output.get(), 0, 1, capacity, stream),
                          "GQA test launch", error) != 0 ||
        device_output.download(actual, query_values, stream,
                               "GQA test download", error) != 0 ||
        cuda_test_sync(stream, error) != 0) {
        return -1;
    }
    for (uint32_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT; ++head) {
        size_t base = static_cast<size_t>(head) * KIPP_ATTENTION_HEAD_DIM;
        if (fabsf(actual[base] - 2.0f) > 1.0e-6f) {
            return cuda_fail(error, KIPP_ERROR_INTERNAL,
                             "CUDA cached GQA mismatch at head %u", head);
        }
    }
    return 0;
}

extern "C" int kipp_cuda_run_operator_tests(kipp_error *error) {
    cuda_clear_error(error);
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return cuda_fail_status(error, "operator test device selection",
                                status);
    }
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return cuda_fail_status(error, "operator test device query", status);
    }
    cudaStream_t stream = NULL;
    status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        return cuda_fail_status(error, "operator test stream creation",
                                status);
    }

    int result = 0;
    if (cuda_test_bf16(stream, error) != 0 ||
        cuda_test_embedding(stream, error) != 0 ||
        cuda_test_norm(stream, error) != 0 ||
        cuda_test_matvec(stream, error) != 0 ||
        cuda_test_head_norm(stream, error) != 0 ||
        cuda_test_rope(stream, error) != 0 ||
        cuda_test_elementwise(stream, error) != 0 ||
        cuda_test_kv_gqa(stream, error) != 0) {
        result = -1;
    }
    (void)cudaStreamDestroy(stream);
    (void)properties;
    return result;
}

extern "C" const char *kipp_cuda_device_name(void) {
    static char name[256];
    int device = 0;
    cudaDeviceProp properties;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return "unavailable";
    }
    (void)snprintf(name, sizeof(name), "%s", properties.name);
    return name;
}
