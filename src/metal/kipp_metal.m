#import "metal/kipp_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kipp_metal_source.inc"

#define KIPP_METAL_ORACLE_CAPACITY 256u
#define KIPP_METAL_RMS_EPSILON 1.0e-6f
#define KIPP_METAL_ROPE_THETA 1000000.0f
/* Tokens encoded per command buffer; bounds the persistent activation
 * buffers and the host-visible token-ID staging buffer. */
#define KIPP_METAL_BATCH 32u
#define KIPP_METAL_NORM_THREADS 256u
#define KIPP_METAL_GROUP_THREADS 128u
#define KIPP_METAL_GQA_THREADS 256u
#define KIPP_METAL_PAIRS_PER_GROUP 4u
#define KIPP_METAL_TOKEN_TILE 8u

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t token_count;
} metal_matvec_params;

typedef struct {
    uint32_t length;
    float epsilon;
} metal_norm_params;

typedef struct {
    uint32_t head_count;
    uint32_t token_count;
    float epsilon;
} metal_head_norm_params;

typedef struct {
    uint32_t head_count;
    uint32_t start_position;
    float theta;
} metal_rope_params;

typedef struct {
    uint32_t layer;
    uint32_t start_position;
    uint32_t token_count;
    uint32_t capacity;
} metal_kv_params;

@class KippMetalModel;

@interface KippMetalSession : NSObject
@property(nonatomic, weak) KippMetalModel *model;
@property(nonatomic) uint32_t capacity;
@property(nonatomic) uint32_t length;
@property(nonatomic, strong) id<MTLBuffer> keyCache;
@property(nonatomic, strong) id<MTLBuffer> valueCache;
@property(nonatomic, strong) id<MTLBuffer> tokens;
@property(nonatomic, strong) id<MTLBuffer> x;
@property(nonatomic, strong) id<MTLBuffer> normalized;
@property(nonatomic, strong) id<MTLBuffer> query;
@property(nonatomic, strong) id<MTLBuffer> key;
@property(nonatomic, strong) id<MTLBuffer> value;
@property(nonatomic, strong) id<MTLBuffer> attention;
@property(nonatomic, strong) id<MTLBuffer> projection;
@property(nonatomic, strong) id<MTLBuffer> gate;
@property(nonatomic, strong) id<MTLBuffer> up;
@property(nonatomic, strong) id<MTLBuffer> staging;
@end

@implementation KippMetalSession
@end

@interface KippMetalModel : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLBuffer> weights;
@property(nonatomic, strong)
    NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(nonatomic) const kipp_model_view *view;
@property(nonatomic) BOOL matrixKernelAvailable;
@property(nonatomic, strong) id<MTLBuffer> batchLogits;
@property(nonatomic, strong) KippMetalSession *oracleSession;
@end

@implementation KippMetalModel
@end

static int metal_fail(kipp_error *error, kipp_error_code code,
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

static void metal_clear_error(kipp_error *error) {
    if (error != NULL) {
        error->code = KIPP_OK;
        error->message[0] = '\0';
    }
}

static NSUInteger metal_weight_offset(KippMetalModel *model,
                                      const kipp_tensor_view *tensor) {
    return (NSUInteger)(tensor->data - model.view->mapping);
}

static id<MTLComputePipelineState>
metal_pipeline(KippMetalModel *model, NSString *name) {
    return model.pipelines[name];
}

static id<MTLBuffer> metal_new_shared_buffer(id<MTLDevice> device,
                                             NSUInteger length) {
    return [device newBufferWithLength:length
                               options:MTLResourceStorageModeShared];
}

static KippMetalSession *
metal_session_new(KippMetalModel *model, uint32_t capacity,
                  kipp_error *error) {
    if (capacity == 0 || capacity > KIPP_CONTEXT_LENGTH) {
        metal_fail(error, KIPP_ERROR_RANGE,
                   "Metal session capacity must be between 1 and %u",
                   KIPP_CONTEXT_LENGTH);
        return nil;
    }
    uint64_t slabBytes =
        (uint64_t)KIPP_BLOCK_COUNT * capacity *
        KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM *
        sizeof(uint16_t);
    if (slabBytes > NSUIntegerMax) {
        metal_fail(error, KIPP_ERROR_MEMORY, "Metal session size overflows");
        return nil;
    }

    KippMetalSession *session = [[KippMetalSession alloc] init];
    session.model = model;
    session.capacity = capacity;
    session.keyCache =
        metal_new_shared_buffer(model.device, (NSUInteger)slabBytes);
    session.valueCache =
        metal_new_shared_buffer(model.device, (NSUInteger)slabBytes);
    session.tokens = metal_new_shared_buffer(
        model.device, KIPP_METAL_BATCH * sizeof(uint32_t));
    session.x = metal_new_shared_buffer(
        model.device,
        KIPP_METAL_BATCH * KIPP_EMBEDDING_LENGTH * sizeof(float));
    session.normalized = metal_new_shared_buffer(
        model.device,
        KIPP_METAL_BATCH * KIPP_EMBEDDING_LENGTH * sizeof(float));
    session.query = metal_new_shared_buffer(
        model.device, KIPP_METAL_BATCH * KIPP_ATTENTION_HEAD_COUNT *
                          KIPP_ATTENTION_HEAD_DIM * sizeof(float));
    session.key = metal_new_shared_buffer(
        model.device, KIPP_METAL_BATCH * KIPP_ATTENTION_HEAD_COUNT_KV *
                          KIPP_ATTENTION_HEAD_DIM * sizeof(float));
    session.value = metal_new_shared_buffer(
        model.device, KIPP_METAL_BATCH * KIPP_ATTENTION_HEAD_COUNT_KV *
                          KIPP_ATTENTION_HEAD_DIM * sizeof(float));
    session.attention = metal_new_shared_buffer(
        model.device, KIPP_METAL_BATCH * KIPP_ATTENTION_HEAD_COUNT *
                          KIPP_ATTENTION_HEAD_DIM * sizeof(float));
    session.projection = metal_new_shared_buffer(
        model.device,
        KIPP_METAL_BATCH * KIPP_EMBEDDING_LENGTH * sizeof(float));
    session.gate = metal_new_shared_buffer(
        model.device,
        KIPP_METAL_BATCH * KIPP_FEED_FORWARD_LENGTH * sizeof(float));
    session.up = metal_new_shared_buffer(
        model.device,
        KIPP_METAL_BATCH * KIPP_FEED_FORWARD_LENGTH * sizeof(float));
    session.staging = metal_new_shared_buffer(
        model.device,
        KIPP_METAL_BATCH * KIPP_FEED_FORWARD_LENGTH * sizeof(uint16_t));

    if (session.keyCache == nil || session.valueCache == nil ||
        session.tokens == nil || session.x == nil ||
        session.normalized == nil || session.query == nil ||
        session.key == nil || session.value == nil ||
        session.attention == nil || session.projection == nil ||
        session.gate == nil || session.up == nil ||
        session.staging == nil) {
        metal_fail(error, KIPP_ERROR_MEMORY,
                   "unable to allocate persistent Metal session buffers");
        return nil;
    }
    return session;
}

/*
 * All dispatches in a batch share one serial compute encoder, so Metal
 * orders them without per-dispatch encoder setup or hazard-tracking cost.
 */
static void metal_enc_threads(id<MTLComputeCommandEncoder> encoder,
                              id<MTLComputePipelineState> pipeline,
                              NSUInteger width, NSUInteger height,
                              void (^configure)(id<MTLComputeCommandEncoder>)) {
    [encoder setComputePipelineState:pipeline];
    configure(encoder);
    NSUInteger groupWidth =
        MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    groupWidth = MIN(groupWidth, MAX((NSUInteger)1, width));
    [encoder dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(groupWidth, 1, 1)];
}

static void metal_enc_groups(id<MTLComputeCommandEncoder> encoder,
                             id<MTLComputePipelineState> pipeline,
                             NSUInteger groupsX, NSUInteger groupsY,
                             NSUInteger threadsPerGroup,
                             void (^configure)(id<MTLComputeCommandEncoder>)) {
    [encoder setComputePipelineState:pipeline];
    configure(encoder);
    [encoder dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
            threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
}

static void metal_dispatch(id<MTLCommandBuffer> commandBuffer,
                           id<MTLComputePipelineState> pipeline,
                           NSUInteger threadCount,
                           void (^configure)(id<MTLComputeCommandEncoder>)) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    metal_enc_threads(encoder, pipeline, threadCount, 1, configure);
    [encoder endEncoding];
}

static void metal_dispatch_2d(id<MTLCommandBuffer> commandBuffer,
                              id<MTLComputePipelineState> pipeline,
                              NSUInteger width, NSUInteger height,
                              void (^configure)(id<MTLComputeCommandEncoder>)) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    metal_enc_threads(encoder, pipeline, width, height, configure);
    [encoder endEncoding];
}

static void metal_dispatch_groups(id<MTLCommandBuffer> commandBuffer,
                                  id<MTLComputePipelineState> pipeline,
                                  NSUInteger groupsX, NSUInteger groupsY,
                                  NSUInteger threadsPerGroup,
                                  void (^configure)(
                                      id<MTLComputeCommandEncoder>)) {
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    metal_enc_groups(encoder, pipeline, groupsX, groupsY, threadsPerGroup,
                     configure);
    [encoder endEncoding];
}

static NSUInteger metal_pair_groups(NSUInteger pairCount) {
    return (pairCount + KIPP_METAL_PAIRS_PER_GROUP - 1) /
           KIPP_METAL_PAIRS_PER_GROUP;
}

static int metal_commit_and_wait(id<MTLCommandBuffer> commandBuffer,
                                 kipp_error *error) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSString *message = commandBuffer.error.localizedDescription;
        if (message == nil) {
            message = @"unknown error";
        }
        return metal_fail(error, KIPP_ERROR_INTERNAL,
                          "Metal command buffer failed: %s",
                          message.UTF8String);
    }
    return 0;
}

static void metal_encode_embed(id<MTLComputeCommandEncoder> target,
                               KippMetalModel *model,
                               KippMetalSession *session,
                               uint32_t tokenCount) {
    const kipp_tensor_view *embedding =
        &model.view->weights.token_embedding;
    metal_enc_threads(target, metal_pipeline(model, @"kipp_embed_gather"),
                      KIPP_EMBEDDING_LENGTH, tokenCount,
                      ^(id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:model.weights
                                    offset:metal_weight_offset(model, embedding)
                                   atIndex:0];
                        [encoder setBuffer:session.tokens offset:0 atIndex:1];
                        [encoder setBuffer:session.x offset:0 atIndex:2];
                      });
}

static void metal_encode_norm(id<MTLComputeCommandEncoder> target,
                              KippMetalModel *model, id<MTLBuffer> input,
                              NSUInteger inputOffset,
                              const kipp_tensor_view *weight,
                              id<MTLBuffer> output, uint32_t length,
                              uint32_t tokenCount) {
    metal_norm_params params = {length, KIPP_METAL_RMS_EPSILON};
    metal_enc_groups(
        target, metal_pipeline(model, @"kipp_rms_norm"), tokenCount, 1,
        KIPP_METAL_NORM_THREADS, ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:input offset:inputOffset atIndex:0];
          [encoder setBuffer:model.weights
                      offset:metal_weight_offset(model, weight)
                     atIndex:1];
          [encoder setBuffer:output offset:0 atIndex:2];
          [encoder setBytes:&params length:sizeof(params) atIndex:3];
        });
}

static void metal_encode_matvec(id<MTLComputeCommandEncoder> target,
                                KippMetalModel *model,
                                const kipp_tensor_view *weight,
                                id<MTLBuffer> input, id<MTLBuffer> output,
                                NSUInteger outputOffset, uint32_t rows,
                                uint32_t columns, uint32_t tokenCount) {
    metal_matvec_params params = {rows, columns, tokenCount};
    NSUInteger rowGroups = metal_pair_groups(rows);
    NSUInteger tokenGroups =
        tokenCount == 1
            ? 1
            : (tokenCount + KIPP_METAL_TOKEN_TILE - 1) / KIPP_METAL_TOKEN_TILE;
    id<MTLComputePipelineState> pipeline = metal_pipeline(
        model, tokenCount == 1 ? @"kipp_matvec_bf16_decode"
                               : @"kipp_matvec_bf16_prefill");
    metal_enc_groups(
        target, pipeline, rowGroups, tokenGroups, KIPP_METAL_GROUP_THREADS,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:model.weights
                      offset:metal_weight_offset(model, weight)
                     atIndex:0];
          [encoder setBuffer:input offset:0 atIndex:1];
          [encoder setBuffer:output offset:outputOffset atIndex:2];
          [encoder setBytes:&params length:sizeof(params) atIndex:3];
        });
}

/* Stage FP32 activations as BF16 in the session's staging buffer. */
static void metal_encode_stage(id<MTLComputeCommandEncoder> target,
                               KippMetalModel *model,
                               KippMetalSession *session,
                               id<MTLBuffer> input, uint32_t count) {
    metal_enc_threads(target, metal_pipeline(model, @"kipp_bf16_stage"),
                      count, 1, ^(id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:input offset:0 atIndex:0];
                        [encoder setBuffer:session.staging
                                    offset:0
                                   atIndex:1];
                        [encoder setBytes:&count
                                   length:sizeof(count)
                                  atIndex:2];
                      });
}

/*
 * One projection. When the matrix kernel is in play the caller has already
 * staged the FP32 input as BF16, and the simdgroup-matrix kernel runs;
 * otherwise the vector kernel reads the FP32 input directly.
 */
static void metal_encode_projection(id<MTLComputeCommandEncoder> target,
                                    KippMetalModel *model,
                                    KippMetalSession *session,
                                    const kipp_tensor_view *weight,
                                    id<MTLBuffer> input, id<MTLBuffer> output,
                                    uint32_t rows, uint32_t columns,
                                    uint32_t tokenCount, bool useMatrix) {
    if (!useMatrix) {
        metal_encode_matvec(target, model, weight, input, output, 0, rows,
                            columns, tokenCount);
        return;
    }
    metal_matvec_params params = {rows, columns, tokenCount};
    NSUInteger rowGroups = (rows + 127u) / 128u;
    NSUInteger tokenGroups = (tokenCount + 15u) / 16u;
    metal_enc_groups(
        target, metal_pipeline(model, @"kipp_matmul_bf16"), rowGroups,
        tokenGroups, KIPP_METAL_GROUP_THREADS,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:model.weights
                      offset:metal_weight_offset(model, weight)
                     atIndex:0];
          [encoder setBuffer:session.staging offset:0 atIndex:1];
          [encoder setBuffer:output offset:0 atIndex:2];
          [encoder setBytes:&params length:sizeof(params) atIndex:3];
        });
}

static void metal_encode_head_norm(id<MTLComputeCommandEncoder> target,
                                   KippMetalModel *model,
                                   id<MTLBuffer> states,
                                   const kipp_tensor_view *weight,
                                   uint32_t headCount, uint32_t tokenCount) {
    metal_head_norm_params params = {headCount, tokenCount,
                                     KIPP_METAL_RMS_EPSILON};
    metal_enc_groups(
        target, metal_pipeline(model, @"kipp_head_norm"),
        metal_pair_groups((NSUInteger)headCount * tokenCount), 1,
        KIPP_METAL_GROUP_THREADS, ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:states offset:0 atIndex:0];
          [encoder setBuffer:model.weights
                      offset:metal_weight_offset(model, weight)
                     atIndex:1];
          [encoder setBytes:&params length:sizeof(params) atIndex:2];
        });
}

static void metal_encode_rope(id<MTLComputeCommandEncoder> target,
                              KippMetalModel *model, id<MTLBuffer> states,
                              NSUInteger statesOffset, uint32_t headCount,
                              uint32_t startPosition, uint32_t tokenCount) {
    metal_rope_params params = {headCount, startPosition,
                                KIPP_METAL_ROPE_THETA};
    metal_enc_threads(target, metal_pipeline(model, @"kipp_rope"),
                      (NSUInteger)headCount * (KIPP_ATTENTION_HEAD_DIM / 2),
                      tokenCount, ^(id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:states
                                    offset:statesOffset
                                   atIndex:0];
                        [encoder setBytes:&params
                                   length:sizeof(params)
                                  atIndex:1];
                      });
}

static void metal_encode_kv_write(id<MTLComputeCommandEncoder> target,
                                  KippMetalModel *model,
                                  KippMetalSession *kvSession,
                                  id<MTLBuffer> key, id<MTLBuffer> value,
                                  NSUInteger stateOffset, uint32_t layer,
                                  uint32_t startPosition,
                                  uint32_t tokenCount) {
    metal_kv_params params = {layer, startPosition, tokenCount,
                              kvSession.capacity};
    metal_enc_threads(target, metal_pipeline(model, @"kipp_kv_write"),
                      KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
                      tokenCount, ^(id<MTLComputeCommandEncoder> encoder) {
                        [encoder setBuffer:key
                                    offset:stateOffset
                                   atIndex:0];
                        [encoder setBuffer:value
                                    offset:stateOffset
                                   atIndex:1];
                        [encoder setBuffer:kvSession.keyCache
                                    offset:0
                                   atIndex:2];
                        [encoder setBuffer:kvSession.valueCache
                                    offset:0
                                   atIndex:3];
                        [encoder setBytes:&params
                                   length:sizeof(params)
                                  atIndex:4];
                      });
}

static void metal_encode_gqa(id<MTLComputeCommandEncoder> target,
                             KippMetalModel *model,
                             KippMetalSession *kvSession,
                             id<MTLBuffer> query, id<MTLBuffer> output,
                             NSUInteger headStateOffset, uint32_t layer,
                             uint32_t startPosition, uint32_t tokenCount) {
    metal_kv_params params = {layer, startPosition, tokenCount,
                              kvSession.capacity};
    metal_enc_groups(
        target, metal_pipeline(model, @"kipp_flash_gqa"),
        KIPP_ATTENTION_HEAD_COUNT, tokenCount, KIPP_METAL_GQA_THREADS,
        ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:query offset:headStateOffset atIndex:0];
          [encoder setBuffer:kvSession.keyCache offset:0 atIndex:1];
          [encoder setBuffer:kvSession.valueCache offset:0 atIndex:2];
          [encoder setBuffer:output offset:headStateOffset atIndex:3];
          [encoder setBytes:&params length:sizeof(params) atIndex:4];
        });
}

static void metal_encode_residual(id<MTLComputeCommandEncoder> target,
                                  KippMetalModel *model,
                                  KippMetalSession *session,
                                  uint32_t tokenCount) {
    uint32_t count = KIPP_EMBEDDING_LENGTH * tokenCount;
    metal_enc_threads(target, metal_pipeline(model, @"kipp_residual_add"),
                   count, 1, ^(id<MTLComputeCommandEncoder> encoder) {
                     [encoder setBuffer:session.x offset:0 atIndex:0];
                     [encoder setBuffer:session.projection
                                 offset:0
                                atIndex:1];
                     [encoder setBytes:&count length:sizeof(count) atIndex:2];
                   });
}

static void metal_encode_swiglu(id<MTLComputeCommandEncoder> target,
                                KippMetalModel *model,
                                KippMetalSession *session,
                                uint32_t tokenCount) {
    uint32_t count = KIPP_FEED_FORWARD_LENGTH * tokenCount;
    metal_enc_threads(target, metal_pipeline(model, @"kipp_swiglu"),
                   count, 1, ^(id<MTLComputeCommandEncoder> encoder) {
                     [encoder setBuffer:session.gate offset:0 atIndex:0];
                     [encoder setBuffer:session.up offset:0 atIndex:1];
                     [encoder setBytes:&count length:sizeof(count) atIndex:2];
                   });
}

/*
 * One part of a round: a contiguous run of workspace token slots that
 * belongs to one session's KV timeline. finishLogits is set when the last
 * slot of this part completes its eval item.
 */
typedef struct {
    KippMetalSession *session;
    uint32_t slot;
    uint32_t tokens;
    uint32_t position;
    float *finishLogits;
    NSUInteger logitsOffset;
} metal_round_part;

/*
 * Encode one round as a single command buffer: batched projections over all
 * workspace slots, per-part RoPE/KV/attention against each part's own
 * session, and final logits for parts that finish their item. The caller
 * fills the workspace token buffer first. One blocking wait per round.
 */
static int metal_encode_round(KippMetalModel *model,
                              KippMetalSession *workspace,
                              const metal_round_part *parts,
                              uint32_t partCount, uint32_t totalTokens,
                              kipp_error *error) {
    id<MTLCommandBuffer> commandBuffer = [model.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        return metal_fail(error, KIPP_ERROR_INTERNAL,
                          "unable to create Metal command buffer");
    }
    const NSUInteger querySlotBytes = (NSUInteger)KIPP_ATTENTION_HEAD_COUNT *
                                      KIPP_ATTENTION_HEAD_DIM * sizeof(float);
    const NSUInteger kvSlotBytes = (NSUInteger)KIPP_ATTENTION_HEAD_COUNT_KV *
                                   KIPP_ATTENTION_HEAD_DIM * sizeof(float);
    /* Below ~8 tokens the vector kernel keeps more threadgroups in
     * flight than the matrix kernel and wins despite re-reading weights. */
    bool useMatrix =
        model.matrixKernelAvailable && totalTokens >= KIPP_METAL_TOKEN_TILE;
    metal_encode_embed(encoder, model, workspace, totalTokens);
    for (uint32_t layerIndex = 0; layerIndex < KIPP_BLOCK_COUNT;
         ++layerIndex) {
        const kipp_qwen3_layer_weights *layer =
            &model.view->weights.layers[layerIndex];
        metal_encode_norm(encoder, model, workspace.x, 0,
                          &layer->attention_norm, workspace.normalized,
                          KIPP_EMBEDDING_LENGTH, totalTokens);
        if (useMatrix) {
            metal_encode_stage(encoder, model, workspace,
                               workspace.normalized,
                               KIPP_EMBEDDING_LENGTH * totalTokens);
        }
        metal_encode_projection(
            encoder, model, workspace, &layer->attention_q,
            workspace.normalized, workspace.query,
            KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM,
            KIPP_EMBEDDING_LENGTH, totalTokens, useMatrix);
        metal_encode_projection(
            encoder, model, workspace, &layer->attention_k,
            workspace.normalized, workspace.key,
            KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
            KIPP_EMBEDDING_LENGTH, totalTokens, useMatrix);
        metal_encode_projection(
            encoder, model, workspace, &layer->attention_v,
            workspace.normalized, workspace.value,
            KIPP_ATTENTION_HEAD_COUNT_KV * KIPP_ATTENTION_HEAD_DIM,
            KIPP_EMBEDDING_LENGTH, totalTokens, useMatrix);
        metal_encode_head_norm(encoder, model, workspace.query,
                               &layer->attention_q_norm,
                               KIPP_ATTENTION_HEAD_COUNT, totalTokens);
        metal_encode_head_norm(encoder, model, workspace.key,
                               &layer->attention_k_norm,
                               KIPP_ATTENTION_HEAD_COUNT_KV, totalTokens);
        for (uint32_t part = 0; part < partCount; ++part) {
            const metal_round_part *p = &parts[part];
            metal_encode_rope(encoder, model, workspace.query,
                              p->slot * querySlotBytes,
                              KIPP_ATTENTION_HEAD_COUNT, p->position,
                              p->tokens);
            metal_encode_rope(encoder, model, workspace.key,
                              p->slot * kvSlotBytes,
                              KIPP_ATTENTION_HEAD_COUNT_KV, p->position,
                              p->tokens);
            metal_encode_kv_write(encoder, model, p->session, workspace.key,
                                  workspace.value, p->slot * kvSlotBytes,
                                  layerIndex, p->position, p->tokens);
            metal_encode_gqa(encoder, model, p->session, workspace.query,
                             workspace.attention, p->slot * querySlotBytes,
                             layerIndex, p->position, p->tokens);
        }
        if (useMatrix) {
            metal_encode_stage(encoder, model, workspace,
                               workspace.attention,
                               KIPP_ATTENTION_HEAD_COUNT *
                                   KIPP_ATTENTION_HEAD_DIM * totalTokens);
        }
        metal_encode_projection(
            encoder, model, workspace, &layer->attention_output,
            workspace.attention, workspace.projection, KIPP_EMBEDDING_LENGTH,
            KIPP_ATTENTION_HEAD_COUNT * KIPP_ATTENTION_HEAD_DIM, totalTokens,
            useMatrix);
        metal_encode_residual(encoder, model, workspace, totalTokens);
        metal_encode_norm(encoder, model, workspace.x, 0,
                          &layer->feed_forward_norm, workspace.normalized,
                          KIPP_EMBEDDING_LENGTH, totalTokens);
        if (useMatrix) {
            metal_encode_stage(encoder, model, workspace,
                               workspace.normalized,
                               KIPP_EMBEDDING_LENGTH * totalTokens);
        }
        metal_encode_projection(encoder, model, workspace,
                                &layer->feed_forward_gate,
                                workspace.normalized, workspace.gate,
                                KIPP_FEED_FORWARD_LENGTH,
                                KIPP_EMBEDDING_LENGTH, totalTokens,
                                useMatrix);
        metal_encode_projection(encoder, model, workspace,
                                &layer->feed_forward_up,
                                workspace.normalized, workspace.up,
                                KIPP_FEED_FORWARD_LENGTH,
                                KIPP_EMBEDDING_LENGTH, totalTokens,
                                useMatrix);
        metal_encode_swiglu(encoder, model, workspace, totalTokens);
        if (useMatrix) {
            metal_encode_stage(encoder, model, workspace, workspace.gate,
                               KIPP_FEED_FORWARD_LENGTH * totalTokens);
        }
        metal_encode_projection(encoder, model, workspace,
                                &layer->feed_forward_down, workspace.gate,
                                workspace.projection, KIPP_EMBEDDING_LENGTH,
                                KIPP_FEED_FORWARD_LENGTH, totalTokens,
                                useMatrix);
        metal_encode_residual(encoder, model, workspace, totalTokens);
    }
    for (uint32_t part = 0; part < partCount; ++part) {
        const metal_round_part *p = &parts[part];
        if (p->finishLogits == NULL) {
            continue;
        }
        NSUInteger lastOffset = (NSUInteger)(p->slot + p->tokens - 1) *
                                KIPP_EMBEDDING_LENGTH * sizeof(float);
        metal_encode_norm(encoder, model, workspace.x, lastOffset,
                          &model.view->weights.output_norm,
                          workspace.normalized, KIPP_EMBEDDING_LENGTH, 1);
        metal_encode_matvec(encoder, model,
                            &model.view->weights.token_embedding,
                            workspace.normalized, model.batchLogits,
                            p->logitsOffset, KIPP_VOCAB_SIZE,
                            KIPP_EMBEDDING_LENGTH, 1);
    }

    [encoder endEncoding];
    if (metal_commit_and_wait(commandBuffer, error) != 0) {
        return -1;
    }
    for (uint32_t part = 0; part < partCount; ++part) {
        const metal_round_part *p = &parts[part];
        if (p->finishLogits != NULL) {
            memcpy(p->finishLogits,
                   (const uint8_t *)model.batchLogits.contents +
                       p->logitsOffset,
                   KIPP_VOCAB_SIZE * sizeof(float));
        }
    }
    return 0;
}

/* Per-item cursor for multi-session evaluation. */
typedef struct {
    KippMetalSession *session;
    const kipp_eval_item *item;
    uint32_t progress;
} metal_batch_cursor;

/*
 * Evaluate several sessions together: each round packs a chunk of every
 * unfinished item into the shared workspace, so projections read the
 * weights once per round for all sequences.
 */
static int metal_eval_multi(KippMetalModel *model, kipp_eval_item *items,
                            size_t itemCount, kipp_error *error) {
    KippMetalSession *workspace = model.oracleSession;
    metal_batch_cursor cursors[KIPP_METAL_BATCH];
    uint32_t initialLengths[KIPP_METAL_BATCH];
    for (size_t index = 0; index < itemCount; ++index) {
        kipp_eval_item *item = &items[index];
        if (item->session == NULL) {
            return metal_fail(error, KIPP_ERROR_ARGUMENT,
                              "stateless evaluation cannot be batched");
        }
        KippMetalSession *session =
            (__bridge KippMetalSession *)item->session;
        if (session.model != model) {
            return metal_fail(error, KIPP_ERROR_ARGUMENT,
                              "Metal session belongs to another model");
        }
        if (item->start_position != session.length) {
            return metal_fail(
                error, KIPP_ERROR_ARGUMENT,
                "Metal start position %u does not match session length %u",
                item->start_position, session.length);
        }
        if (item->token_count > session.capacity - session.length) {
            return metal_fail(error, KIPP_ERROR_RANGE,
                              "Metal session append exceeds capacity %u",
                              session.capacity);
        }
        cursors[index] =
            (metal_batch_cursor){session, item, 0};
        initialLengths[index] = session.length;
    }

    for (;;) {
        uint32_t active = 0;
        for (size_t index = 0; index < itemCount; ++index) {
            active += cursors[index].progress <
                      cursors[index].item->token_count;
        }
        if (active == 0) {
            break;
        }
        uint32_t quota = KIPP_METAL_BATCH / active;
        metal_round_part parts[KIPP_METAL_BATCH];
        uint32_t partCount = 0;
        uint32_t slot = 0;
        for (size_t index = 0; index < itemCount; ++index) {
            metal_batch_cursor *cursor = &cursors[index];
            uint32_t remaining =
                cursor->item->token_count - cursor->progress;
            if (remaining == 0) {
                continue;
            }
            uint32_t chunk = remaining < quota ? remaining : quota;
            bool finishes = cursor->progress + chunk ==
                            cursor->item->token_count;
            parts[partCount] = (metal_round_part){
                cursor->session,
                slot,
                chunk,
                cursor->item->start_position + cursor->progress,
                finishes ? cursor->item->logits : NULL,
                (NSUInteger)index * KIPP_VOCAB_SIZE * sizeof(float),
            };
            memcpy((uint32_t *)workspace.tokens.contents + slot,
                   cursor->item->tokens + cursor->progress,
                   (size_t)chunk * sizeof(uint32_t));
            cursor->progress += chunk;
            slot += chunk;
            ++partCount;
        }
        if (metal_encode_round(model, workspace, parts, partCount, slot,
                               error) != 0) {
            for (size_t index = 0; index < itemCount; ++index) {
                cursors[index].session.length = initialLengths[index];
            }
            return -1;
        }
    }
    for (size_t index = 0; index < itemCount; ++index) {
        cursors[index].session.length += cursors[index].item->token_count;
    }
    return 0;
}

static int metal_compile_pipelines(
    id<MTLDevice> device, id<MTLLibrary> *outLibrary,
    NSDictionary<NSString *, id<MTLComputePipelineState>> **outPipelines,
    BOOL *outMatrixKernelAvailable, kipp_error *error) {
    NSError *libraryError = nil;
    NSString *source = [NSString
        stringWithUTF8String:(const char *)kipp_metal_source];
    /* The batched-prefill matrix kernel needs bfloat simdgroup matrices;
     * on devices without them the library compiles without that kernel
     * and prefill stays on the vector path. */
    BOOL matrixKernel = YES;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.preprocessorMacros = @{@"KIPP_ENABLE_BF16_MMA" : @1};
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:options
                                                    error:&libraryError];
    if (library == nil) {
        fprintf(stderr, "kipp-metal: matrix kernel unavailable: %s\n",
                libraryError.localizedDescription.UTF8String);
        matrixKernel = NO;
        options = [[MTLCompileOptions alloc] init];
        library = [device newLibraryWithSource:source
                                       options:options
                                         error:&libraryError];
    }
    if (library == nil) {
        return metal_fail(error, KIPP_ERROR_UNSUPPORTED,
                          "Metal runtime shader compilation failed: %s",
                          libraryError.localizedDescription.UTF8String);
    }
    /* The matvec kernel is specialized twice through function constant 0:
     * a one-token variant for decode and a KIPP_METAL_TOKEN_TILE variant
     * that reuses each weight read across prefill tokens. */
    NSMutableDictionary<NSString *, NSNumber *> *tiles =
        [NSMutableDictionary dictionaryWithDictionary:@{
          @"kipp_bf16_roundtrip" : @0, @"kipp_embed_gather" : @0,
          @"kipp_rms_norm" : @0, @"kipp_head_norm" : @0, @"kipp_rope" : @0,
          @"kipp_kv_write" : @0, @"kipp_flash_gqa" : @0,
          @"kipp_residual_add" : @0, @"kipp_swiglu" : @0,
          @"kipp_bf16_stage" : @0,
          @"kipp_matvec_bf16_decode" : @1,
          @"kipp_matvec_bf16_prefill" : @KIPP_METAL_TOKEN_TILE,
        }];
    if (matrixKernel) {
        tiles[@"kipp_matmul_bf16"] = @0;
    }
    NSMutableDictionary *pipelines =
        [NSMutableDictionary dictionaryWithCapacity:tiles.count];
    for (NSString *name in tiles) {
        uint32_t tile = tiles[name].unsignedIntValue;
        id<MTLFunction> function;
        NSError *pipelineError = nil;
        if (tile == 0) {
            function = [library newFunctionWithName:name];
        } else {
            MTLFunctionConstantValues *values =
                [[MTLFunctionConstantValues alloc] init];
            [values setConstantValue:&tile type:MTLDataTypeUInt atIndex:0];
            function = [library newFunctionWithName:@"kipp_matvec_bf16"
                                     constantValues:values
                                              error:&pipelineError];
        }
        id<MTLComputePipelineState> pipeline =
            function == nil
                ? nil
                : [device newComputePipelineStateWithFunction:function
                                                        error:&pipelineError];
        if (pipeline == nil) {
            return metal_fail(error, KIPP_ERROR_UNSUPPORTED,
                              "unable to create Metal pipeline %s: %s",
                              name.UTF8String,
                              pipelineError.localizedDescription.UTF8String);
        }
        NSUInteger required = KIPP_METAL_GROUP_THREADS;
        if ([name isEqualToString:@"kipp_rms_norm"] ||
            [name isEqualToString:@"kipp_flash_gqa"]) {
            required = KIPP_METAL_NORM_THREADS;
        }
        (void)tile;
        if (pipeline.maxTotalThreadsPerThreadgroup < required) {
            return metal_fail(
                error, KIPP_ERROR_UNSUPPORTED,
                "Metal pipeline %s supports only %lu threads per group",
                name.UTF8String,
                (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
        }
        pipelines[name] = pipeline;
    }
    *outLibrary = library;
    *outPipelines = pipelines;
    *outMatrixKernelAvailable = matrixKernel;
    return 0;
}

static int metal_model_create(const kipp_model_view *view, void **backendModel,
                              kipp_error *error) {
    metal_clear_error(error);
    if (view == NULL || backendModel == NULL) {
        return metal_fail(error, KIPP_ERROR_ARGUMENT,
                          "Metal model arguments are required");
    }
    *backendModel = NULL;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return metal_fail(error, KIPP_ERROR_UNSUPPORTED,
                              "no Metal device is available");
        }
        NSUInteger pageSize = (NSUInteger)getpagesize();
        if (view->mapping_size > NSUIntegerMax - (pageSize - 1)) {
            return metal_fail(error, KIPP_ERROR_MEMORY,
                              "mapped model length overflows Metal size");
        }
        NSUInteger bufferLength =
            ((NSUInteger)view->mapping_size + pageSize - 1) &
            ~(pageSize - 1);
        if (bufferLength > device.maxBufferLength) {
            return metal_fail(error, KIPP_ERROR_MEMORY,
                              "model mapping exceeds Metal max buffer length");
        }

        id<MTLLibrary> library = nil;
        NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines = nil;
        BOOL matrixKernel = NO;
        if (metal_compile_pipelines(device, &library, &pipelines,
                                    &matrixKernel, error) != 0) {
            return -1;
        }

        id<MTLBuffer> weights =
            [device newBufferWithBytesNoCopy:(void *)view->mapping
                                      length:bufferLength
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        if (weights == nil) {
            return metal_fail(error, KIPP_ERROR_MEMORY,
                              "unable to wrap mmap in no-copy Metal storage");
        }
        KippMetalModel *model = [[KippMetalModel alloc] init];
        model.device = device;
        model.library = library;
        model.queue = [device newCommandQueue];
        model.weights = weights;
        model.pipelines = pipelines;
        model.view = view;
        model.matrixKernelAvailable = matrixKernel;
        model.batchLogits = metal_new_shared_buffer(
            device, (NSUInteger)KIPP_METAL_BATCH * KIPP_VOCAB_SIZE *
                        sizeof(float));
        if (model.batchLogits == nil) {
            return metal_fail(error, KIPP_ERROR_MEMORY,
                              "unable to allocate batched logits storage");
        }
        if (model.queue == nil) {
            return metal_fail(error, KIPP_ERROR_INTERNAL,
                              "unable to create Metal command queue");
        }
        model.oracleSession =
            metal_session_new(model, KIPP_METAL_ORACLE_CAPACITY, error);
        if (model.oracleSession == nil) {
            return -1;
        }
        *backendModel = (__bridge_retained void *)model;
    }
    return 0;
}

static void metal_model_destroy(void *backendModel) {
    if (backendModel != NULL) {
        (void)CFBridgingRelease(backendModel);
    }
}

static int metal_session_create(void *backendModel, uint32_t capacity,
                                void **backendSession, kipp_error *error) {
    metal_clear_error(error);
    if (backendModel == NULL || backendSession == NULL) {
        return metal_fail(error, KIPP_ERROR_ARGUMENT,
                          "Metal session arguments are required");
    }
    *backendSession = NULL;
    @autoreleasepool {
        KippMetalModel *model = (__bridge KippMetalModel *)backendModel;
        KippMetalSession *session = metal_session_new(model, capacity, error);
        if (session == nil) {
            return -1;
        }
        *backendSession = (__bridge_retained void *)session;
    }
    return 0;
}

static void metal_session_destroy(void *backendSession) {
    if (backendSession != NULL) {
        (void)CFBridgingRelease(backendSession);
    }
}

static int metal_session_reset(void *backendSession, kipp_error *error) {
    metal_clear_error(error);
    if (backendSession == NULL) {
        return metal_fail(error, KIPP_ERROR_ARGUMENT,
                          "Metal session is required");
    }
    KippMetalSession *session = (__bridge KippMetalSession *)backendSession;
    session.length = 0;
    return 0;
}

static int metal_session_truncate(void *backendSession, uint32_t length,
                                  kipp_error *error) {
    metal_clear_error(error);
    if (backendSession == NULL) {
        return metal_fail(error, KIPP_ERROR_ARGUMENT,
                          "Metal session is required");
    }
    KippMetalSession *session = (__bridge KippMetalSession *)backendSession;
    if (length > session.length) {
        return metal_fail(error, KIPP_ERROR_RANGE,
                          "cannot truncate Metal session of length %u to %u",
                          session.length, length);
    }
    session.length = length;
    return 0;
}

static int metal_eval(void *backendModel, kipp_eval_item *items,
                      size_t itemCount, kipp_error *error) {
    metal_clear_error(error);
    if (backendModel == NULL || items == NULL || itemCount == 0) {
        return metal_fail(error, KIPP_ERROR_ARGUMENT,
                          "Metal evaluation items are required");
    }
    @autoreleasepool {
        KippMetalModel *model = (__bridge KippMetalModel *)backendModel;
        for (size_t itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
            kipp_eval_item *item = &items[itemIndex];
            if (item->tokens == NULL || item->token_count == 0 ||
                item->logits == NULL) {
                return metal_fail(error, KIPP_ERROR_ARGUMENT,
                                  "Metal evaluation item is incomplete");
            }
        }
        if (itemCount > 1) {
            if (itemCount > KIPP_METAL_BATCH) {
                return metal_fail(error, KIPP_ERROR_RANGE,
                                  "Metal batch accepts at most %u items",
                                  KIPP_METAL_BATCH);
            }
            return metal_eval_multi(model, items, itemCount, error);
        }

        kipp_eval_item *item = &items[0];
        KippMetalSession *session;
        if (item->session == NULL) {
            if (item->start_position != 0 ||
                item->token_count > KIPP_METAL_ORACLE_CAPACITY) {
                return metal_fail(
                    error, KIPP_ERROR_RANGE,
                    "stateless Metal evaluation must start at zero and "
                    "fit %u tokens",
                    KIPP_METAL_ORACLE_CAPACITY);
            }
            session = model.oracleSession;
            session.length = 0;
        } else {
            session = (__bridge KippMetalSession *)item->session;
            if (session.model != model) {
                return metal_fail(error, KIPP_ERROR_ARGUMENT,
                                  "Metal session belongs to another model");
            }
        }
        if (item->start_position != session.length) {
            return metal_fail(
                error, KIPP_ERROR_ARGUMENT,
                "Metal start position %u does not match session length %u",
                item->start_position, session.length);
        }
        if (item->token_count > session.capacity - session.length) {
            return metal_fail(error, KIPP_ERROR_RANGE,
                              "Metal session append exceeds capacity %u",
                              session.capacity);
        }
        uint32_t initialLength = session.length;
        for (uint32_t done = 0; done < item->token_count;) {
            uint32_t batch = item->token_count - done;
            if (batch > KIPP_METAL_BATCH) {
                batch = KIPP_METAL_BATCH;
            }
            bool finalBatch = done + batch == item->token_count;
            metal_round_part part = {
                session,
                0,
                batch,
                item->start_position + done,
                finalBatch ? item->logits : NULL,
                0,
            };
            memcpy(session.tokens.contents, item->tokens + done,
                   (size_t)batch * sizeof(uint32_t));
            if (metal_encode_round(model, session, &part, 1, batch,
                                   error) != 0) {
                session.length = initialLength;
                return -1;
            }
            session.length += batch;
            done += batch;
        }
    }
    return 0;
}

const kipp_backend_ops *kipp_metal_backend_operations(void) {
    static const kipp_backend_ops operations = {
        metal_model_create,    metal_model_destroy, metal_session_create,
        metal_session_destroy, metal_session_reset, metal_eval,
        metal_session_truncate,
    };
    return &operations;
}

const char *kipp_metal_device_name(void) {
    static char name[256];
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return "unavailable";
        }
        (void)snprintf(name, sizeof(name), "%s", device.name.UTF8String);
    }
    return name;
}

static uint16_t metal_test_float_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

static int metal_expect_near(const char *name, const float *actual,
                             const float *expected, size_t count,
                             float tolerance, kipp_error *error) {
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(actual[index]) ||
            fabsf(actual[index] - expected[index]) > tolerance) {
            return metal_fail(
                error, KIPP_ERROR_INTERNAL,
                "Metal operator %s mismatch at %zu: %.9g != %.9g", name,
                index, actual[index], expected[index]);
        }
    }
    return 0;
}

int kipp_metal_run_operator_tests(kipp_error *error) {
    metal_clear_error(error);
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return metal_fail(error, KIPP_ERROR_UNSUPPORTED,
                              "no Metal device is available");
        }
        id<MTLLibrary> library = nil;
        NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines = nil;
        BOOL matrixKernel = NO;
        if (metal_compile_pipelines(device, &library, &pipelines,
                                    &matrixKernel, error) != 0) {
            return -1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        {
            float inputValues[] = {1.0f, -2.0f, 1.00390625f};
            uint16_t expectedBits[] = {
                metal_test_float_to_bf16(inputValues[0]),
                metal_test_float_to_bf16(inputValues[1]),
                metal_test_float_to_bf16(inputValues[2]),
            };
            float expectedValues[3];
            for (size_t i = 0; i < 3; ++i) {
                uint32_t bits = (uint32_t)expectedBits[i] << 16;
                memcpy(&expectedValues[i], &bits, sizeof(bits));
            }
            id<MTLBuffer> input =
                [device newBufferWithBytes:inputValues
                                    length:sizeof(inputValues)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> bits =
                metal_new_shared_buffer(device, sizeof(expectedBits));
            id<MTLBuffer> output =
                metal_new_shared_buffer(device, sizeof(expectedValues));
            uint32_t count = 3;
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch(command, pipelines[@"kipp_bf16_roundtrip"], count,
                           ^(id<MTLComputeCommandEncoder> encoder) {
                             [encoder setBuffer:input offset:0 atIndex:0];
                             [encoder setBuffer:bits offset:0 atIndex:1];
                             [encoder setBuffer:output offset:0 atIndex:2];
                             [encoder setBytes:&count
                                        length:sizeof(count)
                                       atIndex:3];
                           });
            if (metal_commit_and_wait(command, error) != 0 ||
                memcmp(bits.contents, expectedBits, sizeof(expectedBits)) != 0 ||
                metal_expect_near("bf16", output.contents, expectedValues, 3,
                                  0.0f, error) != 0) {
                if (error != NULL && error->code == KIPP_OK) {
                    return metal_fail(error, KIPP_ERROR_INTERNAL,
                                      "Metal BF16 bit conversion mismatch");
                }
                return -1;
            }
        }

        {
            uint16_t *embedding =
                calloc(2 * KIPP_EMBEDDING_LENGTH, sizeof(*embedding));
            float *expected =
                malloc(2 * KIPP_EMBEDDING_LENGTH * sizeof(*expected));
            if (embedding == NULL || expected == NULL) {
                free(embedding);
                free(expected);
                return metal_fail(error, KIPP_ERROR_MEMORY,
                                  "operator test allocation failed");
            }
            for (uint32_t i = 0; i < KIPP_EMBEDDING_LENGTH; ++i) {
                embedding[i] = metal_test_float_to_bf16(-1.0f);
                embedding[KIPP_EMBEDDING_LENGTH + i] =
                    metal_test_float_to_bf16(2.0f);
                expected[i] = 2.0f;
                expected[KIPP_EMBEDDING_LENGTH + i] = -1.0f;
            }
            uint32_t tokenIds[] = {1, 0};
            id<MTLBuffer> weights = [device
                newBufferWithBytes:embedding
                           length:2 * KIPP_EMBEDDING_LENGTH * sizeof(*embedding)
                          options:MTLResourceStorageModeShared];
            id<MTLBuffer> tokens =
                [device newBufferWithBytes:tokenIds
                                    length:sizeof(tokenIds)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> output = metal_new_shared_buffer(
                device, 2 * KIPP_EMBEDDING_LENGTH * sizeof(float));
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch_2d(command, pipelines[@"kipp_embed_gather"],
                              KIPP_EMBEDDING_LENGTH, 2,
                              ^(id<MTLComputeCommandEncoder> encoder) {
                                [encoder setBuffer:weights offset:0 atIndex:0];
                                [encoder setBuffer:tokens offset:0 atIndex:1];
                                [encoder setBuffer:output offset:0 atIndex:2];
                              });
            int status = metal_commit_and_wait(command, error);
            if (status == 0) {
                status = metal_expect_near(
                    "embedding", output.contents, expected,
                    2 * KIPP_EMBEDDING_LENGTH, 0.0f, error);
            }
            free(embedding);
            free(expected);
            if (status != 0) {
                return -1;
            }
        }

        {
            /* Two tokens exercise the per-token threadgroup split. */
            float inputValues[] = {3.0f, 4.0f, 6.0f, 8.0f};
            uint16_t weightValues[] = {
                metal_test_float_to_bf16(1.0f),
                metal_test_float_to_bf16(1.0f),
            };
            float scaleOne = 1.0f / sqrtf(12.5f);
            float scaleTwo = 1.0f / sqrtf(50.0f);
            float expected[] = {3.0f * scaleOne, 4.0f * scaleOne,
                                6.0f * scaleTwo, 8.0f * scaleTwo};
            id<MTLBuffer> input =
                [device newBufferWithBytes:inputValues
                                    length:sizeof(inputValues)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> weight =
                [device newBufferWithBytes:weightValues
                                    length:sizeof(weightValues)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> output =
                metal_new_shared_buffer(device, sizeof(expected));
            metal_norm_params params = {2, 0.0f};
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch_groups(command, pipelines[@"kipp_rms_norm"], 2, 1,
                                  KIPP_METAL_NORM_THREADS,
                                  ^(id<MTLComputeCommandEncoder> encoder) {
                                    [encoder setBuffer:input
                                                offset:0
                                               atIndex:0];
                                    [encoder setBuffer:weight
                                                offset:0
                                               atIndex:1];
                                    [encoder setBuffer:output
                                                offset:0
                                               atIndex:2];
                                    [encoder setBytes:&params
                                               length:sizeof(params)
                                              atIndex:3];
                                  });
            if (metal_commit_and_wait(command, error) != 0 ||
                metal_expect_near("rms_norm", output.contents, expected, 4,
                                  1.0e-6f, error) != 0) {
                return -1;
            }
        }

        {
            uint16_t weights[] = {
                metal_test_float_to_bf16(1.0f),
                metal_test_float_to_bf16(2.0f),
                metal_test_float_to_bf16(3.0f),
                metal_test_float_to_bf16(-1.0f),
                metal_test_float_to_bf16(0.5f),
                metal_test_float_to_bf16(0.0f),
            };
            /* Two tokens through the scalar (columns % 128 != 0) path. */
            float inputValues[] = {2.0f, -1.0f, 0.5f, 1.0f, 1.0f, 2.0f};
            float expected[] = {1.5f, -2.5f, 9.0f, -0.5f};
            id<MTLBuffer> weight =
                [device newBufferWithBytes:weights
                                    length:sizeof(weights)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> input =
                [device newBufferWithBytes:inputValues
                                    length:sizeof(inputValues)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> output =
                metal_new_shared_buffer(device, sizeof(expected));
            metal_matvec_params params = {2, 3, 2};
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch_groups(command,
                                  pipelines[@"kipp_matvec_bf16_prefill"], 1,
                                  1, KIPP_METAL_GROUP_THREADS,
                                  ^(id<MTLComputeCommandEncoder> encoder) {
                                    [encoder setBuffer:weight
                                                offset:0
                                               atIndex:0];
                                    [encoder setBuffer:input
                                                offset:0
                                               atIndex:1];
                                    [encoder setBuffer:output
                                                offset:0
                                               atIndex:2];
                                    [encoder setBytes:&params
                                               length:sizeof(params)
                                              atIndex:3];
                                  });
            if (metal_commit_and_wait(command, error) != 0 ||
                metal_expect_near("matvec", output.contents, expected, 4,
                                  1.0e-6f, error) != 0) {
                return -1;
            }
            metal_matvec_params decodeParams = {2, 3, 1};
            id<MTLCommandBuffer> decodeCommand = [queue commandBuffer];
            metal_dispatch_groups(decodeCommand,
                                  pipelines[@"kipp_matvec_bf16_decode"], 1, 1,
                                  KIPP_METAL_GROUP_THREADS,
                                  ^(id<MTLComputeCommandEncoder> encoder) {
                                    [encoder setBuffer:weight
                                                offset:0
                                               atIndex:0];
                                    [encoder setBuffer:input
                                                offset:0
                                               atIndex:1];
                                    [encoder setBuffer:output
                                                offset:0
                                               atIndex:2];
                                    [encoder setBytes:&decodeParams
                                               length:sizeof(decodeParams)
                                              atIndex:3];
                                  });
            if (metal_commit_and_wait(decodeCommand, error) != 0 ||
                metal_expect_near("matvec-decode", output.contents, expected,
                                  2, 1.0e-6f, error) != 0) {
                return -1;
            }
        }

        {
            float states[KIPP_ATTENTION_HEAD_DIM];
            uint16_t weights[KIPP_ATTENTION_HEAD_DIM];
            for (uint32_t i = 0; i < KIPP_ATTENTION_HEAD_DIM; ++i) {
                states[i] = 1.0f;
                weights[i] = metal_test_float_to_bf16(1.0f);
            }
            id<MTLBuffer> stateBuffer =
                [device newBufferWithBytes:states
                                    length:sizeof(states)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> weightBuffer =
                [device newBufferWithBytes:weights
                                    length:sizeof(weights)
                                   options:MTLResourceStorageModeShared];
            metal_head_norm_params params = {1, 1, KIPP_METAL_RMS_EPSILON};
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch_groups(command, pipelines[@"kipp_head_norm"], 1, 1,
                                  KIPP_METAL_GROUP_THREADS,
                                  ^(id<MTLComputeCommandEncoder> encoder) {
                                    [encoder setBuffer:stateBuffer
                                                offset:0
                                               atIndex:0];
                                    [encoder setBuffer:weightBuffer
                                                offset:0
                                               atIndex:1];
                                    [encoder setBytes:&params
                                               length:sizeof(params)
                                              atIndex:2];
                                  });
            float expected[KIPP_ATTENTION_HEAD_DIM];
            float normalized =
                1.0f / sqrtf(1.0f + KIPP_METAL_RMS_EPSILON);
            for (uint32_t i = 0; i < KIPP_ATTENTION_HEAD_DIM; ++i) {
                expected[i] = normalized;
            }
            if (metal_commit_and_wait(command, error) != 0 ||
                metal_expect_near("head_norm", stateBuffer.contents, expected,
                                  KIPP_ATTENTION_HEAD_DIM, 1.0e-6f,
                                  error) != 0) {
                return -1;
            }
        }

        {
            float states[KIPP_ATTENTION_HEAD_DIM] = {0};
            states[0] = 1.0f;
            states[KIPP_ATTENTION_HEAD_DIM / 2] = 3.0f;
            id<MTLBuffer> stateBuffer =
                [device newBufferWithBytes:states
                                    length:sizeof(states)
                                   options:MTLResourceStorageModeShared];
            metal_rope_params params = {1, 1, 10000.0f};
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch_2d(command, pipelines[@"kipp_rope"],
                              KIPP_ATTENTION_HEAD_DIM / 2, 1,
                              ^(id<MTLComputeCommandEncoder> encoder) {
                                [encoder setBuffer:stateBuffer
                                            offset:0
                                           atIndex:0];
                                [encoder setBytes:&params
                                           length:sizeof(params)
                                          atIndex:1];
                              });
            float expected[KIPP_ATTENTION_HEAD_DIM] = {0};
            expected[0] = cosf(1.0f) - 3.0f * sinf(1.0f);
            expected[KIPP_ATTENTION_HEAD_DIM / 2] =
                3.0f * cosf(1.0f) + sinf(1.0f);
            if (metal_commit_and_wait(command, error) != 0 ||
                metal_expect_near("rope", stateBuffer.contents, expected,
                                  KIPP_ATTENTION_HEAD_DIM, 2.0e-5f,
                                  error) != 0) {
                return -1;
            }
        }

        {
            float residualValues[] = {1.0f, 2.0f};
            float additionValues[] = {3.0f, 4.0f};
            float expectedResidual[] = {4.0f, 6.0f};
            float gateValues[] = {1.0f, 0.0f};
            float upValues[] = {2.0f, 3.0f};
            float expectedGate[] = {1.4621172f, 0.0f};
            id<MTLBuffer> residual =
                [device newBufferWithBytes:residualValues
                                    length:sizeof(residualValues)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> addition =
                [device newBufferWithBytes:additionValues
                                    length:sizeof(additionValues)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> gate =
                [device newBufferWithBytes:gateValues
                                    length:sizeof(gateValues)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> up =
                [device newBufferWithBytes:upValues
                                    length:sizeof(upValues)
                                   options:MTLResourceStorageModeShared];
            uint32_t count = 2;
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch(command, pipelines[@"kipp_residual_add"], count,
                           ^(id<MTLComputeCommandEncoder> encoder) {
                             [encoder setBuffer:residual offset:0 atIndex:0];
                             [encoder setBuffer:addition offset:0 atIndex:1];
                             [encoder setBytes:&count
                                        length:sizeof(count)
                                       atIndex:2];
                           });
            metal_dispatch(command, pipelines[@"kipp_swiglu"], count,
                           ^(id<MTLComputeCommandEncoder> encoder) {
                             [encoder setBuffer:gate offset:0 atIndex:0];
                             [encoder setBuffer:up offset:0 atIndex:1];
                             [encoder setBytes:&count
                                        length:sizeof(count)
                                       atIndex:2];
                           });
            if (metal_commit_and_wait(command, error) != 0 ||
                metal_expect_near("residual", residual.contents,
                                  expectedResidual, 2, 1.0e-6f,
                                  error) != 0 ||
                metal_expect_near("swiglu", gate.contents, expectedGate, 2,
                                  1.0e-6f, error) != 0) {
                return -1;
            }
        }

        if (matrixKernel) {
            /* Integer-valued BF16 matmul: results are exact in FP32. */
            enum { ROWS = 128, COLS = 16, TOKENS = 3 };
            uint16_t *weightsHost = malloc(ROWS * COLS * sizeof(*weightsHost));
            uint16_t *inputHost = malloc(TOKENS * COLS * sizeof(*inputHost));
            float *expected = malloc(TOKENS * ROWS * sizeof(*expected));
            if (weightsHost == NULL || inputHost == NULL || expected == NULL) {
                free(weightsHost);
                free(inputHost);
                free(expected);
                return metal_fail(error, KIPP_ERROR_MEMORY,
                                  "matmul test allocation failed");
            }
            for (uint32_t r = 0; r < ROWS; ++r) {
                for (uint32_t c = 0; c < COLS; ++c) {
                    weightsHost[r * COLS + c] = metal_test_float_to_bf16(
                        (float)((int)((r + 2 * c) % 7) - 3));
                }
            }
            for (uint32_t t = 0; t < TOKENS; ++t) {
                for (uint32_t c = 0; c < COLS; ++c) {
                    inputHost[t * COLS + c] = metal_test_float_to_bf16(
                        (float)((int)((t + c) % 5) - 2));
                }
            }
            for (uint32_t t = 0; t < TOKENS; ++t) {
                for (uint32_t r = 0; r < ROWS; ++r) {
                    float sum = 0.0f;
                    for (uint32_t c = 0; c < COLS; ++c) {
                        sum += (float)((int)((r + 2 * c) % 7) - 3) *
                               (float)((int)((t + c) % 5) - 2);
                    }
                    expected[t * ROWS + r] = sum;
                }
            }
            id<MTLBuffer> weightBuffer =
                [device newBufferWithBytes:weightsHost
                                    length:ROWS * COLS * sizeof(uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> inputBuffer =
                [device newBufferWithBytes:inputHost
                                    length:TOKENS * COLS * sizeof(uint16_t)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = metal_new_shared_buffer(
                device, TOKENS * ROWS * sizeof(float));
            metal_matvec_params params = {ROWS, COLS, TOKENS};
            id<MTLCommandBuffer> command = [queue commandBuffer];
            metal_dispatch_groups(command, pipelines[@"kipp_matmul_bf16"], 1,
                                  1, KIPP_METAL_GROUP_THREADS,
                                  ^(id<MTLComputeCommandEncoder> encoder) {
                                    [encoder setBuffer:weightBuffer
                                                offset:0
                                               atIndex:0];
                                    [encoder setBuffer:inputBuffer
                                                offset:0
                                               atIndex:1];
                                    [encoder setBuffer:outputBuffer
                                                offset:0
                                               atIndex:2];
                                    [encoder setBytes:&params
                                               length:sizeof(params)
                                              atIndex:3];
                                  });
            int status = metal_commit_and_wait(command, error);
            if (status == 0) {
                status = metal_expect_near("matmul", outputBuffer.contents,
                                           expected, TOKENS * ROWS, 0.0f,
                                           error);
            }
            free(weightsHost);
            free(inputHost);
            free(expected);
            if (status != 0) {
                return -1;
            }
        }

        {
            const uint32_t capacity = 2;
            const size_t kvValues =
                capacity * KIPP_ATTENTION_HEAD_COUNT_KV *
                KIPP_ATTENTION_HEAD_DIM;
            float key[KIPP_ATTENTION_HEAD_COUNT_KV *
                      KIPP_ATTENTION_HEAD_DIM] = {0};
            float value[KIPP_ATTENTION_HEAD_COUNT_KV *
                        KIPP_ATTENTION_HEAD_DIM] = {0};
            float query[KIPP_ATTENTION_HEAD_COUNT *
                        KIPP_ATTENTION_HEAD_DIM] = {0};
            for (uint32_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT_KV;
                 ++head) {
                value[head * KIPP_ATTENTION_HEAD_DIM] = 1.0f;
            }
            id<MTLBuffer> keyBuffer =
                [device newBufferWithBytes:key
                                    length:sizeof(key)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> valueBuffer =
                [device newBufferWithBytes:value
                                    length:sizeof(value)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> queryBuffer =
                [device newBufferWithBytes:query
                                    length:sizeof(query)
                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> keyCache = metal_new_shared_buffer(
                device, kvValues * sizeof(uint16_t));
            id<MTLBuffer> valueCache = metal_new_shared_buffer(
                device, kvValues * sizeof(uint16_t));
            id<MTLBuffer> output = metal_new_shared_buffer(
                device, sizeof(query));
            metal_kv_params position0 = {0, 0, 1, capacity};
            id<MTLCommandBuffer> first = [queue commandBuffer];
            metal_dispatch_2d(first, pipelines[@"kipp_kv_write"],
                              KIPP_ATTENTION_HEAD_COUNT_KV *
                                  KIPP_ATTENTION_HEAD_DIM,
                              1, ^(id<MTLComputeCommandEncoder> encoder) {
                                [encoder setBuffer:keyBuffer
                                            offset:0
                                           atIndex:0];
                                [encoder setBuffer:valueBuffer
                                            offset:0
                                           atIndex:1];
                                [encoder setBuffer:keyCache
                                            offset:0
                                           atIndex:2];
                                [encoder setBuffer:valueCache
                                            offset:0
                                           atIndex:3];
                                [encoder setBytes:&position0
                                           length:sizeof(position0)
                                          atIndex:4];
                              });
            if (metal_commit_and_wait(first, error) != 0) {
                return -1;
            }
            for (uint32_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT_KV;
                 ++head) {
                ((float *)valueBuffer.contents)
                    [head * KIPP_ATTENTION_HEAD_DIM] = 3.0f;
            }
            metal_kv_params position1 = {0, 1, 1, capacity};
            id<MTLCommandBuffer> second = [queue commandBuffer];
            metal_dispatch_2d(second, pipelines[@"kipp_kv_write"],
                              KIPP_ATTENTION_HEAD_COUNT_KV *
                                  KIPP_ATTENTION_HEAD_DIM,
                              1, ^(id<MTLComputeCommandEncoder> encoder) {
                                [encoder setBuffer:keyBuffer
                                            offset:0
                                           atIndex:0];
                                [encoder setBuffer:valueBuffer
                                            offset:0
                                           atIndex:1];
                                [encoder setBuffer:keyCache
                                            offset:0
                                           atIndex:2];
                                [encoder setBuffer:valueCache
                                            offset:0
                                           atIndex:3];
                                [encoder setBytes:&position1
                                           length:sizeof(position1)
                                          atIndex:4];
                              });
            metal_dispatch_groups(
                second, pipelines[@"kipp_flash_gqa"],
                KIPP_ATTENTION_HEAD_COUNT, 1, KIPP_METAL_GQA_THREADS,
                ^(id<MTLComputeCommandEncoder> encoder) {
                  [encoder setBuffer:queryBuffer offset:0 atIndex:0];
                  [encoder setBuffer:keyCache offset:0 atIndex:1];
                  [encoder setBuffer:valueCache offset:0 atIndex:2];
                  [encoder setBuffer:output offset:0 atIndex:3];
                  [encoder setBytes:&position1
                             length:sizeof(position1)
                            atIndex:4];
                });
            if (metal_commit_and_wait(second, error) != 0) {
                return -1;
            }
            float *outputValues = output.contents;
            for (uint32_t head = 0; head < KIPP_ATTENTION_HEAD_COUNT; ++head) {
                size_t base = (size_t)head * KIPP_ATTENTION_HEAD_DIM;
                if (fabsf(outputValues[base] - 2.0f) > 1.0e-6f) {
                    return metal_fail(error, KIPP_ERROR_INTERNAL,
                                      "Metal cached GQA mismatch at head %u",
                                      head);
                }
            }
        }
    }
    return 0;
}
