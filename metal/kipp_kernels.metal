#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup_matrix>
using namespace metal;

constant float KIPP_FLT_LOWEST = -3.402823466e+38f;
constant uint KIPP_HEAD_DIM = 128;
constant uint KIPP_KV_HEADS = 8;
constant uint KIPP_NORM_THREADS = 256;
constant uint KIPP_NORM_GROUPS = KIPP_NORM_THREADS / 32;
constant uint KIPP_MV_ROWS_PER_GROUP = 4;
constant uint KIPP_MV_MAX_TILE = 8;
/* Specialized at pipeline-creation time: 1 for decode, KIPP_MV_MAX_TILE for
 * batched prefill. */
constant uint KIPP_MV_TOKEN_TILE [[function_constant(0)]];
/* Per-model dimensions, set when the model compiles its pipelines. The
 * hidden width and query-head count are the only dims that vary across the
 * Qwen3 dense family; head_dim and KV heads are family constants above. */
constant uint KIPP_EMBED [[function_constant(1)]];
constant uint KIPP_Q_HEADS [[function_constant(2)]];
constant uint KIPP_HALF_HEAD_DIM = KIPP_HEAD_DIM / 2;
constant uint KIPP_GQA_GROUPS = 8;
constant uint KIPP_KV_VALUES_PER_TOKEN = KIPP_KV_HEADS * KIPP_HEAD_DIM;

inline float kipp_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline float4 kipp_bf16x4_to_float4(ushort4 value) {
    return float4(as_type<float>(uint(value.x) << 16),
                  as_type<float>(uint(value.y) << 16),
                  as_type<float>(uint(value.z) << 16),
                  as_type<float>(uint(value.w) << 16));
}

inline ushort kipp_float_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint rounding = 0x7fffu + ((bits >> 16) & 1u);
    return ushort((bits + rounding) >> 16);
}

struct MatvecParams {
    uint rows;
    uint columns;
    uint token_count;
};

struct NormParams {
    uint length;
    float epsilon;
};

struct HeadNormParams {
    uint head_count;
    uint token_count;
    float epsilon;
};

struct RopeParams {
    uint head_count;
    uint start_position;
    float theta;
};

struct KvParams {
    uint layer;
    uint start_position;
    uint token_count;
    uint capacity;
};

kernel void kipp_bf16_roundtrip(device const float *input [[buffer(0)]],
                                device ushort *bits [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant uint &count [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= count) {
        return;
    }
    ushort rounded = kipp_float_to_bf16(input[index]);
    bits[index] = rounded;
    output[index] = kipp_bf16_to_float(rounded);
}

/* One thread per (dimension, token). */
kernel void kipp_embed_gather(device const ushort *embedding [[buffer(0)]],
                              device const uint *tokens [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint dimension = gid.x;
    uint token = gid.y;
    if (dimension >= KIPP_EMBED) {
        return;
    }
    output[ulong(token) * KIPP_EMBED + dimension] = kipp_bf16_to_float(
        embedding[ulong(tokens[token]) * KIPP_EMBED + dimension]);
}

/* One 256-thread threadgroup per token; parallel square-sum reduction. */
kernel void kipp_rms_norm(device const float *input [[buffer(0)]],
                          device const ushort *weight [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant NormParams &params [[buffer(3)]],
                          uint token [[threadgroup_position_in_grid]],
                          uint thread_id [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[KIPP_NORM_GROUPS];
    device const float *x = input + ulong(token) * params.length;
    device float *out = output + ulong(token) * params.length;
    float sum = 0.0f;
    for (uint i = thread_id; i < params.length; i += KIPP_NORM_THREADS) {
        sum += x[i] * x[i];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        partials[group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < KIPP_NORM_GROUPS; ++g) {
        total += partials[g];
    }
    float scale = rsqrt(total / float(params.length) + params.epsilon);
    for (uint i = thread_id; i < params.length; i += KIPP_NORM_THREADS) {
        out[i] = x[i] * scale * kipp_bf16_to_float(weight[i]);
    }
}

/*
 * One simdgroup per (head, token) pair; each lane owns four of the 128
 * dimensions, so the square sum reduces with a single simd_sum.
 */
kernel void kipp_head_norm(device float *states [[buffer(0)]],
                           device const ushort *weight [[buffer(1)]],
                           constant HeadNormParams &params [[buffer(2)]],
                           uint group_id [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint group [[simdgroup_index_in_threadgroup]]) {
    uint pair = group_id * KIPP_MV_ROWS_PER_GROUP + group;
    if (pair >= params.head_count * params.token_count) {
        return;
    }
    device float4 *values =
        (device float4 *)(states + ulong(pair) * KIPP_HEAD_DIM);
    float4 v = values[lane];
    float square_sum = simd_sum(dot(v, v));
    float scale =
        rsqrt(square_sum / float(KIPP_HEAD_DIM) + params.epsilon);
    float4 w = kipp_bf16x4_to_float4(
        ((device const ushort4 *)weight)[lane]);
    values[lane] = v * scale * w;
}

/* One thread per (rotation pair, head, token). */
kernel void kipp_rope(device float *states [[buffer(0)]],
                      constant RopeParams &params [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint index = gid.x % KIPP_HALF_HEAD_DIM;
    uint head = gid.x / KIPP_HALF_HEAD_DIM;
    uint token = gid.y;
    if (head >= params.head_count) {
        return;
    }
    device float *values =
        states + (ulong(token) * params.head_count + head) * KIPP_HEAD_DIM;
    float frequency =
        pow(params.theta, -(2.0f * float(index)) / float(KIPP_HEAD_DIM));
    float angle = float(params.start_position + token) * frequency;
    float cosine = cos(angle);
    float sine = sin(angle);
    float first = values[index];
    float second = values[index + KIPP_HALF_HEAD_DIM];
    values[index] = first * cosine - second * sine;
    values[index + KIPP_HALF_HEAD_DIM] = second * cosine + first * sine;
}

/*
 * Four simdgroups per threadgroup, one output row each; lanes read
 * consecutive ushort4 weights so loads coalesce. Each weight read is reused
 * for up to KIPP_MV_TOKEN_TILE tokens, which turns batched prefill from
 * bandwidth-bound into a far denser pass.
 */
kernel void kipp_matvec_bf16(device const ushort *weight [[buffer(0)]],
                             device const float *input [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant MatvecParams &params [[buffer(3)]],
                             uint2 group_id [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint group [[simdgroup_index_in_threadgroup]]) {
    uint row = group_id.x * KIPP_MV_ROWS_PER_GROUP + group;
    if (row >= params.rows) {
        return;
    }
    uint token_base = group_id.y * KIPP_MV_TOKEN_TILE;
    uint tile = min(KIPP_MV_TOKEN_TILE, params.token_count - token_base);
    float accumulators[KIPP_MV_MAX_TILE] = {0.0f};
    device const ushort *weight_row =
        weight + ulong(row) * ulong(params.columns);
    if (params.columns % 128u == 0u) {
        device const ushort4 *weight4 = (device const ushort4 *)weight_row;
        uint vector_count = params.columns / 4u;
        for (uint c = lane; c < vector_count; c += 32u) {
            float4 w = kipp_bf16x4_to_float4(weight4[c]);
            for (uint t = 0; t < KIPP_MV_TOKEN_TILE; ++t) {
                if (t < tile) {
                    device const float4 *in4 = (device const float4 *)(
                        input + ulong(token_base + t) * params.columns);
                    accumulators[t] += dot(w, in4[c]);
                }
            }
        }
    } else {
        for (uint c = lane; c < params.columns; c += 32u) {
            float w = kipp_bf16_to_float(weight_row[c]);
            for (uint t = 0; t < KIPP_MV_TOKEN_TILE; ++t) {
                if (t < tile) {
                    accumulators[t] +=
                        w * input[ulong(token_base + t) * params.columns + c];
                }
            }
        }
    }
    for (uint t = 0; t < KIPP_MV_TOKEN_TILE; ++t) {
        if (t < tile) {
            float total = simd_sum(accumulators[t]);
            if (lane == 0) {
                output[ulong(token_base + t) * params.rows + row] = total;
            }
        }
    }
}

/* Little-endian IEEE fp16 from two device bytes -> float. */
inline float kipp_fp16_bytes(device const uchar *p) {
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

/*
 * Q8_0 matvec, token-tiled exactly like kipp_matvec_bf16 (function constant
 * KIPP_MV_TOKEN_TILE = 1 for decode, KIPP_MV_MAX_TILE for prefill). Weight
 * is rows * (columns/32) blocks of 34 bytes: fp16 scale then int8 qs[32].
 * One simdgroup owns one output row; the 32 lanes stride whole blocks.
 */
kernel void kipp_matvec_q8_0(device const uchar *weight [[buffer(0)]],
                             device const float *input [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant MatvecParams &params [[buffer(3)]],
                             uint2 group_id [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint group [[simdgroup_index_in_threadgroup]]) {
    uint row = group_id.x * KIPP_MV_ROWS_PER_GROUP + group;
    if (row >= params.rows) {
        return;
    }
    uint token_base = group_id.y * KIPP_MV_TOKEN_TILE;
    uint tile = min(KIPP_MV_TOKEN_TILE, params.token_count - token_base);
    uint blocks = params.columns / 32u;
    device const uchar *weight_row = weight + ulong(row) * blocks * 34ul;
    float accumulators[KIPP_MV_MAX_TILE] = {0.0f};
    for (uint b = lane; b < blocks; b += 32u) {
        device const uchar *blk = weight_row + ulong(b) * 34ul;
        float d = kipp_fp16_bytes(blk);
        device const char *qs = (device const char *)(blk + 2);
        for (uint t = 0; t < KIPP_MV_TOKEN_TILE; ++t) {
            if (t < tile) {
                device const float *in =
                    input + ulong(token_base + t) * params.columns + b * 32u;
                float dot = 0.0f;
                for (uint j = 0; j < 32u; ++j) {
                    dot += float(qs[j]) * in[j];
                }
                accumulators[t] += d * dot;
            }
        }
    }
    for (uint t = 0; t < KIPP_MV_TOKEN_TILE; ++t) {
        if (t < tile) {
            float total = simd_sum(accumulators[t]);
            if (lane == 0) {
                output[ulong(token_base + t) * params.rows + row] = total;
            }
        }
    }
}

/*
 * AFFINE4_GS32 matvec, token-tiled. Weight is rows * (columns/32) groups of
 * 20 bytes: 16 packed nibbles (q[2k]=lo, q[2k+1]=hi) then fp16 scale, fp16
 * bias. w = scale*q + bias, folded per group as scale*dot + bias*actsum.
 */
kernel void kipp_matvec_affine4(device const uchar *weight [[buffer(0)]],
                                device const float *input [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant MatvecParams &params [[buffer(3)]],
                                uint2 group_id [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint group [[simdgroup_index_in_threadgroup]]) {
    uint row = group_id.x * KIPP_MV_ROWS_PER_GROUP + group;
    if (row >= params.rows) {
        return;
    }
    uint token_base = group_id.y * KIPP_MV_TOKEN_TILE;
    uint tile = min(KIPP_MV_TOKEN_TILE, params.token_count - token_base);
    uint groups = params.columns / 32u;
    device const uchar *weight_row = weight + ulong(row) * groups * 20ul;
    float accumulators[KIPP_MV_MAX_TILE] = {0.0f};
    for (uint g = lane; g < groups; g += 32u) {
        device const uchar *grp = weight_row + ulong(g) * 20ul;
        float scale = kipp_fp16_bytes(grp + 16);
        float bias = kipp_fp16_bytes(grp + 18);
        for (uint t = 0; t < KIPP_MV_TOKEN_TILE; ++t) {
            if (t < tile) {
                device const float *in =
                    input + ulong(token_base + t) * params.columns + g * 32u;
                float dot = 0.0f;
                float actsum = 0.0f;
                for (uint k = 0; k < 16u; ++k) {
                    uchar p = grp[k];
                    float a0 = in[2 * k];
                    float a1 = in[2 * k + 1];
                    dot += float(p & 0x0fu) * a0 + float(p >> 4) * a1;
                    actsum += a0 + a1;
                }
                accumulators[t] += scale * dot + bias * actsum;
            }
        }
    }
    for (uint t = 0; t < KIPP_MV_TOKEN_TILE; ++t) {
        if (t < tile) {
            float total = simd_sum(accumulators[t]);
            if (lane == 0) {
                output[ulong(token_base + t) * params.rows + row] = total;
            }
        }
    }
}

/* One thread per (KV value, token). */
/* Logical position -> physical KV slot through the session's page table:
 * block_table[pos >> 5] selects the 32-slot physical block, (pos & 31) the
 * slot within it. `params.capacity` is the physical stride (a whole number
 * of blocks). The identity table reproduces the contiguous layout exactly. */
inline ulong kipp_kv_slot(constant KvParams &params,
                          device const uint *block_table, uint position) {
    uint physical = block_table[position >> 5u] * 32u + (position & 31u);
    return ulong(params.layer) * params.capacity + ulong(physical);
}

kernel void kipp_kv_write(device const float *key [[buffer(0)]],
                          device const float *value [[buffer(1)]],
                          device ushort *key_cache [[buffer(2)]],
                          device ushort *value_cache [[buffer(3)]],
                          constant KvParams &params [[buffer(4)]],
                          device const uint *block_table [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint index = gid.x;
    uint token = gid.y;
    if (index >= KIPP_KV_VALUES_PER_TOKEN) {
        return;
    }
    uint position = params.start_position + token;
    ulong cache_index =
        kipp_kv_slot(params, block_table, position) * KIPP_KV_VALUES_PER_TOKEN +
        index;
    ulong state_index = ulong(token) * KIPP_KV_VALUES_PER_TOKEN + index;
    key_cache[cache_index] = kipp_float_to_bf16(key[state_index]);
    value_cache[cache_index] = kipp_float_to_bf16(value[state_index]);
}

/*
 * Causal grouped-query attention with a streaming (online-softmax) scan
 * over the KV cache. One 256-thread threadgroup per (query head, token):
 * each of its eight simdgroups keeps a partial softmax over every eighth
 * source position, and the partials merge in threadgroup memory. The
 * partitioning keeps the GPU busy during single-token decode even at long
 * contexts, and no score buffer is ever materialized.
 */
kernel void kipp_flash_gqa(device const float *query [[buffer(0)]],
                           device const ushort *key_cache [[buffer(1)]],
                           device const ushort *value_cache [[buffer(2)]],
                           device float *output [[buffer(3)]],
                           constant KvParams &params [[buffer(4)]],
                           device const uint *block_table [[buffer(5)]],
                           uint2 group_id [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial_values[KIPP_GQA_GROUPS][KIPP_HEAD_DIM];
    uint thread_id = group * 32u + lane;
    threadgroup float partial_maximum[KIPP_GQA_GROUPS];
    threadgroup float partial_denominator[KIPP_GQA_GROUPS];
    threadgroup uint partial_count[KIPP_GQA_GROUPS];

    uint head = group_id.x;
    uint token = group_id.y;
    uint kv_head = head / (KIPP_Q_HEADS / KIPP_KV_HEADS);
    device const float4 *query4 = (device const float4 *)(
        query + (ulong(token) * KIPP_Q_HEADS + head) * KIPP_HEAD_DIM);
    float4 q = query4[lane];
    const float scale = rsqrt(float(KIPP_HEAD_DIM));
    uint last_source = params.start_position + token;
    ulong head_offset = ulong(kv_head) * KIPP_HEAD_DIM;

    float maximum = 0.0f;
    float denominator = 0.0f;
    float4 accumulator = 0.0f;
    uint count = 0;
    for (uint source = group; source <= last_source;
         source += KIPP_GQA_GROUPS) {
        ulong offset =
            kipp_kv_slot(params, block_table, source) *
                KIPP_KV_VALUES_PER_TOKEN +
            head_offset;
        float4 k = kipp_bf16x4_to_float4(
            ((device const ushort4 *)(key_cache + offset))[lane]);
        float score = simd_sum(dot(q, k)) * scale;
        float4 v = kipp_bf16x4_to_float4(
            ((device const ushort4 *)(value_cache + offset))[lane]);
        if (count == 0) {
            maximum = score;
            denominator = 1.0f;
            accumulator = v;
        } else {
            float new_maximum = max(maximum, score);
            float correction = exp(maximum - new_maximum);
            float weight = exp(score - new_maximum);
            accumulator = accumulator * correction + weight * v;
            denominator = denominator * correction + weight;
            maximum = new_maximum;
        }
        ++count;
    }

    ((threadgroup float4 *)partial_values[group])[lane] = accumulator;
    if (lane == 0) {
        partial_maximum[group] = maximum;
        partial_denominator[group] = denominator;
        partial_count[group] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_id < KIPP_HEAD_DIM) {
        float global_maximum = KIPP_FLT_LOWEST;
        for (uint g = 0; g < KIPP_GQA_GROUPS; ++g) {
            if (partial_count[g] != 0 &&
                partial_maximum[g] > global_maximum) {
                global_maximum = partial_maximum[g];
            }
        }
        float value = 0.0f;
        float total = 0.0f;
        for (uint g = 0; g < KIPP_GQA_GROUPS; ++g) {
            if (partial_count[g] == 0) {
                continue;
            }
            float weight = exp(partial_maximum[g] - global_maximum);
            value += partial_values[g][thread_id] * weight;
            total += partial_denominator[g] * weight;
        }
        output[(ulong(token) * KIPP_Q_HEADS + head) * KIPP_HEAD_DIM +
               thread_id] = value / total;
    }
}

kernel void kipp_residual_add(device float *residual [[buffer(0)]],
                              device const float *addition [[buffer(1)]],
                              constant uint &count [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    if (index < count) {
        residual[index] += addition[index];
    }
}

kernel void kipp_swiglu(device float *gate [[buffer(0)]],
                        device const float *up [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < count) {
        float value = gate[index];
        gate[index] = (value / (1.0f + exp(-value))) * up[index];
    }
}

/* One thread per element: stage FP32 activations as BF16 for the matrix
 * kernel below. */
kernel void kipp_bf16_stage(device const float *input [[buffer(0)]],
                            device ushort *output [[buffer(1)]],
                            constant uint &count [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < count) {
        output[index] = kipp_float_to_bf16(input[index]);
    }
}

#if defined(KIPP_ENABLE_BF16_MMA)
/*
 * Batched-prefill projection using simdgroup matrix row_blocks. Each of the
 * four simdgroups in a threadgroup owns a 32-row x 8-token output tile and
 * walks the shared dimension in 8-wide steps: one transposed activation
 * row_block is reused against four weight row_blocks, so activation traffic
 * collapses compared with the vector kernel. Weights and staged activations
 * are BF16; accumulation is FP32.
 */
constant uint KIPP_MM_SIMDGROUPS = 4;
constant uint KIPP_MM_ROWS_PER_SIMDGROUP = 32;
constant uint KIPP_MM_ROW_FRAGMENTS = KIPP_MM_ROWS_PER_SIMDGROUP / 8;
constant uint KIPP_MM_TOKEN_FRAGMENTS = 2;
constant uint KIPP_MM_TOKEN_TILE = KIPP_MM_TOKEN_FRAGMENTS * 8;

kernel void kipp_matmul_bf16(device const ushort *weight [[buffer(0)]],
                             device const ushort *input [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant MatvecParams &params [[buffer(3)]],
                             uint2 group_id [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float staged[KIPP_MM_SIMDGROUPS][64];
    uint row_base =
        (group_id.x * KIPP_MM_SIMDGROUPS + group) * KIPP_MM_ROWS_PER_SIMDGROUP;
    if (row_base >= params.rows) {
        return;
    }
    uint token_base = group_id.y * KIPP_MM_TOKEN_TILE;
    uint tile = min(KIPP_MM_TOKEN_TILE, params.token_count - token_base);
    device const bfloat *w = (device const bfloat *)weight;
    device const bfloat *x = (device const bfloat *)input;

    simdgroup_float8x8 accumulators[KIPP_MM_ROW_FRAGMENTS]
                                   [KIPP_MM_TOKEN_FRAGMENTS];
    for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS; ++row_block) {
        for (uint tf = 0; tf < KIPP_MM_TOKEN_FRAGMENTS; ++tf) {
            accumulators[row_block][tf] = simdgroup_float8x8(0.0f);
        }
    }
    uint token_blocks = (tile + 7u) / 8u;
    for (uint column = 0; column < params.columns; column += 8u) {
        simdgroup_bfloat8x8 activation[KIPP_MM_TOKEN_FRAGMENTS];
        /* Transposed loads turn [token][column] rows into [column][token]
         * fragments; each weight fragment below is reused against all of
         * them. */
        for (uint tf = 0; tf < KIPP_MM_TOKEN_FRAGMENTS; ++tf) {
            if (tf < token_blocks) {
                simdgroup_load(activation[tf],
                               x + (ulong)(token_base + tf * 8u) *
                                       params.columns +
                                   column,
                               params.columns, 0, true);
            }
        }
        for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS;
             ++row_block) {
            simdgroup_bfloat8x8 weights;
            simdgroup_load(weights,
                           w + (ulong)(row_base + row_block * 8u) *
                                   params.columns +
                               column,
                           params.columns);
            for (uint tf = 0; tf < KIPP_MM_TOKEN_FRAGMENTS; ++tf) {
                if (tf < token_blocks) {
                    simdgroup_multiply_accumulate(
                        accumulators[row_block][tf], weights, activation[tf],
                        accumulators[row_block][tf]);
                }
            }
        }
    }
    for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS; ++row_block) {
        for (uint tf = 0; tf < KIPP_MM_TOKEN_FRAGMENTS; ++tf) {
            uint sub_base = token_base + tf * 8u;
            if (sub_base >= params.token_count) {
                continue;
            }
            uint sub_tile = min(8u, params.token_count - sub_base);
            if (sub_tile == 8u) {
                /* Transposed store writes the [row][token] accumulator into
                 * the [token][row] output. */
                simdgroup_store(accumulators[row_block][tf],
                                output + (ulong)sub_base * params.rows +
                                    row_base + row_block * 8u,
                                params.rows, 0, true);
            } else {
                /* Partial token tile: stage and copy the valid tokens. */
                simdgroup_store(accumulators[row_block][tf], staged[group],
                                8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint index = lane; index < 64u; index += 32u) {
                    uint row = index / 8u;
                    uint token = index % 8u;
                    if (token < sub_tile) {
                        output[(ulong)(sub_base + token) * params.rows +
                               row_base + row_block * 8u + row] =
                            staged[group][index];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
#endif
