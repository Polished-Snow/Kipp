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
/* Q8_0 KV cache: each 32-value block is {bf16 scale; int8 qs[32]} = 34 bytes.
 * A position holds KIPP_KV_VALUES_PER_TOKEN/32 blocks; the scale is bf16, not
 * fp16, to match the CPU reference (src/kipp.c kv_read_row/kv_write_row). */
constant uint KIPP_KV_Q8_BLOCK = 32;
constant uint KIPP_KV_Q8_BLOCK_BYTES = 34;
constant uint KIPP_KV_Q8_POS_BYTES =
    KIPP_KV_VALUES_PER_TOKEN / KIPP_KV_Q8_BLOCK * KIPP_KV_Q8_BLOCK_BYTES;

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
    /* Test hook: 0 = automatic split-K, otherwise a hard cap (1 forces the
     * legacy single-split decode path regardless of position). */
    uint ksplit_cap;
};

/*
 * Split-K decode: contexts past KIPP_KSPLIT_CHUNK positions split the KV
 * scan across up to KIPP_KSPLIT_MAX threadgroups per (head, token), each
 * writing an online-softmax partial that kipp_flash_gqa_reduce merges. The
 * split count derives ONLY from the token's own position, so a token is
 * partitioned - and therefore reduced - identically whether it arrives as
 * single-token decode or inside a multi-row speculative verify; that
 * position-independence is the token-identity contract for speculation.
 */
constant uint KIPP_KSPLIT_MAX = 8;
constant uint KIPP_KSPLIT_CHUNK = 1024;
/* floats per (head, split, token) partial: max, denominator, 2 pad,
 * then the 128 accumulator lanes. */
constant uint KIPP_KSPLIT_STRIDE = 4 + KIPP_HEAD_DIM;

inline uint kipp_ksplit_count(uint last_source, uint cap) {
    uint splits = min(KIPP_KSPLIT_MAX, 1u + last_source / KIPP_KSPLIT_CHUNK);
    return cap != 0u ? min(splits, cap) : splits;
}

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

/* Quantize one 32-value block into {bf16 scale; int8 qs[32]} (symmetric,
 * scale = maxabs/127, round-half-away-from-zero to match the CPU roundf). */
inline void kipp_kv_q8_store(device uchar *block, device const float *v) {
    float amax = 0.0f;
    for (uint i = 0; i < KIPP_KV_Q8_BLOCK; ++i) {
        amax = max(amax, fabs(v[i]));
    }
    float scale = amax / 127.0f;
    float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    *(device ushort *)block = kipp_float_to_bf16(scale);
    device char *qs = (device char *)(block + 2);
    for (uint i = 0; i < KIPP_KV_Q8_BLOCK; ++i) {
        qs[i] = char(clamp(round(v[i] * inverse), -127.0f, 127.0f));
    }
}

/* Dequantize the four head-dim values a lane owns (values [4*lane, 4*lane+3]
 * of the head that starts at value index head_offset). Those four values lie
 * in a single 32-value block. */
inline float4 kipp_kv_q8_load(device const uchar *cache, ulong slot,
                              uint head_offset, uint lane) {
    uint value_index = head_offset + 4u * lane;
    uint block = value_index / KIPP_KV_Q8_BLOCK;
    uint local = value_index & (KIPP_KV_Q8_BLOCK - 1u);
    device const uchar *blk = cache + slot * KIPP_KV_Q8_POS_BYTES +
                              ulong(block) * KIPP_KV_Q8_BLOCK_BYTES;
    float scale = kipp_bf16_to_float(*(device const ushort *)blk);
    device const char *qs = (device const char *)(blk + 2);
    return scale * float4(float(qs[local]), float(qs[local + 1]),
                          float(qs[local + 2]), float(qs[local + 3]));
}

/* One thread per (32-value block, token). */
kernel void kipp_kv_write_q8_0(device const float *key [[buffer(0)]],
                               device const float *value [[buffer(1)]],
                               device uchar *key_cache [[buffer(2)]],
                               device uchar *value_cache [[buffer(3)]],
                               constant KvParams &params [[buffer(4)]],
                               device const uint *block_table [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint block = gid.x;
    uint token = gid.y;
    if (block >= KIPP_KV_VALUES_PER_TOKEN / KIPP_KV_Q8_BLOCK) {
        return;
    }
    uint position = params.start_position + token;
    ulong slot = kipp_kv_slot(params, block_table, position);
    ulong block_byte =
        slot * KIPP_KV_Q8_POS_BYTES + ulong(block) * KIPP_KV_Q8_BLOCK_BYTES;
    ulong state_base =
        ulong(token) * KIPP_KV_VALUES_PER_TOKEN + ulong(block) * KIPP_KV_Q8_BLOCK;
    kipp_kv_q8_store(key_cache + block_byte, key + state_base);
    kipp_kv_q8_store(value_cache + block_byte, value + state_base);
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
                           device float *partials [[buffer(6)]],
                           uint3 group_id [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial_values[KIPP_GQA_GROUPS][KIPP_HEAD_DIM];
    uint thread_id = group * 32u + lane;
    threadgroup float partial_maximum[KIPP_GQA_GROUPS];
    threadgroup float partial_denominator[KIPP_GQA_GROUPS];
    threadgroup uint partial_count[KIPP_GQA_GROUPS];

    uint head = group_id.x;
    uint split = group_id.y;
    uint token = group_id.z;
    uint kv_head = head / (KIPP_Q_HEADS / KIPP_KV_HEADS);
    device const float4 *query4 = (device const float4 *)(
        query + (ulong(token) * KIPP_Q_HEADS + head) * KIPP_HEAD_DIM);
    float4 q = query4[lane];
    const float scale = rsqrt(float(KIPP_HEAD_DIM));
    uint last_source = params.start_position + token;
    uint splits = kipp_ksplit_count(last_source, params.ksplit_cap);
    if (split >= splits) {
        return;
    }
    ulong head_offset = ulong(kv_head) * KIPP_HEAD_DIM;

    /* splits == 1 makes first/stride collapse to the legacy scan (source =
     * group, stride 8): identical arithmetic in identical order, keeping
     * short-context decode bit-for-bit unchanged. */
    uint first_source = split * KIPP_GQA_GROUPS + group;
    uint source_stride = splits * KIPP_GQA_GROUPS;
    float maximum = 0.0f;
    float denominator = 0.0f;
    float4 accumulator = 0.0f;
    uint count = 0;
    for (uint source = first_source; source <= last_source;
         source += source_stride) {
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
        if (splits == 1u) {
            output[(ulong(token) * KIPP_Q_HEADS + head) * KIPP_HEAD_DIM +
                   thread_id] = value / total;
        } else {
            /* Un-normalized numerator at this split's local maximum; the
             * reduce kernel merges the splits in fixed order. */
            ulong base = ((ulong(token) * KIPP_Q_HEADS + head) *
                              KIPP_KSPLIT_MAX +
                          split) *
                         KIPP_KSPLIT_STRIDE;
            partials[base + 4u + thread_id] = value;
            if (thread_id == 0u) {
                partials[base + 0u] = global_maximum;
                partials[base + 1u] = total;
            }
        }
    }
}

/*
 * Q8_0-KV variant of kipp_flash_gqa. Identical streaming/split-K structure
 * (so decode, speculative-verify, and the reduce kernel are unchanged); only
 * the KV load dequantizes each lane's four head-dim values from its 34-byte
 * block. Shares kipp_flash_gqa_reduce for the split merge.
 */
kernel void kipp_flash_gqa_q8_0(device const float *query [[buffer(0)]],
                                device const uchar *key_cache [[buffer(1)]],
                                device const uchar *value_cache [[buffer(2)]],
                                device float *output [[buffer(3)]],
                                constant KvParams &params [[buffer(4)]],
                                device const uint *block_table [[buffer(5)]],
                                device float *partials [[buffer(6)]],
                                uint3 group_id [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial_values[KIPP_GQA_GROUPS][KIPP_HEAD_DIM];
    uint thread_id = group * 32u + lane;
    threadgroup float partial_maximum[KIPP_GQA_GROUPS];
    threadgroup float partial_denominator[KIPP_GQA_GROUPS];
    threadgroup uint partial_count[KIPP_GQA_GROUPS];

    uint head = group_id.x;
    uint split = group_id.y;
    uint token = group_id.z;
    uint kv_head = head / (KIPP_Q_HEADS / KIPP_KV_HEADS);
    device const float4 *query4 = (device const float4 *)(
        query + (ulong(token) * KIPP_Q_HEADS + head) * KIPP_HEAD_DIM);
    float4 q = query4[lane];
    const float scale = rsqrt(float(KIPP_HEAD_DIM));
    uint last_source = params.start_position + token;
    uint splits = kipp_ksplit_count(last_source, params.ksplit_cap);
    if (split >= splits) {
        return;
    }
    uint head_offset = kv_head * KIPP_HEAD_DIM;

    uint first_source = split * KIPP_GQA_GROUPS + group;
    uint source_stride = splits * KIPP_GQA_GROUPS;
    float maximum = 0.0f;
    float denominator = 0.0f;
    float4 accumulator = 0.0f;
    uint count = 0;
    for (uint source = first_source; source <= last_source;
         source += source_stride) {
        ulong slot = kipp_kv_slot(params, block_table, source);
        float4 k = kipp_kv_q8_load(key_cache, slot, head_offset, lane);
        float score = simd_sum(dot(q, k)) * scale;
        float4 v = kipp_kv_q8_load(value_cache, slot, head_offset, lane);
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
        if (splits == 1u) {
            output[(ulong(token) * KIPP_Q_HEADS + head) * KIPP_HEAD_DIM +
                   thread_id] = value / total;
        } else {
            ulong base = ((ulong(token) * KIPP_Q_HEADS + head) *
                              KIPP_KSPLIT_MAX +
                          split) *
                         KIPP_KSPLIT_STRIDE;
            partials[base + 4u + thread_id] = value;
            if (thread_id == 0u) {
                partials[base + 0u] = global_maximum;
                partials[base + 1u] = total;
            }
        }
    }
}

/*
 * Merge kipp_flash_gqa's split-K partials for one (head, token) with the
 * standard streaming-softmax rescale, in fixed split order. Recomputes the
 * split count from the token's own position exactly as the split kernel
 * does; single-split tokens were written directly and are skipped.
 */
kernel void kipp_flash_gqa_reduce(device float *output [[buffer(0)]],
                                  device const float *partials [[buffer(1)]],
                                  constant KvParams &params [[buffer(2)]],
                                  uint2 group_id
                                  [[threadgroup_position_in_grid]],
                                  uint thread_id
                                  [[thread_index_in_threadgroup]]) {
    uint head = group_id.x;
    uint token = group_id.y;
    uint last_source = params.start_position + token;
    uint splits = kipp_ksplit_count(last_source, params.ksplit_cap);
    if (splits <= 1u || thread_id >= KIPP_HEAD_DIM) {
        return;
    }
    ulong base = (ulong(token) * KIPP_Q_HEADS + head) * KIPP_KSPLIT_MAX *
                 KIPP_KSPLIT_STRIDE;
    float global_maximum = KIPP_FLT_LOWEST;
    for (uint s = 0; s < splits; ++s) {
        ulong entry = base + ulong(s) * KIPP_KSPLIT_STRIDE;
        if (partials[entry + 1u] != 0.0f &&
            partials[entry] > global_maximum) {
            global_maximum = partials[entry];
        }
    }
    float value = 0.0f;
    float total = 0.0f;
    for (uint s = 0; s < splits; ++s) {
        ulong entry = base + ulong(s) * KIPP_KSPLIT_STRIDE;
        float denominator = partials[entry + 1u];
        if (denominator == 0.0f) {
            continue;
        }
        float weight = exp(partials[entry] - global_maximum);
        value += partials[entry + 4u + thread_id] * weight;
        total += denominator * weight;
    }
    output[(ulong(token) * KIPP_Q_HEADS + head) * KIPP_HEAD_DIM +
           thread_id] = value / total;
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
 * kernels below. */
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
 * Batched-prefill projections using simdgroup matrices. Each of the four
 * simdgroups in a threadgroup owns a 32-row x 16-token output tile and
 * walks the shared dimension in 8-wide steps: one transposed activation
 * fragment is reused against four weight fragments, so activation traffic
 * collapses compared with the vector kernel. Weights and staged activations
 * are BF16; accumulation is FP32.
 */
constant uint KIPP_MM_SIMDGROUPS = 4;
constant uint KIPP_MM_ROWS_PER_SIMDGROUP = 32;
constant uint KIPP_MM_ROW_FRAGMENTS = KIPP_MM_ROWS_PER_SIMDGROUP / 8;
constant uint KIPP_MM_TOKEN_FRAGMENTS = 4;
constant uint KIPP_MM_TOKEN_TILE = KIPP_MM_TOKEN_FRAGMENTS * 8;
/*
 * The quantized kernels keep a narrower 16-token slice per simdgroup than the
 * BF16 kernel's 32: doubling their accumulator fragments on top of the staged
 * dequantized weights collapses occupancy (measured at 2,048 tokens, a
 * 32-token slice takes Q8_0 prefill from 1139 to 135 tok/s while it *gains*
 * 7% for BF16). Tile width is therefore a property of each kernel, not of the
 * projection layer. The threadgroup still covers 64 tokens, because its four
 * simdgroups share one 32-row dequantized block instead of staging four
 * private ones: the staging allocation drops 16.9 KiB -> 4.2 KiB (so more
 * threadgroups fit a core) and each weight block is dequantized once per 64
 * tokens instead of once per 16.
 */
constant uint KIPP_MM_QUANT_TOKEN_FRAGMENTS = 2;
constant uint KIPP_MM_QUANT_SG_TOKEN_TILE = KIPP_MM_QUANT_TOKEN_FRAGMENTS * 8;
constant uint KIPP_MM_QUANT_TOKEN_TILE =
    KIPP_MM_SIMDGROUPS * KIPP_MM_QUANT_SG_TOKEN_TILE;
/* Geometry of the Metal 4 tensor projection kernel (defined further down,
 * inside the KIPP_ENABLE_TENSOR_OPS block); declared here so the geometry
 * probe can report it alongside the simdgroup tiles. */
#if defined(KIPP_ENABLE_TENSOR_OPS)
constant uint KIPP_MM_TENSOR_ROWS = 64;
constant uint KIPP_MM_TENSOR_TOKEN_TILE = 128;
constant uint KIPP_MM_TENSOR_K_CHUNK = 32;
#endif

/*
 * The host mirrors the tile geometry above to size matmul dispatch grids
 * (KIPP_METAL_MM_TOKEN_TILE and KIPP_METAL_MM_ROWS_PER_GROUP in
 * src/metal/kipp_metal.m). Drift between the two silently mis-shapes every
 * projection, so model open reads these values back from the compiled library
 * instead of trusting that both sides were edited together.
 */
kernel void kipp_mm_geometry(device uint *values [[buffer(0)]],
                             uint index [[thread_position_in_grid]]) {
    if (index != 0) {
        return;
    }
    values[0] = KIPP_MM_TOKEN_TILE;
    values[1] = KIPP_MM_ROWS_PER_SIMDGROUP;
    values[2] = KIPP_MM_SIMDGROUPS;
    values[3] = KIPP_MM_QUANT_TOKEN_TILE;
    /* The quantized kernels share one 32-row block across the whole
     * threadgroup, so their row grid is per-simdgroup-rows, not
     * rows-per-simdgroup times simdgroups. */
    values[4] = KIPP_MM_ROWS_PER_SIMDGROUP;
    /* Tensor-kernel tile; zero when this library was compiled without the
     * tensor path, which tells the host not to compare these slots. */
#if defined(KIPP_ENABLE_TENSOR_OPS)
    values[5] = KIPP_MM_TENSOR_TOKEN_TILE;
    values[6] = KIPP_MM_TENSOR_ROWS;
#else
    values[5] = 0;
    values[6] = 0;
#endif
}

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
    /* A host grid taller than the token count would wrap the subtraction
     * below and process a full phantom tile, writing past `output`. The host
     * derives its grid from KIPP_METAL_MM_TOKEN_TILE, which the geometry probe
     * pins to KIPP_MM_TOKEN_TILE at model open; this keeps a mismatch to
     * wasted threadgroups rather than memory corruption. */
    if (token_base >= params.token_count) {
        return;
    }
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

/*
 * Quantized batched-prefill projections. The threadgroup owns a 32-row x
 * 64-token output tile: its four simdgroups cooperatively dequantize one
 * shared 32-row quantization block into a threadgroup float tile (eight
 * values per thread), then each sweeps the block against its own 16-token
 * slice with 8-wide simdgroup matrix fragments. Weight bytes are read and
 * dequantized exactly once per 64 tokens, and the shared block keeps the
 * staging allocation at 4.2 KiB so several threadgroups stay resident per
 * core. Activations are read directly from the FP32 buffer (no BF16
 * staging), so the dequantized math matches the vector kernels and the CPU
 * oracle up to reduction order.
 *
 * The block loop carries threadgroup barriers, so a simdgroup whose token
 * slice falls past the token count keeps dequantizing and barriering with
 * token_blocks == 0 instead of returning early.
 *
 * Both kernels assume params.rows % 32 == 0 and params.columns % 32 == 0:
 * every quantized projection in the registry satisfies both.
 */
constant uint KIPP_MM_BLOCK = 32;
/* +1 float of padding per staged row so the 32 lanes' dequant writes land in
 * different threadgroup-memory banks. */
constant uint KIPP_MM_STAGED_STRIDE = KIPP_MM_BLOCK + 1;
constant uint KIPP_MM_Q8_BLOCK_BYTES = 34;
constant uint KIPP_MM_A4_GROUP_BYTES = 20;

kernel void kipp_matmul_q8_0(device const uchar *weight [[buffer(0)]],
                             device const float *input [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant MatvecParams &params [[buffer(3)]],
                             uint2 group_id [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float staged[KIPP_MM_SIMDGROUPS][64];
    threadgroup float staged_weights[KIPP_MM_ROWS_PER_SIMDGROUP]
                                    [KIPP_MM_STAGED_STRIDE];
    uint row_base = group_id.x * KIPP_MM_ROWS_PER_SIMDGROUP;
    /* Threadgroup-uniform, so it cannot strand the barriers below. */
    if (row_base >= params.rows) {
        return;
    }
    uint tile_base = group_id.y * KIPP_MM_QUANT_TOKEN_TILE;
    /* See kipp_matmul_bf16: guards the unsigned wrap on an over-tall grid.
     * Also threadgroup-uniform. */
    if (tile_base >= params.token_count) {
        return;
    }
    /* Each simdgroup owns a 16-token slice of the 64-token tile; a slice
     * past the token count runs the loop for dequant and barriers only. */
    uint token_base = tile_base + group * KIPP_MM_QUANT_SG_TOKEN_TILE;
    uint tile = token_base < params.token_count
                    ? min(KIPP_MM_QUANT_SG_TOKEN_TILE,
                          params.token_count - token_base)
                    : 0u;
    uint blocks = params.columns / KIPP_MM_BLOCK;

    simdgroup_float8x8 accumulators[KIPP_MM_ROW_FRAGMENTS]
                                   [KIPP_MM_QUANT_TOKEN_FRAGMENTS];
    for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS; ++row_block) {
        for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
            accumulators[row_block][tf] = simdgroup_float8x8(0.0f);
        }
    }
    uint token_blocks = (tile + 7u) / 8u;
    /* Thread t dequantizes values 8*(t%4) .. 8*(t%4)+7 of row t/4's block;
     * the addresses land in 32 distinct threadgroup-memory banks. */
    uint dq_row = (group * 32u + lane) / 4u;
    uint dq_chunk = ((group * 32u + lane) % 4u) * 8u;
    for (uint block = 0; block < blocks; ++block) {
        device const uchar *blk = weight +
            (ulong(row_base + dq_row) * blocks + block) *
                ulong(KIPP_MM_Q8_BLOCK_BYTES);
        float d = kipp_fp16_bytes(blk);
        device const char *qs = (device const char *)(blk + 2);
        for (uint j = 0; j < 8u; ++j) {
            staged_weights[dq_row][dq_chunk + j] =
                d * float(qs[dq_chunk + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint column = block * KIPP_MM_BLOCK;
        for (uint sub = 0; sub < KIPP_MM_BLOCK / 8u; ++sub) {
            simdgroup_float8x8 activation[KIPP_MM_QUANT_TOKEN_FRAGMENTS];
            for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
                if (tf < token_blocks) {
                    simdgroup_load(activation[tf],
                                   input + (ulong)(token_base + tf * 8u) *
                                               params.columns +
                                       column + sub * 8u,
                                   params.columns, 0, true);
                }
            }
            for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS;
                 ++row_block) {
                simdgroup_float8x8 weights;
                simdgroup_load(
                    weights,
                    &staged_weights[row_block * 8u][sub * 8u],
                    KIPP_MM_STAGED_STRIDE);
                for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
                    if (tf < token_blocks) {
                        simdgroup_multiply_accumulate(
                            accumulators[row_block][tf], weights,
                            activation[tf], accumulators[row_block][tf]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS; ++row_block) {
        for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
            uint sub_base = token_base + tf * 8u;
            if (sub_base >= params.token_count) {
                continue;
            }
            uint sub_tile = min(8u, params.token_count - sub_base);
            if (sub_tile == 8u) {
                simdgroup_store(accumulators[row_block][tf],
                                output + (ulong)sub_base * params.rows +
                                    row_base + row_block * 8u,
                                params.rows, 0, true);
            } else {
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

kernel void kipp_matmul_affine4(device const uchar *weight [[buffer(0)]],
                                device const float *input [[buffer(1)]],
                                device float *output [[buffer(2)]],
                                constant MatvecParams &params [[buffer(3)]],
                                uint2 group_id
                                    [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float staged[KIPP_MM_SIMDGROUPS][64];
    threadgroup float staged_weights[KIPP_MM_ROWS_PER_SIMDGROUP]
                                    [KIPP_MM_STAGED_STRIDE];
    uint row_base = group_id.x * KIPP_MM_ROWS_PER_SIMDGROUP;
    /* Threadgroup-uniform, so it cannot strand the barriers below. */
    if (row_base >= params.rows) {
        return;
    }
    uint tile_base = group_id.y * KIPP_MM_QUANT_TOKEN_TILE;
    /* See kipp_matmul_bf16: guards the unsigned wrap on an over-tall grid.
     * Also threadgroup-uniform. */
    if (tile_base >= params.token_count) {
        return;
    }
    /* Each simdgroup owns a 16-token slice of the 64-token tile; a slice
     * past the token count runs the loop for dequant and barriers only. */
    uint token_base = tile_base + group * KIPP_MM_QUANT_SG_TOKEN_TILE;
    uint tile = token_base < params.token_count
                    ? min(KIPP_MM_QUANT_SG_TOKEN_TILE,
                          params.token_count - token_base)
                    : 0u;
    uint groups = params.columns / KIPP_MM_BLOCK;

    simdgroup_float8x8 accumulators[KIPP_MM_ROW_FRAGMENTS]
                                   [KIPP_MM_QUANT_TOKEN_FRAGMENTS];
    for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS; ++row_block) {
        for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
            accumulators[row_block][tf] = simdgroup_float8x8(0.0f);
        }
    }
    uint token_blocks = (tile + 7u) / 8u;
    /* Thread t dequantizes nibble bytes 4*(t%4) .. 4*(t%4)+3 of row t/4's
     * group (values 8*(t%4) .. 8*(t%4)+7): w = scale*q + bias. */
    uint dq_row = (group * 32u + lane) / 4u;
    uint dq_byte = ((group * 32u + lane) % 4u) * 4u;
    for (uint quant_group = 0; quant_group < groups; ++quant_group) {
        device const uchar *grp = weight +
            (ulong(row_base + dq_row) * groups + quant_group) *
                ulong(KIPP_MM_A4_GROUP_BYTES);
        float scale = kipp_fp16_bytes(grp + 16);
        float bias = kipp_fp16_bytes(grp + 18);
        for (uint k = 0; k < 4u; ++k) {
            uchar packed = grp[dq_byte + k];
            staged_weights[dq_row][2u * (dq_byte + k)] =
                scale * float(packed & 0x0fu) + bias;
            staged_weights[dq_row][2u * (dq_byte + k) + 1u] =
                scale * float(packed >> 4) + bias;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint column = quant_group * KIPP_MM_BLOCK;
        for (uint sub = 0; sub < KIPP_MM_BLOCK / 8u; ++sub) {
            simdgroup_float8x8 activation[KIPP_MM_QUANT_TOKEN_FRAGMENTS];
            for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
                if (tf < token_blocks) {
                    simdgroup_load(activation[tf],
                                   input + (ulong)(token_base + tf * 8u) *
                                               params.columns +
                                       column + sub * 8u,
                                   params.columns, 0, true);
                }
            }
            for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS;
                 ++row_block) {
                simdgroup_float8x8 weights;
                simdgroup_load(
                    weights,
                    &staged_weights[row_block * 8u][sub * 8u],
                    KIPP_MM_STAGED_STRIDE);
                for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
                    if (tf < token_blocks) {
                        simdgroup_multiply_accumulate(
                            accumulators[row_block][tf], weights,
                            activation[tf], accumulators[row_block][tf]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint row_block = 0; row_block < KIPP_MM_ROW_FRAGMENTS; ++row_block) {
        for (uint tf = 0; tf < KIPP_MM_QUANT_TOKEN_FRAGMENTS; ++tf) {
            uint sub_base = token_base + tf * 8u;
            if (sub_base >= params.token_count) {
                continue;
            }
            uint sub_tile = min(8u, params.token_count - sub_base);
            if (sub_tile == 8u) {
                simdgroup_store(accumulators[row_block][tf],
                                output + (ulong)sub_base * params.rows +
                                    row_base + row_block * 8u,
                                params.rows, 0, true);
            } else {
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

/*
 * Batched-prefill attention. One 32-thread threadgroup (a single simdgroup)
 * owns an 8-query tile for one query head and streams the KV cache in
 * 32-position tiles that coincide with KV blocks, so each tile costs one
 * block-table lookup and then reads a physically contiguous range. Q K^T and
 * P V run on the simdgroup matrix units with an online softmax between them
 * (running per-query maximum and denominator, rescaled per tile); the scalar
 * softmax step round-trips the 8x32 score tile through threadgroup memory,
 * which with a single simdgroup needs only simdgroup_barrier. K and V
 * fragments load directly from the BF16 cache; only the 8x128 query tile is
 * staged (once, as bfloat). Each query head re-reads its KV head's tiles;
 * sharing a tile across the 4 query heads of a KV head is future work.
 *
 * Decode and speculative-verify rounds keep kipp_flash_gqa: its per-token
 * reduction order is the token-identity contract for speculation, and at one
 * token the split-K kernel keeps more of the GPU busy.
 */
constant uint KIPP_FA_QUERIES = 8;
constant uint KIPP_FA_KV_TILE = 32;
constant uint KIPP_FA_SCORE_STRIDE = KIPP_FA_KV_TILE + 8;

kernel void
kipp_flash_gqa_prefill(device const float *query [[buffer(0)]],
                       device const ushort *key_cache [[buffer(1)]],
                       device const ushort *value_cache [[buffer(2)]],
                       device float *output [[buffer(3)]],
                       constant KvParams &params [[buffer(4)]],
                       device const uint *block_table [[buffer(5)]],
                       uint2 group_id [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat q_tile[KIPP_FA_QUERIES * KIPP_HEAD_DIM];
    threadgroup float scores[KIPP_FA_QUERIES * KIPP_FA_SCORE_STRIDE];
    threadgroup bfloat probs[KIPP_FA_QUERIES * KIPP_FA_SCORE_STRIDE];
    threadgroup float rescale[8 * 8];
    threadgroup float staged_out[KIPP_FA_QUERIES * KIPP_HEAD_DIM];
    threadgroup float maxima[KIPP_FA_QUERIES];
    threadgroup float denominators[KIPP_FA_QUERIES];

    uint head = group_id.x;
    uint kv_head = head / (KIPP_Q_HEADS / KIPP_KV_HEADS);
    uint tile_first = group_id.y * KIPP_FA_QUERIES;
    /* As in the matmul kernels: an over-tall grid would wrap the subtraction
     * and stage a phantom query tile. No cross-simdgroup barrier runs before
     * this point, so returning early is safe. */
    if (tile_first >= params.token_count) {
        return;
    }
    uint tile_rows = min(KIPP_FA_QUERIES, params.token_count - tile_first);
    const float scale = rsqrt(float(KIPP_HEAD_DIM));
    ulong head_offset = ulong(kv_head) * KIPP_HEAD_DIM;
    ulong query_stride = ulong(KIPP_Q_HEADS) * KIPP_HEAD_DIM;

    /* Stage the query tile as bfloat once; ragged rows duplicate the last
     * real query (their outputs are never stored). Zero the off-diagonal of
     * the 8x8 rescale matrix once; the diagonal is rewritten per KV tile. */
    for (uint index = lane; index < KIPP_FA_QUERIES * KIPP_HEAD_DIM;
         index += 32u) {
        uint row = index / KIPP_HEAD_DIM;
        uint clamped = min(row, tile_rows - 1u);
        q_tile[index] = bfloat(
            query[(ulong(tile_first + clamped)) * query_stride +
                  head * KIPP_HEAD_DIM + index % KIPP_HEAD_DIM]);
    }
    for (uint index = lane; index < 64u; index += 32u) {
        rescale[index] = 0.0f;
    }
    if (lane < KIPP_FA_QUERIES) {
        maxima[lane] = KIPP_FLT_LOWEST;
        denominators[lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 out_acc[KIPP_HEAD_DIM / 8];
    /* `fragment` is a reserved MSL keyword; using it as an identifier makes
     * the whole MMA library fail to compile at runtime and silently drops
     * every matrix kernel to the vector path (found 2026-07-22). */
    for (uint frag = 0; frag < KIPP_HEAD_DIM / 8; ++frag) {
        out_acc[frag] = simdgroup_float8x8(0.0f);
    }

    /* The final tile can cover positions past the session's high-water
     * mark. Those columns are masked out of the softmax, and their values
     * are finite by construction: MTLBuffer allocations are documented
     * zero-filled, and every later write stores a finite bf16, so the
     * masked 0-weight x value products stay zero instead of poisoning the
     * accumulator with NaN. */
    uint last_query = params.start_position + tile_first + tile_rows - 1u;
    uint kv_tiles = last_query / KIPP_FA_KV_TILE + 1u;
    for (uint tile = 0; tile < kv_tiles; ++tile) {
        uint tile_base = tile * KIPP_FA_KV_TILE;
        /* One table lookup covers the whole tile: tile_base is
         * block-aligned, so the 32 positions are physically contiguous. */
        ulong kv_base =
            kipp_kv_slot(params, block_table, tile_base) *
            KIPP_KV_VALUES_PER_TOKEN;
        device const bfloat *keys =
            (device const bfloat *)key_cache + kv_base + head_offset;
        device const bfloat *values =
            (device const bfloat *)value_cache + kv_base + head_offset;

        /* Scores = Q K^T over the tile, on the matrix units. */
        simdgroup_float8x8 score_frag[KIPP_FA_KV_TILE / 8];
        for (uint column = 0; column < KIPP_FA_KV_TILE / 8; ++column) {
            score_frag[column] = simdgroup_float8x8(0.0f);
        }
        for (uint depth = 0; depth < KIPP_HEAD_DIM / 8; ++depth) {
            simdgroup_bfloat8x8 q_frag;
            simdgroup_load(q_frag, &q_tile[depth * 8u], KIPP_HEAD_DIM);
            for (uint column = 0; column < KIPP_FA_KV_TILE / 8; ++column) {
                simdgroup_bfloat8x8 k_frag;
                /* Transposed load turns 8 cached positions x 8 dims into
                 * the (dim, position) fragment Q K^T needs. */
                simdgroup_load(k_frag,
                               keys + (ulong)(column * 8u) *
                                          KIPP_KV_VALUES_PER_TOKEN +
                                   depth * 8u,
                               KIPP_KV_VALUES_PER_TOKEN, 0, true);
                simdgroup_multiply_accumulate(score_frag[column], q_frag,
                                              k_frag, score_frag[column]);
            }
        }
        for (uint column = 0; column < KIPP_FA_KV_TILE / 8; ++column) {
            simdgroup_store(score_frag[column],
                            &scores[column * 8u], KIPP_FA_SCORE_STRIDE);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        /* Online-softmax step on all 32 lanes: quad r (lanes 4r..4r+3)
         * owns query row r, each lane owns eight score columns, and quad
         * shuffles reduce the row maximum and exp-sum, so no lane idles
         * and the serial exp work per lane drops from 32 columns to 8. */
        {
            uint row = lane / 4u;
            uint sub = lane % 4u;
            uint query_position =
                params.start_position + tile_first + min(row, tile_rows - 1u);
            float row_max = KIPP_FLT_LOWEST;
            for (uint index = 0; index < KIPP_FA_KV_TILE / 4u; ++index) {
                uint column = sub * (KIPP_FA_KV_TILE / 4u) + index;
                if (tile_base + column <= query_position) {
                    row_max = max(row_max,
                                  scores[row * KIPP_FA_SCORE_STRIDE + column] *
                                      scale);
                }
            }
            /* row*4 is quad-aligned, so masks 1 and 2 stay in the quad. */
            row_max = max(row_max, simd_shuffle_xor(row_max, 1u));
            row_max = max(row_max, simd_shuffle_xor(row_max, 2u));
            float previous = maxima[row];
            float merged = max(previous, row_max);
            float correction =
                merged == previous ? 1.0f : exp(previous - merged);
            float sum = 0.0f;
            for (uint index = 0; index < KIPP_FA_KV_TILE / 4u; ++index) {
                uint column = sub * (KIPP_FA_KV_TILE / 4u) + index;
                float weight = 0.0f;
                if (tile_base + column <= query_position) {
                    weight = exp(scores[row * KIPP_FA_SCORE_STRIDE + column] *
                                     scale -
                                 merged);
                }
                probs[row * KIPP_FA_SCORE_STRIDE + column] = bfloat(weight);
                sum += weight;
            }
            sum += simd_shuffle_xor(sum, 1u);
            sum += simd_shuffle_xor(sum, 2u);
            if (sub == 0u) {
                maxima[row] = merged;
                denominators[row] = denominators[row] * correction + sum;
                rescale[row * 8u + row] = correction;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        /* Rescale the running output by diag(correction), then accumulate
         * P V for this tile. */
        simdgroup_float8x8 diag;
        simdgroup_load(diag, rescale, 8);
        simdgroup_bfloat8x8 p_frag[KIPP_FA_KV_TILE / 8];
        for (uint column = 0; column < KIPP_FA_KV_TILE / 8; ++column) {
            simdgroup_load(p_frag[column], &probs[column * 8u],
                           KIPP_FA_SCORE_STRIDE);
        }
        for (uint frag = 0; frag < KIPP_HEAD_DIM / 8; ++frag) {
            simdgroup_multiply(out_acc[frag], diag, out_acc[frag]);
            for (uint column = 0; column < KIPP_FA_KV_TILE / 8; ++column) {
                simdgroup_bfloat8x8 v_frag;
                simdgroup_load(v_frag,
                               values + (ulong)(column * 8u) *
                                            KIPP_KV_VALUES_PER_TOKEN +
                                   frag * 8u,
                               KIPP_KV_VALUES_PER_TOKEN);
                simdgroup_multiply_accumulate(out_acc[frag],
                                              p_frag[column], v_frag,
                                              out_acc[frag]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Stage the tile's output and write the valid rows, dividing by each
     * query's softmax denominator on the way out. */
    for (uint frag = 0; frag < KIPP_HEAD_DIM / 8; ++frag) {
        simdgroup_store(out_acc[frag], &staged_out[frag * 8u],
                        KIPP_HEAD_DIM);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = lane; index < tile_rows * KIPP_HEAD_DIM; index += 32u) {
        uint row = index / KIPP_HEAD_DIM;
        output[(ulong(tile_first + row)) * query_stride +
               head * KIPP_HEAD_DIM + index % KIPP_HEAD_DIM] =
            staged_out[index] / denominators[row];
    }
}
#endif

/*
 * Metal 4 tensor-ops path (M5-class neural accelerators). Compiled only when
 * the host's device gate passes (Metal 4 family, device allow-list, env
 * switches) AND this block compiles and builds pipelines on the running
 * toolchain — the includes below do not exist before Metal 4, so they live
 * inside the guard. The host treats any failure here as "tensor path
 * unavailable" and falls back to the simdgroup-matrix kernels above.
 */
#if defined(KIPP_ENABLE_TENSOR_OPS)
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

/*
 * Smallest kernel that forces the toolchain to instantiate matmul2d, a
 * cooperative destination tensor, and its store. Building this pipeline is
 * the host's runtime probe: a device can advertise the Metal 4 family and
 * still fail here (llama.cpp probes the same way), and a probe failure must
 * demote the whole tensor path rather than surface later as a pipeline error
 * on the real kernel.
 */
/*
 * Batched-prefill projection on the Metal 4 tensor pipeline. Weights,
 * activations, and output are all device tensors whose extents carry the true
 * matrix dimensions, so ragged row/token edges are clipped by the tensor API
 * on both reads and stores -- there are no manual tail guards. No operand is
 * staged: on this hardware class the accelerators stream both operands faster
 * than any threadgroup-staging scheme (measured 4.9-6.2x the simdgroup
 * kernels at a 512-token round, and ~2x even at 8 tokens; the llama.cpp-style
 * staged variant reached only 0.96-2.9x). The kernel assumes
 * params.columns % 32 == 0, which every registry projection satisfies.
 *
 * The threadgroup covers a 64-row x 128-token output tile with four
 * cooperating simdgroups, and the dispatch grid is TRANSPOSED relative to
 * the simdgroup kernels: x advances tokens, y advances rows. The host-side
 * mirror constants are KIPP_METAL_MM_TENSOR_* and the geometry probe reads
 * these values back at model open.
 */
kernel void kipp_matmul_bf16_tensor(
        device bfloat *weight [[buffer(0)]],
        device float *input [[buffer(1)]],
        device float *output [[buffer(2)]],
        constant MatvecParams &params [[buffer(3)]],
        uint2 group_id [[threadgroup_position_in_grid]]) {
    const int columns = (int)params.columns;
    const int rows = (int)params.rows;
    const int tokens = (int)params.token_count;

    const int row_base = (int)(group_id.y * KIPP_MM_TENSOR_ROWS);
    const int token_base = (int)(group_id.x * KIPP_MM_TENSOR_TOKEN_TILE);

    /* Weights are [row][column], activations [token][column], output
     * [token][row] -- exactly the layouts the engine already keeps, wrapped
     * with K-contiguous strides. Activations are consumed as FP32 straight
     * from the round's working buffer: matmul2d takes float sources, which
     * makes the whole kipp_bf16_stage pass unnecessary on this path and
     * removes an activation rounding step (measured ~12-18% slower per GEMM
     * than bf16 activations, but the staging it deletes costs more). */
    auto tA = tensor(weight, dextents<int32_t, 2>(columns, rows),
                     array<int, 2>({1, columns}));
    auto tB = tensor(input, dextents<int32_t, 2>(columns, tokens),
                     array<int, 2>({1, columns}));
    auto tD = tensor(output, dextents<int32_t, 2>(rows, tokens),
                     array<int, 2>({1, rows}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            (int)KIPP_MM_TENSOR_TOKEN_TILE, (int)KIPP_MM_TENSOR_ROWS,
            (int)KIPP_MM_TENSOR_K_CHUNK, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto accumulator =
        mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA),
                                              float>();
    for (int chunk = 0; chunk < columns; chunk += (int)KIPP_MM_TENSOR_K_CHUNK) {
        auto weightSlice = tA.slice(chunk, row_base);
        auto activationSlice = tB.slice(chunk, token_base);
        mm.run(activationSlice, weightSlice, accumulator);
    }
    auto outputSlice = tD.slice(row_base, token_base);
    accumulator.store(outputSlice);
}

kernel void kipp_tensor_probe_bf16(device bfloat *a [[buffer(0)]],
                                   device bfloat *b [[buffer(1)]],
                                   device float *c [[buffer(2)]]) {
    auto tA = tensor(a, dextents<int32_t, 2>(32, 16), array<int, 2>({1, 32}));
    auto tB = tensor(b, dextents<int32_t, 2>(32, 16), array<int, 2>({1, 32}));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            16, 16, 32, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA),
                                                    float>();
    auto mB = tB.slice(0, 0);
    auto mA = tA.slice(0, 0);
    mm.run(mB, mA, cT);
    auto tC = tensor(c, dextents<int32_t, 2>(16, 16), array<int, 2>({1, 16}));
    cT.store(tC.slice(0, 0));
}
#endif
