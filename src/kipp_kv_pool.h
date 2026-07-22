/*
 * Paged KV block pool (core-owned, GPU-free).
 *
 * A pool of fixed-size logical KV blocks with content-addressed reuse, the
 * bookkeeping half of Kipp's paged attention. It hands out *logical* block
 * ids only; a backend maps each id to device memory separately. Blocks are
 * 32 tokens; a full block is hashed (chained on its parent block's hash) so
 * that sequences sharing a prefix reuse the same blocks, and a reference
 * count with an intrusive LRU free list lets released blocks be either
 * revived on a later cache hit or reclaimed under pressure.
 *
 * Design follows nano-vllm's block manager with vLLM-v1 refinements:
 * verify-by-tokens on every hit (collision-proof), keep freed blocks in the
 * index for O(1) revival, and cap a prefix match one token short so a
 * sequence always has a token left to produce logits. This module is pure
 * bookkeeping and is unit-tested without any GPU backend. It backs
 * cross-request KV prefix sharing in production: pooled models
 * (kipp_model_open_pooled, the server default on CPU and Metal) publish a
 * finished session's full blocks here at reset/destroy and later sessions
 * adopt matching prefixes via kipp_session_match_prefix.
 */
#ifndef KIPP_KV_POOL_H
#define KIPP_KV_POOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define KIPP_KV_BLOCK_TOKENS 32u
#define KIPP_KV_INVALID_BLOCK UINT32_MAX

typedef struct kipp_kv_pool kipp_kv_pool;

/* Create a pool of `block_count` blocks (must be > 0). NULL on failure. */
kipp_kv_pool *kipp_kv_pool_create(uint32_t block_count);
void kipp_kv_pool_destroy(kipp_kv_pool *pool);

/* Number of blocks currently free (ref_count == 0). */
uint32_t kipp_kv_pool_free_count(const kipp_kv_pool *pool);

/*
 * Deterministic content hash for a block, chained onto its parent block's
 * hash (`parent_hash` is 0 for a sequence's first block). Callers use it to
 * drive prefix lookups; kipp_kv_pool_acquire computes it internally too.
 */
uint64_t kipp_kv_pool_hash(uint64_t parent_hash, const uint32_t *tokens,
                           uint32_t count);

/*
 * Acquire a block holding `tokens[0..count)` (count in [1, 32]) chained after
 * `parent_hash`. A full block (count == 32) is content-addressed: if an
 * identical one already exists (referenced, or revivable from the free list)
 * it is reused with its ref count incremented and `*reused` set true.
 * Otherwise a free (or LRU-evictable) block is claimed and filled with
 * `*reused` false. Partial blocks (count < 32) are always private and never
 * shared. Returns the block id, or KIPP_KV_INVALID_BLOCK when exhausted.
 * `reused` may be NULL.
 */
uint32_t kipp_kv_pool_acquire(kipp_kv_pool *pool, uint64_t parent_hash,
                              const uint32_t *tokens, uint32_t count,
                              bool *reused);

/* Releasing to 0 makes the block LRU-evictable while keeping its hash so a
 * later matching acquire can revive it. */
void kipp_kv_pool_release(kipp_kv_pool *pool, uint32_t block_id);

/*
 * Longest cached prefix of `tokens[0..token_count)`: walks full 32-token
 * blocks by chained hash, and for each hit appends its (ref-incremented)
 * block id to `out_blocks` (capacity `max_blocks`). Stops at the first miss
 * and caps the match at token_count - 1 tokens so at least one token is left
 * to run through the model. Returns the number of matched blocks; when
 * `matched_tokens` is non-NULL it receives the covered token count
 * (matched_blocks * 32). Matched blocks are owned by the caller (release
 * them). The chained parent hash of the matched prefix is returned via
 * `out_parent_hash` when non-NULL (0 when nothing matched).
 */
uint32_t kipp_kv_pool_prefix_match(kipp_kv_pool *pool, const uint32_t *tokens,
                                   uint32_t token_count, uint32_t *out_blocks,
                                   uint32_t max_blocks, uint32_t *matched_tokens,
                                   uint64_t *out_parent_hash);

/*
 * Claim the LRU free block as a private, unindexed block (ref count 1) with
 * no recorded tokens, evicting stale indexed content if needed. This is
 * acquire without the content lookup: the caller writes KV into the block
 * incrementally and may later publish it with kipp_kv_pool_seal. Returns
 * KIPP_KV_INVALID_BLOCK when the pool is exhausted.
 */
uint32_t kipp_kv_pool_alloc(kipp_kv_pool *pool);

/*
 * Publish a full private block (ref count >= 1, not yet indexed): record its
 * 32 tokens, compute the chained hash, and insert it into the content index
 * so later prefix matches can share it. If an identical block is already
 * indexed the block stays private (the index keeps one copy per content).
 * Returns 0 on success, -1 on invalid arguments.
 */
int kipp_kv_pool_seal(kipp_kv_pool *pool, uint32_t block_id,
                      uint64_t parent_hash, const uint32_t *tokens);

typedef struct {
    uint32_t total_blocks;
    uint32_t free_blocks;         /* ref_count == 0 (including revivable) */
    uint64_t reused_blocks_total; /* content hits: acquire + prefix match */
    uint64_t evicted_blocks_total;/* indexed content reclaimed for new use */
} kipp_kv_pool_stats;

void kipp_kv_pool_get_stats(const kipp_kv_pool *pool,
                            kipp_kv_pool_stats *out_stats);

#ifdef KIPP_TESTING
/* Force every internal content hash to a constant so the collision tests can
 * prove that hash-equal, token-unequal blocks never alias (the memcmp
 * verify). Affects this pool's internal hashing only. */
void kipp_kv_pool_test_force_hash(kipp_kv_pool *pool, bool force);
#endif

#endif /* KIPP_KV_POOL_H */
