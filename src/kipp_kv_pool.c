#include "kipp_kv_pool.h"

#include <stdlib.h>
#include <string.h>

/*
 * One logical block. A block is in exactly one of two states: referenced
 * (ref_count > 0, off the free list) or evictable (ref_count == 0, on the
 * intrusive free list). `in_index` tracks whether it is registered in the
 * content-hash table; a full block stays indexed across a release so it can
 * be revived, and is only removed when actually reclaimed for new content.
 */
typedef struct {
    uint32_t ref_count;
    uint32_t token_count;
    uint64_t hash;
    bool in_index;
    uint32_t tokens[KIPP_KV_BLOCK_TOKENS];
    uint32_t hash_next; /* chain within a hash bucket */
    uint32_t prev_free; /* intrusive LRU free list */
    uint32_t next_free;
} kv_block;

struct kipp_kv_pool {
    uint32_t block_count;
    uint32_t bucket_count;
    uint32_t free_head; /* least-recently-used: evicted first */
    uint32_t free_tail; /* most-recently-used: evicted last */
    uint32_t free_blocks;
    uint64_t reused_total;
    uint64_t evicted_total;
    bool force_hash; /* KIPP_TESTING collision hook */
    kv_block *blocks;
    uint32_t *buckets;
};

/* Internal hash: the public function unless the collision hook is armed. */
static uint64_t pool_hash(const kipp_kv_pool *pool, uint64_t parent_hash,
                          const uint32_t *tokens, uint32_t count) {
    if (pool->force_hash) {
        return UINT64_C(0x1234);
    }
    return kipp_kv_pool_hash(parent_hash, tokens, count);
}

/* FNV-1a over the parent hash then the block's tokens: deterministic and
 * order-sensitive so the chain distinguishes prefixes. */
uint64_t kipp_kv_pool_hash(uint64_t parent_hash, const uint32_t *tokens,
                           uint32_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const uint64_t prime = UINT64_C(1099511628211);
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (parent_hash >> shift) & 0xFFu;
        hash *= prime;
    }
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t token = tokens[index];
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (token >> shift) & 0xFFu;
            hash *= prime;
        }
    }
    return hash;
}

kipp_kv_pool *kipp_kv_pool_create(uint32_t block_count) {
    if (block_count == 0) {
        return NULL;
    }
    kipp_kv_pool *pool = calloc(1, sizeof(*pool));
    if (pool == NULL) {
        return NULL;
    }
    uint32_t buckets = 16;
    while (buckets < block_count * 2u) {
        buckets *= 2u;
    }
    pool->block_count = block_count;
    pool->bucket_count = buckets;
    pool->blocks = calloc(block_count, sizeof(*pool->blocks));
    pool->buckets = malloc((size_t)buckets * sizeof(*pool->buckets));
    if (pool->blocks == NULL || pool->buckets == NULL) {
        kipp_kv_pool_destroy(pool);
        return NULL;
    }
    for (uint32_t index = 0; index < buckets; ++index) {
        pool->buckets[index] = KIPP_KV_INVALID_BLOCK;
    }
    /* Seed the free list with every block, id order (0 is LRU). */
    for (uint32_t index = 0; index < block_count; ++index) {
        pool->blocks[index].hash_next = KIPP_KV_INVALID_BLOCK;
        pool->blocks[index].prev_free =
            index == 0 ? KIPP_KV_INVALID_BLOCK : index - 1;
        pool->blocks[index].next_free =
            index + 1 == block_count ? KIPP_KV_INVALID_BLOCK : index + 1;
    }
    pool->free_head = 0;
    pool->free_tail = block_count - 1;
    pool->free_blocks = block_count;
    return pool;
}

void kipp_kv_pool_destroy(kipp_kv_pool *pool) {
    if (pool == NULL) {
        return;
    }
    free(pool->blocks);
    free(pool->buckets);
    free(pool);
}

uint32_t kipp_kv_pool_free_count(const kipp_kv_pool *pool) {
    return pool != NULL ? pool->free_blocks : 0;
}

/* ------------------------------------------------------- free list + index */

static void free_list_remove(kipp_kv_pool *pool, uint32_t id) {
    kv_block *block = &pool->blocks[id];
    if (block->prev_free != KIPP_KV_INVALID_BLOCK) {
        pool->blocks[block->prev_free].next_free = block->next_free;
    } else {
        pool->free_head = block->next_free;
    }
    if (block->next_free != KIPP_KV_INVALID_BLOCK) {
        pool->blocks[block->next_free].prev_free = block->prev_free;
    } else {
        pool->free_tail = block->prev_free;
    }
    block->prev_free = KIPP_KV_INVALID_BLOCK;
    block->next_free = KIPP_KV_INVALID_BLOCK;
    --pool->free_blocks;
}

static void free_list_push_tail(kipp_kv_pool *pool, uint32_t id) {
    kv_block *block = &pool->blocks[id];
    block->next_free = KIPP_KV_INVALID_BLOCK;
    block->prev_free = pool->free_tail;
    if (pool->free_tail != KIPP_KV_INVALID_BLOCK) {
        pool->blocks[pool->free_tail].next_free = id;
    } else {
        pool->free_head = id;
    }
    pool->free_tail = id;
    ++pool->free_blocks;
}

static void index_insert(kipp_kv_pool *pool, uint32_t id) {
    uint32_t bucket = (uint32_t)(pool->blocks[id].hash % pool->bucket_count);
    pool->blocks[id].hash_next = pool->buckets[bucket];
    pool->buckets[bucket] = id;
    pool->blocks[id].in_index = true;
}

static void index_remove(kipp_kv_pool *pool, uint32_t id) {
    uint32_t bucket = (uint32_t)(pool->blocks[id].hash % pool->bucket_count);
    uint32_t current = pool->buckets[bucket];
    uint32_t previous = KIPP_KV_INVALID_BLOCK;
    while (current != KIPP_KV_INVALID_BLOCK && current != id) {
        previous = current;
        current = pool->blocks[current].hash_next;
    }
    if (current != id) {
        return;
    }
    if (previous != KIPP_KV_INVALID_BLOCK) {
        pool->blocks[previous].hash_next = pool->blocks[id].hash_next;
    } else {
        pool->buckets[bucket] = pool->blocks[id].hash_next;
    }
    pool->blocks[id].hash_next = KIPP_KV_INVALID_BLOCK;
    pool->blocks[id].in_index = false;
}

/* Find an indexed full block with this hash and identical tokens. */
static uint32_t find_full_block(const kipp_kv_pool *pool, uint64_t hash,
                                const uint32_t *tokens) {
    uint32_t bucket = (uint32_t)(hash % pool->bucket_count);
    for (uint32_t id = pool->buckets[bucket]; id != KIPP_KV_INVALID_BLOCK;
         id = pool->blocks[id].hash_next) {
        const kv_block *block = &pool->blocks[id];
        if (block->hash == hash &&
            block->token_count == KIPP_KV_BLOCK_TOKENS &&
            memcmp(block->tokens, tokens,
                   KIPP_KV_BLOCK_TOKENS * sizeof(*tokens)) == 0) {
            return id;
        }
    }
    return KIPP_KV_INVALID_BLOCK;
}

/* Reference a block, pulling it off the free list if it was evictable. */
static void claim(kipp_kv_pool *pool, uint32_t id) {
    if (pool->blocks[id].ref_count == 0) {
        free_list_remove(pool, id);
    }
    ++pool->blocks[id].ref_count;
}

/* Pop the LRU free block, evicting any stale indexed content it holds.
 * KIPP_KV_INVALID_BLOCK when nothing is evictable. */
static uint32_t claim_free_block(kipp_kv_pool *pool) {
    if (pool->free_head == KIPP_KV_INVALID_BLOCK) {
        return KIPP_KV_INVALID_BLOCK;
    }
    uint32_t id = pool->free_head;
    free_list_remove(pool, id);
    if (pool->blocks[id].in_index) {
        index_remove(pool, id);
        ++pool->evicted_total;
    }
    return id;
}

/* --------------------------------------------------------------- public ops */

uint32_t kipp_kv_pool_acquire(kipp_kv_pool *pool, uint64_t parent_hash,
                              const uint32_t *tokens, uint32_t count,
                              bool *reused) {
    if (reused != NULL) {
        *reused = false;
    }
    if (pool == NULL || tokens == NULL || count == 0 ||
        count > KIPP_KV_BLOCK_TOKENS) {
        return KIPP_KV_INVALID_BLOCK;
    }
    bool full = count == KIPP_KV_BLOCK_TOKENS;
    uint64_t hash = full ? pool_hash(pool, parent_hash, tokens, count) : 0;
    if (full) {
        uint32_t hit = find_full_block(pool, hash, tokens);
        if (hit != KIPP_KV_INVALID_BLOCK) {
            claim(pool, hit);
            ++pool->reused_total;
            if (reused != NULL) {
                *reused = true;
            }
            return hit;
        }
    }
    uint32_t id = claim_free_block(pool);
    if (id == KIPP_KV_INVALID_BLOCK) {
        return KIPP_KV_INVALID_BLOCK; /* exhausted: nothing evictable */
    }
    kv_block *block = &pool->blocks[id];
    memcpy(block->tokens, tokens, count * sizeof(*tokens));
    block->token_count = count;
    block->ref_count = 1;
    block->hash = hash;
    if (full) {
        index_insert(pool, id);
    } else {
        block->in_index = false;
        block->hash_next = KIPP_KV_INVALID_BLOCK;
    }
    return id;
}

void kipp_kv_pool_release(kipp_kv_pool *pool, uint32_t block_id) {
    if (pool == NULL || block_id >= pool->block_count) {
        return;
    }
    kv_block *block = &pool->blocks[block_id];
    if (block->ref_count == 0) {
        return;
    }
    if (--block->ref_count == 0) {
        free_list_push_tail(pool, block_id);
    }
}

uint32_t kipp_kv_pool_prefix_match(kipp_kv_pool *pool, const uint32_t *tokens,
                                   uint32_t token_count, uint32_t *out_blocks,
                                   uint32_t max_blocks, uint32_t *matched_tokens,
                                   uint64_t *out_parent_hash) {
    uint32_t matched = 0;
    uint64_t parent = 0;
    if (pool != NULL && tokens != NULL && out_blocks != NULL) {
        /* Cap one token short so a token is left to produce logits. */
        uint32_t cap_blocks =
            token_count > 0 ? (token_count - 1) / KIPP_KV_BLOCK_TOKENS : 0;
        while (matched < cap_blocks && matched < max_blocks) {
            const uint32_t *block_tokens =
                tokens + (size_t)matched * KIPP_KV_BLOCK_TOKENS;
            uint64_t hash =
                pool_hash(pool, parent, block_tokens, KIPP_KV_BLOCK_TOKENS);
            uint32_t hit = find_full_block(pool, hash, block_tokens);
            if (hit == KIPP_KV_INVALID_BLOCK) {
                break;
            }
            claim(pool, hit);
            ++pool->reused_total;
            out_blocks[matched] = hit;
            parent = hash;
            ++matched;
        }
    }
    if (matched_tokens != NULL) {
        *matched_tokens = matched * KIPP_KV_BLOCK_TOKENS;
    }
    if (out_parent_hash != NULL) {
        *out_parent_hash = parent;
    }
    return matched;
}

uint32_t kipp_kv_pool_alloc(kipp_kv_pool *pool) {
    if (pool == NULL) {
        return KIPP_KV_INVALID_BLOCK;
    }
    uint32_t id = claim_free_block(pool);
    if (id == KIPP_KV_INVALID_BLOCK) {
        return KIPP_KV_INVALID_BLOCK;
    }
    kv_block *block = &pool->blocks[id];
    block->ref_count = 1;
    block->token_count = 0;
    block->hash = 0;
    block->in_index = false;
    block->hash_next = KIPP_KV_INVALID_BLOCK;
    return id;
}

int kipp_kv_pool_seal(kipp_kv_pool *pool, uint32_t block_id,
                      uint64_t parent_hash, const uint32_t *tokens) {
    if (pool == NULL || block_id >= pool->block_count || tokens == NULL) {
        return -1;
    }
    kv_block *block = &pool->blocks[block_id];
    if (block->ref_count == 0 || block->in_index) {
        return -1;
    }
    uint64_t hash =
        pool_hash(pool, parent_hash, tokens, KIPP_KV_BLOCK_TOKENS);
    memcpy(block->tokens, tokens,
           KIPP_KV_BLOCK_TOKENS * sizeof(*tokens));
    block->token_count = KIPP_KV_BLOCK_TOKENS;
    block->hash = hash;
    /* One indexed copy per content: if an identical block is already
     * published, this one stays private and the index keeps the original. */
    if (find_full_block(pool, hash, tokens) == KIPP_KV_INVALID_BLOCK) {
        index_insert(pool, block_id);
    }
    return 0;
}

void kipp_kv_pool_get_stats(const kipp_kv_pool *pool,
                            kipp_kv_pool_stats *out_stats) {
    if (out_stats == NULL) {
        return;
    }
    memset(out_stats, 0, sizeof(*out_stats));
    if (pool == NULL) {
        return;
    }
    out_stats->total_blocks = pool->block_count;
    out_stats->free_blocks = pool->free_blocks;
    out_stats->reused_blocks_total = pool->reused_total;
    out_stats->evicted_blocks_total = pool->evicted_total;
}

#ifdef KIPP_TESTING
void kipp_kv_pool_test_force_hash(kipp_kv_pool *pool, bool force) {
    if (pool != NULL) {
        pool->force_hash = force;
    }
}
#endif
