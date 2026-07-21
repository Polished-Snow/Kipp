# Continuous Batching and KV Block Research

State as of v0.1.0-dev. These are design observations from reference
repositories, not vendored implementations.

## Delivered groundwork

- `kipp_eval_batch` evaluates several sessions per backend call. The Metal
  backend packs chunks of unfinished items into shared rounds so weights are
  read once per step across sequences. Gates prove batched output equals
  isolated output.
- The server runs a `poll()` event loop with FIFO admission, chunked prefill,
  batched decode, and disconnect cancellation.
- `kipp_session_truncate` powers the server's single-slot serial prefix cache
  (now the non-pooled fallback path only).
- CPU and Metal sessions use paged addressing.

**Delivered (v0.1.0-dev):** cross-session KV sharing is in production on CPU
and Metal. `kipp_model_open_pooled` gives every session a block table into
one model-owned slab; finished sessions publish full blocks to the
content-addressed pool (`src/kipp_kv_pool.c`), new sessions adopt matching
prefixes with `kipp_session_match_prefix`, and the server admits requests
under a worst-case block reservation so cache pressure delays admission
instead of corrupting active sequences (the exhaustion/eviction/collision
gates below are implemented as `--pooled-cpu` / `--pooled-metal` and the
server-level pool tests). The design followed the nano-vllm notes below:
immutable full shared blocks, private partial blocks, publish-at-finish
instead of copy-on-write.

## nano-vllm block manager

The design in `GeeeekExplorer/nano-vllm` is a useful small-scale reference:

- Fixed-size blocks carry an id, reference count, chained content hash, and
  token ids.
- A hash hit is verified against exact token ids before reuse.
- Full released blocks remain indexed for revival and are reclaimed through
  an LRU free list under pressure. Partial blocks remain private.
- Prefill skips cached tokens by separating query length from key length.

For Kipp, the corresponding production design would use one pooled KV slab
per backend model plus per-session block tables. Backend kernels would map
logical positions through those tables without changing the public session
API.

## Scheduler shape

- Waiting and running queues keep admission separate from execution.
- Admission must reserve enough KV capacity before a request starts.
- Cache exhaustion should delay admission rather than corrupt active
  sequences.
- Any future preemption policy must preserve Kipp's FIFO behavior.

## Metal attention cross-check

ggml's Metal attention code confirms the structure used by Kipp's online
softmax decode kernel: parallel ranges maintain running maxima, sums, and
rescaled accumulators before reduction.

Two possible future optimizations remain outside this cleanup:

- Multiple threadgroups per head with a combine pass for long contexts.
- Tiled matrix operations for prefill attention, complementing Kipp's current
  matrix projection kernels.

## Questions for production KV sharing

- Choose a block size using measured fragmentation and kernel costs rather
  than copying a reference default.
- Define backend ownership for pooled slabs and block-table updates.
- Preserve batched-equals-isolated tolerances when cached and uncached
  sequences share an evaluation step.
- Add cache exhaustion, collision verification, cancellation, and eviction
  integration gates before advertising cross-request prefix reuse.
