# Batching and KV Pooling Design Notes

These notes record the design decisions behind Kipp's continuous batching,
paged KV addressing, and cross-request prefix reuse. Reference repositories
informed the boundaries; no implementation was vendored.

## Delivered design

- `kipp_eval_batch` evaluates multiple sessions per backend call. Metal packs
  unfinished items into shared rounds so weights are reused across sequences.
- The server uses a `poll()` event loop with FIFO admission, chunked prefill,
  batched decode, idle-connection reaping, and disconnect cancellation.
- CPU and Metal sessions address KV through 32-position block tables.
- `kipp_model_open_pooled` gives CPU and Metal sessions block tables into one
  model-owned slab.
- Finished sessions publish complete blocks to a content-addressed pool;
  `kipp_session_match_prefix` adopts the longest verified prefix in a new
  session.
- Shared blocks are complete and immutable. Partial append blocks remain
  private, so truncation and speculative rollback cannot mutate shared state.
- Admission reserves worst-case blocks before starting work. Pool pressure
  delays a request in the FIFO instead of corrupting an active sequence.

The `--paged-cpu`, `--paged-metal`, `--pooled-cpu`, and `--pooled-metal`
gates cover placement invariance, shared-prefix equivalence, mixed batched
evaluation, exhaustion, collision verification, truncation, and eviction.
Server integration tests cover cancellation and pressure at the API boundary.

## Reference design: nano-vllm

The small block manager in `GeeeekExplorer/nano-vllm` provided a useful
comparison:

- Fixed-size blocks carry an identifier, reference count, chained content
  hash, and token IDs.
- Hash hits are verified against exact token IDs before reuse.
- Full released blocks remain indexed for revival and are reclaimed through
  an LRU free list under pressure.
- Partial blocks remain private.
- Prefill separates query length from key length so cached tokens can be
  skipped.

Kipp follows those safety properties while keeping ownership inside each
backend model and preserving the existing public session API.

## Scheduler contract

- Waiting and running queues keep admission separate from execution.
- Capacity is reserved before a request starts.
- Cache exhaustion delays admission rather than failing an active request.
- Batching must preserve each backend's isolated-evaluation tolerance.
- Future preemption policies must preserve deterministic FIFO semantics.

## Metal attention

ggml's Metal attention implementation confirmed the online-softmax reduction
shape used by Kipp: parallel ranges maintain running maxima, sums, and
rescaled accumulators before reduction.

Kipp now uses tiled matrix prefill attention and split-K long-context decode.
Further KV work remains deliberately separate: radix-tree indexing,
persistence, offload, and KV quantization each require their own correctness
gate and measured justification.
