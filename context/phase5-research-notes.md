# Phase 5 Research Notes — Continuous Batching and KV Blocks

State as of release v0.0.1.

Concept notes gathered from the indexed reference repositories (via Nia).
Per `AGENT.md`, nothing here is vendored; these are design observations to
reimplement natively when Phase 5 begins.

## What Kipp already has (v1 groundwork)

- `kipp_eval_batch` evaluates several sessions per backend call; the Metal
  backend packs chunks of every unfinished item into shared rounds so
  weights are read once per step across sequences (measured ~3x aggregate
  decode at eight sequences, for both `n > 1` and independent concurrent
  clients). Gates prove batched output equals isolated output. The server
  runs a poll() event loop with FIFO admission, chunked prefill (32-token
  steps), and disconnect cancellation. Remaining Phase 5 material: KV
  blocks and cross-session prefix sharing.
- `kipp_session_truncate` rolls a session's timeline back to a shared
  prefix; the server's single-slot cache uses it to skip re-prefilling
  repeated prompt prefixes (measured ~15x latency win on a repeated
  259-token prompt). This is the degenerate, serial form of prefix caching.

## nano-vllm block manager (GeeeekExplorer/nano-vllm, `nanovllm/engine/block_manager.py`)

The smallest credible paged-KV design; a good template for Kipp's scale:

- Fixed-size blocks; each `Block` carries `block_id`, `ref_count`, `hash`,
  and its `token_ids` for verification.
- **Chained content hashing**: block hash = xxh64(previous block hash ||
  block token IDs). A chain therefore identifies the entire prefix, not
  just one block's contents.
- **Cache lookup with double verification**: a hash hit must also match
  the stored `token_ids` exactly before reuse (hash collisions cannot
  poison the cache).
- **Reference counting**: shared blocks increment `ref_count` per
  sequence; a block returns to the free list only at zero. Full blocks are
  hashed and registered after each generation step
  (`scheduler.py:postprocess`); the trailing partial block is never shared.
- Prefill skips cached tokens by giving the attention kernel different
  query and key sequence lengths (`model_runner.py:prepare_prefill`).

Implications for Kipp: the same design works with one pooled KV slab per
backend model plus per-session block tables. The Metal kernels would take a
block-table buffer and translate logical positions to physical block slots;
the backend contract stays unchanged as promised in `ARCHITECTURE.md`.

## Scheduler shape (nano-vllm `scheduler.py`)

- Two queues (waiting/running); admission requires `can_allocate` to
  succeed against the free-block pool, so cache exhaustion blocks
  admission instead of corrupting running sequences.
- Preemption evicts the newest running sequence back to waiting and frees
  its blocks — acceptable for Kipp's FIFO promise.

## Metal attention cross-check (ggml-org/ggml, `src/ggml-metal/ggml-metal.metal`)

ggml's single-token decode kernel (`kernel_flash_attn_ext_vec`) confirms the
architecture Kipp's `kipp_flash_gqa` now uses: several simdgroups stride the
KV sequence (`ic0 += NWG*NSG`), each keeps an online softmax (running
maximum, running sum, rescaled accumulator), and partials merge with
`simd_max`/`simd_sum` reductions. Two things ggml does beyond Kipp v1, noted
for future rounds:

- A second parallelization level across *threadgroups* (multiple workgroups
  per head with a scratch-buffer combine pass) — worth revisiting if Kipp
  targets GPUs with many more cores than the M5, or 32k-context decode.
- The batched-prefill attention path computes Q*K with
  `simdgroup_multiply_accumulate` matrix ops. Kipp now uses
  simdgroup-matrix MMA for prefill *projections* (`kipp_matmul_bf16`,
  ~2.4x prefill on M5, gated at 1.851e-6 NMSE); extending matrix ops into
  the prefill attention kernel itself (a flash-attention-2-style tiled
  Q*K^T with K/V reuse across query tokens) is the remaining route to
  faster long-context prefill, where per-layer KV re-reads through L2 now
  dominate.

## Open questions before Phase 5 starts

- Block size trade-off for a 36-layer, 8-KV-head model (KV per token is
  147,456 bytes across layers; per layer 4 KB — block sizes of 16-32
  tokens match the reference engines).
- Whether Kipp v1's exact-tolerance gates (batched output must equal
  isolated output within backend tolerance) permit sharing one softmax
  scale path between batched and single evaluation — batching must not
  change kernel numerics.
