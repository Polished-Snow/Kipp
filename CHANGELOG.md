# Changelog

All notable changes to Kipp are recorded here. Versions are pinned to the
BF16 reference behavior: the v0.0.1 forward pass remains byte-identical.

## Unreleased

### Cross-request KV prefix sharing (CPU + Metal)
- **Pooled models** (`kipp_model_open_pooled`, CPU and Metal backends): all
  sessions
  share one model-owned KV slab; a finished session's full 32-token blocks
  are published to the content-addressed block pool, and
  `kipp_session_match_prefix` lets a new session adopt a cached prefix
  instead of re-evaluating it. Shared blocks are immutable and complete;
  appends always land in private blocks, so speculative rollback never
  touches shared state. Gated by `--pooled-cpu` (`make test-pooled-cpu`):
  pooled identity, shared and batched-mixed evaluation bitwise-equal to
  unshared runs, clean exhaustion, truncation, and eviction. The Metal
  backend shares one model-owned `MTLBuffer` slab with zero shader changes,
  gated the same way by `--pooled-metal` (`make test-pooled-metal`) plus a
  `1e-4` NMSE anchor against the CPU oracle.
- `kipp_kv_pool` gains `alloc`/`seal` (publish-at-finish), reuse/eviction
  stats, and a collision-safety test hook; `kipp_model_kv_pool_stats`
  exposes the counters.
- The eval-item contract gains an optional per-item `block_table`; backend
  function signatures are unchanged, and non-pooled models are
  byte-identical to v0.0.2.
- **Serving**: the server opens CPU/Metal models pooled by default
  (`--kv-pool-mib` sizes the pool, default = the checkpoint's context
  length; `0` disables). Every choice adopts the longest published prompt
  prefix at admission, and admission reserves worst-case pool blocks so
  pool pressure delays requests in the FIFO instead of failing them.
  `GET /metrics` gains `kipp_kv_pool_*` occupancy/reuse counters and
  `kipp_prefix_tokens_reused_total`. The single-slot serial prefix cache
  now serves only non-pooled (CUDA) models. Covered by `make test-server`,
  including cross-request reuse, multi-choice determinism, mid-stream
  disconnect release, and a tiny-pool pressure test.

## v0.0.2

Expands Kipp from a single pinned checkpoint to the **Qwen3 dense family**
while keeping the engine narrow and every feature gated against the CPU
oracle on real hardware.

### Model support
- Compiled-in **checkpoint registry** (`src/kipp_checkpoints.h`): the Qwen3
  dense family (0.6B–32B, base + instruct) with strict per-checkpoint
  revision and tensor-shape validation. Support grows only by adding a
  registry entry; the BF16 path stays byte-identical to v0.0.1.
- Whole family gated on CPU + a GPU backend — Metal on Apple M5, CUDA on
  NVIDIA A100 (ephemeral cloud) for the larger checkpoints.

### Quantization
- **Q8_0** (near-lossless) and **4-bit affine, group-size 32** on the seven
  per-layer projections; CPU + Metal, bit-accurate between backends.
  Q8_0 brings the 8B checkpoint under the M5 single-buffer cap.

### Serving
- **Chat Completions** (`/v1/chat/completions`) with a native Qwen3 ChatML
  renderer (`enable_thinking` aware), alongside the existing Completions API.
- Sampling pack: `top_k`, `min_p`, frequency/presence/repetition penalties,
  `logit_bias`, per-request seed.
- **Generated-token logprobs** (`logprobs` / `top_logprobs`),
  `stream_options.include_usage`, a llama.cpp-style `timings` object, and a
  Prometheus **`/metrics`** endpoint.

### Decode & evaluation
- **Multi-logit evaluation** (`kipp_session_eval_n`) returning logits for the
  last N tokens.
- Greedy **speculative decoding** via prompt-lookup drafting (CLI `--spec`),
  token-identical to plain greedy decoding.

### Paged KV (in progress)
- Both the CPU oracle and the Metal backend now address KV through a
  per-session **32-position block table**. The identity mapping is
  byte-for-byte the contiguous layout; the `--paged-cpu` and `--paged-metal`
  gates prove correctness under a deliberately scrambled table.
- A shared cross-request KV **block pool** (`src/kipp_kv_pool.c`) is
  implemented and unit-tested, not yet wired into the backends. CUDA remains
  contiguous.

## v0.0.1

Initial release: a hand-written C11 inference engine for one pinned
Qwen3-4B-Base checkpoint, with a CPU reference oracle, a validated Metal
backend, best-effort CUDA kernels, and a minimal OpenAI Completions server.
