# Changelog

All notable changes to Kipp are recorded here. Versions are pinned to the
BF16 reference behavior: the v0.0.1 forward pass remains byte-identical.

## Unreleased

### Metal matrix kernels restored; harness tripwire (2026-07-22)
- **Fixed a reserved-MSL-keyword bug** (`fragment` used as a loop variable
  in `kipp_flash_gqa_prefill`) that made the entire `KIPP_ENABLE_BF16_MMA`
  library fail its runtime compile since 2026-07-20. The bridge silently
  fell back to vector kernels for every matmul and the tiled attention
  kernel, which had therefore never actually executed; every correctness
  gate still passed on the fallback. With the matrix path live, 4B prefill
  measures 528/488/509 tok/s (BF16/Q8_0/affine4) at 348 tokens and
  481/441/466 at 2,048 — quantized prefill at BF16 parity — and Q8_0
  context prefill declines gently to 177 tok/s at 12,800 tokens instead of
  collapsing to 62. Decode is unchanged (bandwidth-bound).
- **`tools/bench.py` now refuses to record Metal results when the
  matrix-kernel fallback warning is present**, so a degraded build can
  never again contaminate committed numbers.
- The pooled gate's batched-mixed case now asserts the documented batching
  contract (bitwise on CPU; `1e-4` NMSE + identical argmax on Metal) — the
  old bitwise demand only ever held because everything ran on the vector
  fallback.
- Full re-measurement campaign committed (`bench/results/`, 32 files, all
  `dirty:false` at one commit), including a same-session llama.cpp
  head-to-head: Kipp decode leads ~1.7× at Q8_0; llama.cpp prefill leads
  ~4.5× at a matched 2,048-token prompt.

### Serving hardening (2026-07-22)
- **32-way concurrent decode**: `SERVER_MAX_GENERATIONS` raised 8 → 32 to
  match `KIPP_EVAL_BATCH_LIMIT`, so single-choice traffic can fill the
  whole batch.
- **Idle-connection reaping**: per-connection activity stamps; sockets
  idle in the reading/draining phases beyond `--idle-timeout` (default
  30 s, `0` disables) are closed. Parsed requests waiting on admission are
  exempt — that backpressure is server-driven.
- **Latency metrics**: `/metrics` gains `kipp_queue_wait_seconds_{sum,count}`,
  `kipp_ttft_seconds_{sum,count}` (to first logits), and
  `kipp_decode_seconds_sum`.
- **JSON and HTTP parsing extracted** into `src/kipp_json.{c,h}` and
  `src/kipp_http.{c,h}` — smaller server file, and both parsers now have
  deterministic fuzz tests (`test_json_parse_fuzz`, `test_http_header_fuzz`)
  running under ASan/UBSan in `make test-sanitize`.
- New server tests: >8-way concurrency, idle-timeout reaping, boundary
  cases (`n`, `stop`, `logit_bias` limits), latency-metric presence.

### CLI chat REPL and sampling fast paths (2026-07-22)
- **`--chat`**: multi-turn REPL for instruct checkpoints reusing the native
  ChatML renderer. Each turn re-renders the full transcript but evaluates
  only the byte suffix (the renderer's prefix-continuation property is
  unit-tested per variant), so the KV cache carries across turns. Flags:
  `--system`, `--no-think`, `--ctx`, per-turn `--decode`; model-gated
  `make test-chat` smoke.
- **Sampling fast paths** in `kipp_sample_ex`: thread-local reusable
  scratch replaces the per-call 600 KB copy + 1.2 MB allocation, and
  greedy sampling without bias/penalties reads the caller's logits
  directly. Sampled tokens are bitwise-identical to the previous
  implementation.
- **`kipp_session_eval_scored`**: multi-row evaluation bound by the
  backend's documented tolerance instead of bitwise decode order, letting
  perplexity scoring use the matrix kernels (spec-verify keeps the strict
  path and its token-identity gate). The Metal lm_head routes through the
  simdgroup-matrix kernel for relaxed rows ≥ 8.

### CUDA revalidation on H100 (2026-07-22)
- `tools/ops/verda_cuda_gate.sh` gained provenance markers and IP-polling
  for current CLI releases; new `tools/ops/collect_cuda_gates.py` turns a
  gate log into `bench/results/cuda-h100-gates.json`. All four default
  checkpoints pass `--model` and `--phase4-cuda` on an ephemeral NVIDIA
  H100 80GB (worst observed NMSE 5.9e-7); the paper's CUDA correctness row
  is now bound to that committed file.

### Paper revision: measurement, quality, and provenance (2026-07-21)
- **CLI `--ppl` perplexity mode**: wikitext-2 perplexity over LE-uint32
  token files (non-overlapping windows, 32-token multi-row chunks,
  double-precision log-sum-exp). CPU and Metal agree to ~1e-4 relative;
  `bench/ppl_bench.py` measures all three weight schemes and verifies the
  backends against each other.
- **CLI `--spec-gate on|off`** makes ungated speculation reproducible;
  `bench/spec_bench.py --gate both` measures a drift-immune paired A/B
  (baseline, ungated, gated adjacent within each run).
- **`bench/_provenance.py`**: every bench script now records the same
  engine/hardware/model block, with model identity read from the GGUF
  manifest; the dirty flag excludes results and derived paper data.
- **`tools/paper_data.py`** + `make paper-data` / `make paper-check`:
  bench/results JSONs are the single source for the paper's macros and
  pgfplots data; `paper-check` (wired into `test-tools`) fails on missing,
  uncommitted, modified, or incoherent inputs and on generated-file drift.
- **GPU steady-state measurement protocol** documented in `bench/README.md`
  and `REPRODUCE.md`; all committed numbers re-measured under it on the
  M5 Max, including new evals: perplexity per scheme, a 0.6B/4B/8B
  model-size sweep, context scaling to 12,800 tokens, open-loop serving
  load, and a matched llama.cpp head-to-head (pinned commit, 2,048-token
  prefill, Q4_0 point).
- **Paper** rebuilt on `acmart` with every number macro-bound to committed
  results; new survey, LoC, and pooled-gate tables; quality-vs-speed,
  model-size, and serving-load figures. (The 2026-07-21 ablation that
  called the tiled flash-prefill kernel slower than split-K was an
  artifact of the silent kernel-compile fallback fixed below.)

### Cross-request KV prefix sharing (CPU + Metal)
- **Pooled models** (`kipp_model_open_pooled`, CPU and Metal backends): all
  sessions
  share one model-owned KV slab; a finished session's full 32-token blocks
  are published to the content-addressed block pool, and
  `kipp_session_match_prefix` lets a new session adopt a cached prefix
  instead of re-evaluating it. Shared blocks are immutable and complete;
  appends always land in private blocks, so speculative rollback never
  touches shared state. Gated by `--pooled-cpu` (`make test-pooled-cpu`):
  pooled identity and shared-prefix evaluation bitwise-equal to unshared
  runs and batched-mixed evaluation within the backend's batching contract
  (bitwise on CPU; `1e-4` NMSE + identical argmax on Metal, where kernel
  selection legitimately differs between batched rounds and isolated
  evaluation), clean exhaustion, truncation, and eviction. The Metal
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
