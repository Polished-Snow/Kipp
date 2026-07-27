# Changelog

All notable changes to Kipp are recorded here. Versions are pinned to the
BF16 reference behavior: the v0.0.1 forward pass remains byte-identical.

## v0.0.4 — 2026-07-27

### Metal prefill: ~2.5× faster, and the gate hole that hid the kernels (2026-07-26)
- **Metal prefill throughput ~2.5× faster on every weight scheme.**
  Back-to-back same-session A/B, Qwen3-4B at a 2,048-token prompt (M5 Max, five
  runs each): BF16 504.1 → **1308.2 tok/s (2.60×)**, Q8_0 453.6 → **1141.7
  (2.52×)**, affine4 482.1 → **1139.5 (2.36×)**. **Decode is unchanged** for all
  three (55.99 → 56.47, 85.79 → 85.68, 110.04 → 110.21). This narrows the
  llama.cpp prefill gap from ~4.5× to ~1.7× while decode stays ~1.6× ahead.
  Every step is bit-exact: the full-logit fingerprint is unchanged for all three
  schemes through the whole change set.
- **The matmul token tile is per-kernel, not global.** The quantized kernels
  already stage dequantized weights in ~16.9 KiB of threadgroup memory, so
  widening their tile costs more occupancy than it saves traffic: on Q8_0 it is
  **−88%** (1139 → 135 tok/s) where BF16 gains 7%. The quantized kernels keep
  the 16-token tile. Correctness could not have caught this — every gate passed
  at either tile — which is why the change was benchmarked per scheme rather
  than extrapolated from BF16.
- **Three changes were measured and reverted rather than shipped:** four query
  tiles per attention threadgroup (flat at 2,048 tokens, −7.3% at 12,800 — the
  KV tiles were already cache-resident and the larger threadgroup allocation cut
  occupancy), a 64-token matmul tile (−90%, register spilling), and a
  1,024-token round (faster prefill but −2.6% decode). The round size was swept
  rather than assumed.
- **Split one overloaded constant into three.** `KIPP_METAL_BATCH` simultaneously
  bounded the shared logits buffer (per item), the activation workspace (per
  token), the split-K partial buffer, and the per-item host arrays. It is now
  `KIPP_METAL_LOGIT_ROWS` (rows, still 32), a device-derived prefill token count,
  and a separate partial-token width. This is a throughput change, not only a
  memory one: widening everything together costs 11% of decode, because decode
  reads the split-K partials every step.
- **One activation workspace per model instead of per session.** Sizing a wide
  round per session would charge every live server session ~85 MB at 4B.
- **Wider prefill rounds** (device-derived, floor at the previous value) fill the
  GPU: the 1,024-row K and V projections previously dispatched 16 threadgroups
  onto a 40-core device. This is the dominant lever, worth +142% alone.
- **32-token matmul tile** (was 16), halving projection weight traffic for a
  further +7%. Note the ordering: applied *before* the wider round it is a 29%
  regression, because it halves the dispatch grid.
- **New `--prefill-metal` gate.** Every pinned test vector is three tokens long,
  but a round only takes the simdgroup-matrix path at eight tokens or more, so
  `--model` and `--phase3-metal` never executed `kipp_matmul_*` or
  `kipp_flash_gqa_prefill` at all. Only `--pooled-metal` reached them,
  incidentally, at a length that is an exact multiple of the round size — so no
  gate covered a ragged final round, and widening the round would have collapsed
  even that coverage to a single round. The new gate checks a ragged 650-token
  multi-round prefill against both the vector path and the CPU oracle, and prints
  a full-logit fingerprint so a change meant to be bit-exact can be verified with
  a one-line diff.
- **Fixed an out-of-bounds write.** All three matmul kernels and the prefill
  attention kernel computed `min(tile, token_count - token_base)`, which wraps in
  unsigned arithmetic when the grid is taller than the token count, yielding a
  full phantom tile that reads *and writes* past the buffers. The quantized
  operator test hardcoded its token-group count and would have written 32 rows
  past a 19-row output buffer as soon as the tile changed.
- **Two new tripwires for silent degradation**, the failure mode that once cost
  two days to a reserved-keyword typo: `KIPP_METAL_REQUIRE_MMA=1` turns a
  fallback to the vector path into a load failure rather than a warning, and a
  geometry probe reads the shader's own tile constants back at model open and
  fails on drift from the host mirror. Both are negative-tested.
- **First operator test for `kipp_flash_gqa_prefill`**, which had none: compared
  against the streaming kernel over a ragged token count under a reversed page
  table. The two cannot agree bitwise — the streaming kernel keeps the query in
  FP32 while the matrix kernel stages a BF16 query tile for `simdgroup_load` — so
  the bound sits just above that ~2e-6 floor.
- Corrected `docs/BENCHMARKS.md`: its claim that prefill was limited by per-layer
  KV re-reads rather than projection matmuls was measured on a 32-token round and
  does not describe the current engine. Attention *is* the dominant remaining
  term, but for a different reason; the corrected traffic model is recorded.

### Quantized KV cache (2026-07-24)
- **Opt-in Q8_0 KV cache** (`--kv-quant q8_0` on the CLI and server;
  `kipp_model_open_ex`): each 32-value block of the key/value cache is stored
  as a bf16 scale plus int8 quants, ~1.9× smaller than the BF16 default (136
  vs 256 bytes per head-row). It composes with the pooled KV cache and
  speculation; CUDA rejects it and stays BF16. The BF16 KV path is
  byte-identical to before (all existing gates unchanged).
- **Gates**: `--qkv-cpu` / `--qkv-metal` (`make test-qkv-cpu` /
  `test-qkv-metal`) prove Q8_0 KV within NMSE 1e-3 and an identical arg max
  of the BF16 cache, and bitwise placement-invariant under a scrambled block
  table. Measured on the 4B: CPU NMSE 1.36e-5, Metal 2.58e-5, arg max
  identical, scramble bitwise; greedy output byte-identical to BF16 KV.
- **Throughput** (Metal 4B, same-session A/B): Q8_0 KV trades a modest decode
  slowdown (0.85–0.97× across 512–16,384 tokens) for the ~1.9× memory
  reduction — the per-value dequantization outweighs the reduced byte
  traffic, so it is a memory feature, not a speed one. Quantized prefill
  routes through the streaming attention kernel (the MMA prefill kernel
  cannot consume Q8_0 blocks); a threadgroup-dequant MMA prefill and a
  faster decode dequant are the natural follow-ups.

## v0.0.3 — 2026-07-23

### Draft-model speculative decoding (2026-07-23)
- **`--draft-model M.gguf`** on the CLI: a small pinned-family checkpoint
  (e.g. Qwen3-0.6B) drafts up to eight tokens autoregressively and the
  target (`--model`) verifies the block in one multi-row forward, accepting
  the longest prefix that matches its greedy arg max. The whole family
  shares one tokenizer and vocabulary, so no token remapping is needed, and
  the two sessions are kept in lockstep on committed tokens. The emitted
  sequence is byte-identical to the target's plain greedy decode — gated by
  `make test-draft-spec` and verified across backends and prompts. Benchable
  via `bench/spec_bench.py --draft-model`.

### Long-context decode and prefill softmax (2026-07-23)
- **Split-K long-context decode**: the Metal flash-GQA decode kernel splits
  each head's KV scan across up to eight threadgroups past 1,024 cached
  positions; 12,800-token Q8_0 decode improves ~1.7× (26 → 44.7 tok/s, from
  ~36% to ~67% of the bandwidth roofline). The split count derives from a
  token's own position, so decode and speculative verify partition
  identically and shorter contexts stay bit-identical (`--longctx-metal`).
- **All-lane prefill softmax**: the tiled prefill kernel's online-softmax
  step now uses all 32 simdgroup lanes via quad shuffles.

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
  H100 80GB (worst observed NMSE 5.9e-7), with the complete gate record
  bound to that committed file.

### Measurement, quality, and provenance (2026-07-21)
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
  manifest; the dirty flag excludes freshly written benchmark results.
- **GPU steady-state measurement protocol** documented in `bench/README.md`
  and `docs/REPRODUCING.md`; all committed numbers re-measured under it on the
  M5 Max, including new evals: perplexity per scheme, a 0.6B/4B/8B
  model-size sweep, context scaling to 12,800 tokens, open-loop serving
  load, and a matched llama.cpp head-to-head (pinned commit, 2,048-token
  prefill, Q4_0 point).

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
