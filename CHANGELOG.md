# Changelog

All notable changes to Kipp are recorded here. Versions are pinned to the
BF16 reference behavior: the v0.0.1 forward pass remains byte-identical.

## v0.0.8 — 2026-08-09

### Quantized projections on the Metal 4 tensor pipeline
- **Q8_0 and affine4 weight projections now run on `mpp::tensor_ops`** on
  M5-class devices — the last prefill path still on the simdgroup kernels.
  Each K-block is dequantized into a threadgroup bf16 tile, then fed to
  `matmul2d` (the BF16 kernel needs no dequant, so it stages nothing; the
  quant kernels reuse the simdgroup cooperative dequant prologue). Same
  output tile geometry, transposed grid, and geometry probe as the BF16
  tensor path. Decode is untouched (matvec; prefill-only change).
- **Q8_0 prefill @2,048: 1,247 → 2,875 tok/s (2.31×)**; affine4 1,250 →
  2,857 (2.29×) — same-session A/B against the simdgroup kernels each
  replaces, on an M5 Max, steady-state (MAD ≤0.3%). Committed records
  (tensor path): q8_0 2,882 @2k / 1,940 @348, affine4 2,868 / 1,960.
- **Why it works:** the same neural-accelerator instruction class that gave
  BF16 projections 2.8× in v0.0.6. A Phase-0 spike first confirmed
  `matmul2d` accepts a threadgroup-address-space weight operand on M5
  (bitwise-exact); `kipp_tensor_probe_bf16` was extended to exercise it so a
  device that can't build it demotes cleanly rather than hard-failing.
- **Gated like the BF16 tensor path.** Selection is a pure function of token
  count (quant-tensor at ≥ one token tile, simdgroup below), so the
  paged/pooled bitwise gates stay single-class; Q8_0 KV and non-M5 /
  `KIPP_METAL_TENSOR_DISABLE` keep the simdgroup kernels, whose fingerprints
  are frozen (7421c99f0af71365 / 0709c0eb78978e61). Tensor-state quant
  fingerprints re-baseline (q8_0 33b605f4f5a05243, affine4 0953f2975a2692d5;
  nmse vs CPU oracle 9.2e-07 / 2.2e-06). New tolerance-0 quant-tensor
  operator test; bf16 + pooled reprint unchanged — the change is isolated.
- **No llama.cpp head-to-head this release:** llama-bench's quantized
  prefill is thermally unstable on this laptop chassis (measured 2.1k–4.4k
  tok/s across thermal states in one session; it has no steady-state
  protocol), so a fair same-session comparison could not be taken. The
  result above is Kipp-internal, steady-state, and same-session.

### Validation
- Gated on Apple M5 Max (CPU + Metal); operator suite 38/0, all per-class
  fingerprints reprint. CUDA was not revalidated this release.

## v0.0.7 — 2026-08-08

### Panel-flash attention on the Metal 4 tensor pipeline: long-context prefill +37%
- **Batched-prefill GQA attention now runs on `mpp::tensor_ops`** on
  M5-class devices, closing the residue v0.0.6 named as dominant above ~8K
  tokens. Attention is processed one 1,024-KV-position panel at a time:
  gather (block-table honored) → Q·Kᵀ `matmul2d` per head → panel-granular
  online softmax reusing the streaming kernel's merge expressions (bf16
  probabilities) → P·V `matmul2d` → cross-panel rescale/accumulate. Decode
  is untouched — the change is prefill-only.
- **Long-context prefill gets faster the longer the context** (same-session
  A/B vs the simdgroup flash-attention kernel it replaces, cool-start
  matched on an M5 Max): **+37% at 12,800 tokens** (2,410 vs 1,754 tok/s),
  +26% at 6,400, +11% at 2,048 — the win tracks attention's share of the
  wall, which grows linearly with context. Decode unchanged (+0.1%).
- **Kipp now leads llama.cpp at long context.** Same-session, Kipp
  panel-flash reaches 2,410 tok/s at a 12,800-token prefill against
  llama.cpp's 1,561 (**1.54×**); at 2,048 tokens Kipp's 4,443 edges
  llama.cpp's 4,100. llama.cpp's flash-attention is simdgroup-only even on
  M5, so it stays on the slow attention class at long context.
- **Why it works:** attention is ~85% of the 12.8k prefill wall (~48.3
  TFLOP at 12,800 tokens and growing linearly), and the simdgroup
  flash-attention kernel ran only ~3 TFLOP/s effective at these panel
  shapes versus `matmul2d`'s 20–25 TFLOP/s. Moving the class onto the
  tensor units converts the largest remaining prefill cost to the fast path.
- **Gated like the projection path, and re-baselined tighter.** The tensor
  attention rung runs only behind the tensor-state check and BF16 KV; the
  simdgroup FA kernel stays frozen for other devices, for
  `KIPP_METAL_TENSOR_DISABLE`, and for Q8_0 KV. Tensor fingerprints were
  re-baselined tighter on every scheme (bf16 721ea327c1facaed / 7.46e-07,
  q8_0 0c251632a7a38a3d, affine4 52161133c69d0d76, pooled 6.05e-07);
  `KIPP_METAL_TENSOR_DISABLE` reprints the frozen simdgroup values
  (49e2ada96bce2804 / 2.08738076e-06). Paged placement invariance is
  bitwise through the new gather, and the geometry probe was extended to
  pin the panel tile constants. The engine and its operator test share one
  panel-encode routine, so their dispatch geometry cannot drift.
- **Also in this release:** a bitwise skip-identity-rescale in the
  simdgroup FA kernel (no numeric change); a new `--fa-bench-metal`
  attention instrument; and `bench.py` / `_provenance` now refuse to record
  on battery, in Low Power Mode, or behind a sub-90 W adapter — three
  disguises of the same silent GPU power-limiting.

### Validation
- Gated on Apple M5 Max (CPU + Metal); operator suite 38/0, all
  per-kernel-class fingerprints reprint. CUDA was not revalidated this
  release. The 12,800-token figures are cool-start medians: the laptop
  chassis cannot hold steady state under sustained long-context prefill,
  so both Kipp and llama.cpp are measured with an idle-GPU cooldown before
  each run (see `docs/BENCHMARKS.md`).

## v0.0.6 — 2026-07-30

### Metal 4 tensor-ops prefill: BF16 2.80× — the instruction-class gap is closed
- **BF16 prefill @2048 tokens: 1,312 → 3,682 tok/s (2.80×)** on the M5 Max,
  via a `mpp::tensor_ops` (neural-accelerator) matmul path for BF16 layer
  projections. Same-session interleaved A/B with `KIPP_METAL_TENSOR_DISABLE`
  toggling the class: 348-token prefill 2.41× (1,031 → 2,489), 12,800-token
  1.40× (attention now dominates long context), **decode identical** (the
  tensor path never touches decode) and the **Q8_0 control equal in both
  states** (quant fingerprints bit-frozen: 7421c99f0af71365,
  0709c0eb78978e61). This is at parity with llama.cpp's own tensor path —
  the ~2.9× instruction-class deficit v0.0.5 documented is gone.
- **The kernel departs from llama.cpp's design where measurement said to.**
  A 64-row × 128-token threadgroup runs matmul2d over device tensors whose
  extents clip every ragged edge — but **no operand is staged**: llama.cpp's
  staged-weight shape measured 0.96–2.9× vs Kipp's simdgroup kernels while
  the no-stage shape reaches 4.9–6.2× (45–65 TFLOP/s, rotation-immune), the
  sixth loss for explicit threadgroup staging on this GPU. Activations are
  consumed as **FP32 straight from the round's working buffer** (matmul2d
  takes float sources): ~12–18% slower per GEMM than bf16 inputs, but it
  deletes the entire `kipp_bf16_stage` pass (four dispatches per layer,
  ~90 ms per 2,048-token prefill) and one activation rounding step — a net
  ~+10% end-to-end and strictly better numerics.
- **Numerically tighter than the class it replaces**, not merely in-gate:
  pooled-vs-oracle NMSE 5.12e-07 (simdgroup: 2.09e-06), matrix-vs-vector
  2.78e-06 (simdgroup: 1.52e-05), identical argmax throughout. matmul2d
  output measured **bitwise invariant to the dispatch token count and
  bitwise deterministic across runs**, which is what lets the wholesale
  class swap keep the paged/pooled bitwise placement gates valid (they now
  cover the tensor path). Per-class tripwires recorded: tensor
  `--prefill-metal` fingerprint 794f1799c6a7326a / pooled 5.12341885e-07;
  the frozen simdgroup values (49e2ada96bce2804 / 2.08738076e-06) stay
  verifiable via `KIPP_METAL_TENSOR_DISABLE=1`.
- **Gated like everything else, plus tripwires for a path CI cannot run.**
  Three-rung runtime compile ladder: the tensor rung runs only behind an
  env kill switch, the Metal 4 GPU family check, an M5/M6/A19/A20 device
  allow-list (`KIPP_METAL_TENSOR_ENABLE=1` overrides), and a probe-pipeline
  compile; any failure demotes cleanly to the simdgroup rungs.
  `KIPP_METAL_REQUIRE_TENSOR=1` turns a silent fallback into a load failure
  (it caught a real kernel bug during development), bench.py refuses
  degraded-build numbers under that env, CI asserts the affirmative
  "tensor path disabled (<reason>)" line on its non-M5 runners, and the
  geometry probe grew tensor-tile slots. The operator suite gained an
  exact-integer tensor matmul comparison at tolerance 0.0 on a deliberately
  ragged 72×133 shape (integer FP32 accumulation is order-independent, so a
  wrong matmul2d descriptor cannot hide inside a tolerance).
- **Scope and honesty:** BF16 projections only — quantized matmuls, the
  LM head, and attention keep their existing kernels (quant tensor variants
  and the now-dominant attention/elementwise residue are the next
  campaigns; the server's 32-token prefill chunks also benefit, ~2× at
  8-token rounds, but server chunk tuning remains future work). CUDA is
  untouched and was not run.

## v0.0.5 — 2026-07-28

### The llama.cpp comparison was stale — corrected, and the loss stated plainly
- **The published head-to-head was wrong in Kipp's favor and is withdrawn.**
  Re-running the exact committed llama-bench command, binary, weights, and OS
  gives ~2× the recorded llama.cpp numbers on both axes (prefill 2,174 →
  ~3,800; decode 35.0 → ~62), while Kipp reproduces its own committed numbers
  in the same session — the 2026-07-22 llama.cpp session ran in a degraded
  GPU-clock state. Corrected table (Qwen3-4B, M5 Max,
  `llamacpp-qwen3-4b.json`): decode is **llama.cpp ~3% ahead on matched
  schemes** (63.1/100.5 vs Kipp 61.2/97.4, BF16/Q8_0, isolated cooled runs)
  and clearly ahead on 4-bit decode (Q4_0 149.6 vs affine4 128.8, schemes
  differ); prefill @2048 is a **2.9×/2.25× llama.cpp win** on BF16/Q8_0 and
  parity on 4-bit. Cross-engine numbers are now re-validated in the same
  session as the Kipp numbers they are compared against.
- **The cause is an instruction class, and it sets the roadmap.** On M5,
  llama.cpp routes prefill GEMM through the Metal 4 tensor API
  (`mpp::tensor_ops`, the neural accelerators) — enabled by its own allow-list
  only on M5/M6/A19/A20, worth ~2.3–3× on prefill and nothing on decode.
  Forced onto the simdgroup-matrix class Kipp uses
  (`GGML_METAL_TENSOR_DISABLE=1`), llama.cpp measures 1,261/1,022/608 —
  **Kipp leads its own instruction class on every scheme.** A gated
  tensor-API matmul path is the next roadmap item.

### Quantized prefill +10–12%, and what the matmuls are actually bound by
- **One shared dequant block per quantized matmul threadgroup.** The
  Q8_0/affine4 kernels' four private 32-row staging blocks (16.9 KiB) became
  one cooperatively-dequantized shared block (4.2 KiB) under a 32-row ×
  64-token tile: weights dequantize once per 64 tokens instead of once per
  16, and occupancy rises several-fold. Bit-exact (fingerprints and the
  pooled tripwire unchanged to the digit); against the committed v0.0.4
  records, 2,048-token prefill is **Q8_0 1,101 → 1,213 (+10.2%)** and
  **affine4 1,087 → 1,214 (+11.7%)**, decode unchanged. BF16 untouched.
- **`--mm-bench-metal`: an isolated projection-matmul instrument.** Runs the
  live matmul pipelines on every 4B shape with rotating vs reused weight
  buffers. It shows the matmuls are **compute/issue-bound** (rotating ≈
  reused; 17–24 GB/s effective weight fetch against ~455 GB/s available;
  8.7–12.3 TFLOP/s), attributing ~82% of prefill wall clock to projections —
  which retires the roadmap's traffic-motivated "K-blocked matmul" plan.
- **Measured and reverted:** threadgroup-staging the activation tile (−7–10%
  every shape — the fourth loss of explicit staging to the implicit cache);
  widening the quant token slice 16 → 32 per simdgroup (~−90%, register
  spilling, confirming the slice's register ceiling).

### Distribution
- **Prebuilt binaries on releases.** `make dist` builds
  `kipp-macos-arm64.tar.gz` (kipp, kipp-metal, kipp-server,
  kipp-server-metal) and `kipp-linux-x86_64.tar.gz` (CPU binaries), and
  `.github/workflows/release.yml` gates them (version check, ad-hoc-signature
  and deployment-target verification, hermetic suite, Metal operators with
  `KIPP_METAL_REQUIRE_MMA=1`) and attaches them with `SHA256SUMS` to every
  published release. macOS binaries target macOS 14+.

### Validation
- CPU and Metal gated on M5 Max (full suite, including the 67-token
  inactive-simdgroup barrier case added to the quant operator test and a
  five-value shader geometry probe). **CUDA is untouched and was not run** —
  no NVIDIA hardware was available; the backend carries its v0.0.3 H100
  validation.

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
