# Kipp Benchmarks

Kipp has a scalar CPU correctness oracle plus correctness-gated Metal and CUDA
backends. Results must not mix unsupported checkpoints, weight schemes, or
backend variants.

## Required report fields

- Kipp commit and clean/dirty working-tree state
- Model identifier, revision, parameter count, and quantization
- Backend, build target, compiler, and relevant build flags
- Hardware model, memory capacity, and operating-system/driver versions
- Prompt length, generated-token count, batch size, and concurrency
- Warm-up procedure and number of measured runs
- Peak memory, prefill throughput, generation throughput, and latency
- Sampling settings and any cache or scheduler configuration

## Benchmark policy

Report medians and a dispersion measure rather than only the best run. Keep
prefill and token-generation measurements separate. Comparisons with other
engines must use equivalent model weights, context, sampling settings, and
hardware, and must link the exact commands used.

`tools/bench.py` is the canonical subprocess harness. It performs discarded
warm-up runs, captures the CLI's separate prefill and decode timers, samples
peak process RSS with `/usr/bin/time -l`, and reports the median, median
absolute deviation, minimum, maximum, and each raw run as JSON.
`bench/spec_bench.py`, `bench/server_bench.py`, `bench/prefix_bench.py`,
`bench/load_bench.py`, and `bench/ppl_bench.py` cover speculation, server
batching, cross-request prefix reuse, open-loop serving load, and
quantization quality; all record the full engine/hardware/model provenance
block via `bench/_provenance.py`.

All numbers are **sustained steady-state** measurements: Apple-silicon GPU
clocks are demand-scaled, so benches run on an otherwise idle machine,
back-to-back in one session after a multi-minute GPU warm-up, and a file is
trusted only when its recorded dispersion is tight (see `bench/README.md`,
"Measurement protocol").

## Apple M5 Max (v0.0.3 measurement campaign, 2026-07-22)

The development machine changed from a base M5 (10-core GPU, 24 GiB) to an
M5 Max (40-core GPU, 128 GB); throughput is roughly 4× the sections below,
which are retained for the base-M5 configuration. An earlier 2026-07-21
campaign was measured on a build whose Metal matrix kernels had silently
failed to compile (a reserved MSL keyword; every correctness gate still
passed) — those numbers are superseded, and the harness now refuses to
record results from a fallback build. Current reference numbers
(Qwen3-4B, Metal, greedy; every value traces to a committed
`bench/results/*.json`):

- Decode tok/s (64 tokens, median of 5, 2026-07-30 session): BF16 **59.7**,
  Q8_0 **94.4**, affine4 **127.9**. Session-to-session drift on this machine
  is a few percent in either direction (previous campaigns read 61.2 / 97.4 /
  128.8 and 59.8 / 94.3 / 127.9); the controlled back-to-back A/Bs measure
  decode unchanged through every kernel change — including the tensor path,
  which decode never takes — and each campaign is re-run in one session so
  its numbers stay comparable with each other.
- Wikitext-2 perplexity (full test set, 2,048-token windows): BF16
  **7.731**, Q8_0 **7.733** (+0.02%), affine4 **8.171** (+5.7%) — Q8_0 is
  effectively lossless; affine4 is Q4-class.
- Prefill tok/s, BF16, Metal 4 tensor path with panel-flash attention
  (348 / 2,048 / 12,800-token prompt, 2026-08-08): **2601 / 4443 / 2410**
  ("Panel-flash attention" below). The 12,800 figure is a cool-start
  median (the chassis cannot sustain long-context prefill); the 2,048
  figure sits above the 2026-07-30 3,679 mostly from cooler ambient, so
  the panel-flash gain is isolated in the same-session A/B (+37% at 12,800,
  +11% at 2,048) rather than read off the absolute. Quantized prefill,
  Metal 4 tensor path (348 / 2,048-token prompt, 2026-08-09; "Quantized
  projections on the tensor units" below): Q8_0 **1940 / 2882**, affine4
  **1960 / 2868** — 2.3× the simdgroup kernels at 2,048 tokens (recoverable
  with `KIPP_METAL_TENSOR_DISABLE=1`), which stay the shipped path on pre-M5
  devices. The quantized kernels dequantize each 32-weight block once per
  tile.
- Context scaling (Q8_0): decode 98.4 → 44.7 tok/s from a 3- to a
  12,800-token prompt after the split-K long-context path (`ctx-*.json`).
- Model-size sweep (BF16 decode): 0.6B **269**, 4B **60.7**, 8B **33.5**
  tok/s — bandwidth-bound decode scales inversely with streamed weight
  bytes across the family (8B Q8_0: 57.8 tok/s; 8B prefill 316/331).
- Server aggregate (Q8_0, sampled, 48 tokens/sequence): n = 1/2/4/8
  choices → 76.9/96.8/89.8/80.1 tok/s; 1/2/4/8 concurrent connections →
  55.9/61.3/78.5/80.8. Aggregate flattens as per-step scheduling gaps let
  the demand-scaled GPU clocks sag — a consumer-SoC serving effect.
- Cross-request prefix reuse: a 6,890-token prompt sent twice adopts 6,880
  tokens on the second request; prefill drops 30.4 s → 174 ms (**175×**
  TTFT).
- Speculation (Q8_0, 256-token decode, paired-baseline A/B): adaptive-gated
  **2.27×** on repetitive text, **above parity on code (1.24×)**, and a
  **0.84×** floor elsewhere (ungated: 2.10× / down to 0.27×). Gains are
  smaller than earlier drafts because the sampling fast path made the
  plain-decode baseline itself faster.
- llama.cpp A/B: **the 2026-07-22 comparison (decode 35.0/56.5, prefill
  2,174) proved irreproducible and is superseded** — see "The llama.cpp
  comparison was stale" below for the corrected, much less flattering
  numbers and what caused the discrepancy. `llamacpp-qwen3-4b.json` now
  holds the 2026-07-28 measurements.
- CUDA revalidation (`cuda-h100-gates.json`): all four default checkpoints
  pass `--model` and `--phase4-cuda` on an ephemeral NVIDIA H100 80GB
  (worst observed NMSE 5.9e-7 against the CPU oracle).
- **Quantized KV cache** (`--kv-quant q8_0`, opt-in; `bench/results/qkv/`):
  the KV cache is **~1.9× smaller** than BF16 (136 vs 256 bytes per
  head-row), which is the point — it extends context and concurrent-session
  count under Apple's single-buffer cap (the server's 0.6B pool drops from
  ~3,584 to 1,904 MiB). It is a memory feature, **not** a speed one: in a
  same-session Metal 4B decode A/B the per-value dequantization slightly
  *outweighs* the reduced byte traffic, so Q8_0 KV decode runs at 0.85–0.97×
  of BF16 across 512–16,384 tokens rather than faster. Quality is
  near-lossless (NMSE 1.4e-5 CPU / 2.6e-5 Metal vs BF16 KV, arg max
  identical). A faster decode dequant and an MMA quantized-prefill path are
  follow-ups.

### Prefill round shape (2026-07-26)

Metal prefill was the one axis where Kipp lost to llama.cpp: 504 tok/s against
2,174 at a matched 2,048-token prompt, measured back to back on the same host,
weights, and session. Two independent limits caused it, and the arithmetic
matters because the intuitive reading is wrong.

Projection weight traffic is set by the **in-kernel token tile**, not by the
round size:

```
weight bytes per prefill = ceil(total_tokens / matmul_token_tile) x projection_bytes
```

The dispatch grid is `(row groups, token groups)` and each threadgroup streams
its own weight rows over the full shared dimension, so threadgroups that would
share weight rows are never co-resident. Widening the round therefore moves *no*
weight bytes; only the tile does. What the round width fixes is **occupancy**:
the K and V projections have 1,024 rows, so at a 32-token round they dispatched
16 threadgroups onto a 40-core GPU, and the engine sustained only ~255 GB/s,
roughly half of what the device can hold.

Both levers were measured separately, one session, idle machine, five runs each
(Qwen3-4B BF16, 2,048-token prompt, M5 Max):

| matmul tile | round | prefill tok/s | decode tok/s |
|---|---|---|---|
| 16 | 32 (previous) | 504.4 (MAD 0.45) | 55.75 |
| 32 | 32 | **356.6** (MAD 4.58) | 54.16 |
| 16 | 512 | 1222.3 (MAD 8.22) | 56.44 |
| 32 | 512 | **1310.0** (MAD 0.60) | 56.10 |

Three results worth keeping:

- The round is the dominant lever (**+142%** with zero byte reduction); the tile
  adds **+7%** on top. Weight bandwidth was therefore *not* the binding
  constraint, and the widely-assumed fix had the smaller effect.
- Raising the tile **first** is a 29% regression, because at a 32-token round a
  32-token tile halves the grid. Ordering is not cosmetic here.
- Naively widening the round also costs **11% of decode**, because one constant
  sized the split-K partial buffer as well. Separating the limits (see
  `docs/ARCHITECTURE.md`, "Activations and logits") recovers that and lands 40%
  higher prefill than the naive change.

The round size itself was swept rather than guessed (same session, 2,048-token
prompt): 256 -> 1191.4, 512 -> 1273.8, 1024 -> 1340.3, 2048 -> 1292.0 tok/s.
Prefill peaks near 1024, but decode falls from 56.25 to 54.78 there and to 53.57
at 2048, so **512 is the shipped default**: it takes the bulk of the prefill win
while leaving decode -- the larger competitive margin -- untouched.

A third limit turned up only when the quantized schemes were measured, and it
nearly shipped a large regression. The quantized matmul kernels already stage
dequantized weights in ~16.9 KiB of threadgroup memory, so widening *their*
token tile costs far more in occupancy than it saves in weight traffic. Isolated
on Q8_0 at a 2,048-token prompt:

| Q8_0 | 16-token tile | 32-token tile |
|---|---|---|
| 32-token round | 455.1 | 74.9 |
| 512-token round | **1139.4** | 135.1 |

The wider round is worth 2.5x for Q8_0 as well, but the wider tile is **-88%**
for it while being **+7%** for BF16. Tile width is therefore a property of each
kernel rather than of the projection layer, and the two quantized kernels keep
the narrower tile. Correctness alone would never have caught this: all three
schemes passed every gate at either tile.

Back-to-back A/B of the shipped configuration against the previous one, all
three weight schemes, same session, after a sustained warm-up (Qwen3-4B,
2,048-token prompt, five runs each):

| scheme | previous | shipped | ratio | decode before -> after |
|---|---|---|---|---|
| BF16 | 504.1 (MAD 0.11) | **1308.2** (MAD 0.64) | **2.60x** | 55.99 -> 56.47 |
| Q8_0 | 453.6 (MAD 1.00) | **1141.7** (MAD 1.14) | **2.52x** | 85.79 -> 85.68 |
| affine4 gs32 | 482.1 (MAD 0.18) | **1139.5** (MAD 0.31) | **2.36x** | 110.04 -> 110.21 |

Decode is unchanged for every scheme. An earlier run of the same comparison
measured BF16 prefill at 407 tok/s with a median absolute deviation of 10.5 and
affine4 decode 20% low; those readings were thermally polluted and are not used.
The rule this reinforces is in `bench/README.md`: only a back-to-back
same-session A/B is meaningful here, because the narrow-round configuration
spent proportionally more time in per-round fixed cost and therefore tracked GPU
clock state more closely than the new one does.

Every step is bit-exact: the full-logit fingerprint printed by
`--prefill-metal` is unchanged through the refactor, the wider round, and the
larger tile, verified by pinning the round back to 32 to separate the refactor
from the speedup.

Two further changes were implemented, measured, and **reverted** rather than
shipped, because the repository's standard is that a change earns its place:

- **Four query tiles per attention threadgroup**, so the simdgroups sharing a KV
  head walk the same K/V tiles together and three of four reads hit cache. Flat
  at 2,048 tokens (+0.3%) and **-7.3% at 12,800** (532.3 vs 574.0). One layer's
  KV at these lengths is a few megabytes and was already cache-resident, so the
  sharing bought nothing while the larger threadgroup allocation (8.3 -> 25.3
  KiB) cut resident threadgroups per core.
- **A 64-token matmul tile**, which would halve weight traffic again: **133.8
  tok/s, a 90% regression**, from register spilling once the accumulator set
  reaches 32 fragments.

Remaining prefill cost is therefore neither weight traffic nor attention
locality. *(Superseded 2026-07-28: the conclusion that "closing the rest needs
a threadgroup-staged, K-blocked matmul" was tested and falsified — see the next
section. The 2,174 tok/s reference figure it reasoned from was itself stale.)*

*Provenance note: the shipped-configuration figures in this section are backed
by the committed `bench/results/4b-*.json` records from the release re-run; the
intermediate sweep values were measured on the working tree that carried the
change under test, per the policy at the top of this file.*

### What the matmuls are actually bound by, and the llama.cpp comparison was stale (2026-07-28)

Two findings from one measurement day, both of which correct this file.

**The llama.cpp comparison was stale.** The 2026-07-22 A/B recorded llama.cpp
at 2,174 tok/s prefill / 35.0 decode (BF16). Re-running the *exact committed
command, binary, model, and OS* on 2026-07-28 gives ~3,800–4,100 tok/s prefill
and ~60–63 decode — roughly 2× on both axes — while Kipp reproduces its own
committed numbers to within noise in the same session. The old llama.cpp
session was evidently measured in a degraded GPU-clock state (the uniform ~2×
across unrelated workloads is the signature; the exact cause is not
reconstructable). The head-to-head below replaces it, and the lesson is
recorded here precisely because this repository's claims are only as good as
their worst measurement: **cross-engine numbers are now re-validated in the
same session as the Kipp numbers they are compared against.**

**Where the missing prefill time actually goes.** A new isolated instrument
(`build/kipp_test_metal --mm-bench-metal`) runs the live projection-matmul
pipelines on every Qwen3-4B shape at a 512-token round, cycling through eight
distinct weight buffers (so the system-level cache cannot serve one resident
copy) versus reusing one. The result is unambiguous: rotating equals reusing
on every shape, effective weight fetch is 17–24 GB/s against ~455 GB/s of
demonstrated sustained bandwidth, and the kernels run at 8.7–12.3 TFLOP/s.
The matmuls are **compute/issue-bound, not traffic-bound**, and summing the
per-shape times attributes ~82% of 2,048-token prefill wall clock to them.
Weight-traffic reduction — the motivation this file previously assigned to a
future "K-blocked matmul" — cannot help; that plan is retired.

Consistent with that, explicitly staging the activation tile through
threadgroup memory (removing the transposed device-memory gathers from the hot
loop) was implemented, measured at **−7 to −10% on every shape**, and
reverted: the fourth time explicit staging has lost to this GPU's implicit
cache hierarchy. The simdgroup-matrix BF16 kernel sits at ~80% of the
achievable ALU-issue ceiling and stays as is.

**What did work: one shared dequant block per quantized threadgroup.** The
Q8_0/affine4 kernels staged four private 32-row dequantized blocks (16.9 KiB
of threadgroup memory) — one per simdgroup, each serving 16 tokens. They now
share a single cooperatively-dequantized 32-row block (4.2 KiB) across a
32-row × 64-token threadgroup tile: weight bytes are dequantized once per 64
tokens instead of once per 16, and the 4× smaller staging keeps several
threadgroups resident per core. Accumulators, dequant expressions, and FP32
accumulation order are unchanged, so the kernels are **bit-exact** — the
`--prefill-metal` fingerprints and the `--pooled-metal` tripwire did not move
a digit. Same-session interleaved A/Bs: Q8_0 1137.7 → 1206.0 tok/s, affine4
1139.4 → 1213.9 (two passes each, decode dead flat); against the committed
v0.0.4 records the release re-run reads **+10.2% (Q8_0) and +11.7% (affine4)
at 2,048 tokens**. Widening the quant token slice on top of it (16 → 32
tokens per simdgroup) collapses to ~1 TFLOP/s from register spilling — the
third confirmation that the 2-fragment slice is that kernel's register
ceiling — and was reverted.

**The corrected head-to-head** (same session, same host and weights, llama.cpp
178a6c449 with Metal, `llama-bench -p 2048 -n 256 -r 5`; every figure traces
to `llamacpp-qwen3-4b.json` and `4b-*.json`, 2026-07-28):

| Qwen3-4B, M5 Max | Kipp | llama.cpp (default) | llama.cpp (`GGML_METAL_TENSOR_DISABLE=1`) |
|---|---|---|---|
| Prefill BF16 @2048 | 1,312 | **3,788** | 1,261 |
| Prefill Q8_0 @2048 | 1,213 | **2,729** | 1,022 |
| Prefill Q4-class @2048 | **1,214** (affine4) | 1,225 (Q4_0) | 608 (Q4_0) |
| Decode BF16 | 61.2 | **63.1** | 47.0 (interleaved; see note) |
| Decode Q8_0 | 97.4 | **100.5** | 46.8 (interleaved; see note) |
| Decode Q4-class | 128.8 (affine4) | **149.6** (Q4_0) | 51.2 (interleaved) |

llama.cpp's decode figures are from isolated decode-only runs (`-p 0 -n 256`,
cooled machine): its interleaved `-p 2048 -n 256` run heats the GPU enough to
depress its own tg readings to 47–69 tok/s, and quoting those would flatter
Kipp. The honest decode summary is therefore: **llama.cpp is ~3% ahead on the
matched schemes** (63.1/100.5 vs 61.2/97.4) and clearly ahead on 4-bit decode
(scale-only Q4_0 dequantizes more cheaply than scale+bias affine4; the schemes
also differ in quality — affine4's perplexity cost is documented above, Q4_0's
on the llama.cpp stack was not measured here). The prior "decode 1.7× in
Kipp's favor" claim derived entirely from the stale llama.cpp session and is
withdrawn.

The decisive prefill variable is the **instruction class**. On M5, llama.cpp
routes GEMM through the Metal 4 tensor API (`mpp::tensor_ops`, the neural
accelerators) by default — it enables that path only on M5/M6/A19/A20-class
devices, and its own source notes it is a wash or a loss on M4 and earlier.
That path is worth ~2.3–3× on prefill GEMM and nothing on decode (matvec).
Forced onto the same simdgroup-matrix instruction class Kipp uses
(`GGML_METAL_TENSOR_DISABLE=1`), llama.cpp's prefill drops to 1,261 / 1,022 /
608 — **Kipp leads its own instruction class on every scheme** (1.04× /
1.19× / 2.0×).

What follows from this: matching llama.cpp's default BF16/Q8_0 prefill on M5
requires an `mpp::tensor_ops` matmul path, not further simdgroup tuning —
that is the next roadmap item, behind the same runtime-probe-and-fall-back
ladder and oracle gates as the simdgroup kernels. *(Delivered in v0.0.6 —
next section.)*

### The tensor-ops path: BF16 prefill 2.80×, and where llama.cpp's design was wrong for this engine (2026-07-30)

v0.0.6 ships the `mpp::tensor_ops` (M5 neural-accelerator) matmul path for
BF16 layer projections, and it closes the instruction-class gap the previous
section documented:

| Qwen3-4B BF16, M5 Max, same-session interleaved A/B | simdgroup (`KIPP_METAL_TENSOR_DISABLE=1`) | tensor (default on M5-class) | ratio |
|---|---|---|---|
| Prefill @348 | 1,031.4 | **2,489.3** | 2.41× |
| Prefill @2,048 | 1,312.5 (MAD 1.2) | **3,682.1** (MAD 0.28) | **2.80×** |
| Prefill @12,800 | 691.8 | **968.7** | 1.40× |
| Decode | 67.6 | 68.1 | identical |
| Q8_0 control @2,048 | equal | equal | — |

The committed records (`4b-bf16-*.json`, 2026-07-30 session) read
3,679 / 2,500 at 2,048 / 348 tokens. That is parity with llama.cpp's own
tensor path (its cool-session 3,788 from the 2026-07-28 record; its quant
rows are unchanged — Kipp's quantized schemes still run the simdgroup
kernels, bit-frozen by fingerprint in both env states). The 12,800-token
ratio falling to 1.40× says what the next campaign is: with projections
~5× faster, **long-context prefill attention is now the dominant cost**.

Three design findings, two of them against the llama.cpp blueprint this
port started from:

- **Do not stage the weight tile.** llama.cpp's tensor kernel stages a
  64×32 weight tile through threadgroup memory (it must — its quant types
  dequantize there). Measured on Kipp's shapes, that staged design reaches
  only 0.96–2.9× the simdgroup kernels, while feeding both operands as
  plain device tensors reaches **4.9–6.2× (45–65 TFLOP/s)**, rotation-immune
  at 111–128 GB/s of weight streaming. This is the sixth consecutive loss
  for explicit threadgroup staging on this GPU; for pure BF16 the tensor
  units stream device memory faster than any hand-staging scheme.
- **Consume FP32 activations directly.** `matmul2d` accepts float sources,
  so the tensor path reads the round's FP32 working buffer instead of the
  staged-BF16 copy the simdgroup kernel needs. Float activations cost
  ~12–18% per GEMM, but they delete the entire `kipp_bf16_stage` pass
  (four dispatches per layer, ~90 ms per 2,048-token prefill) plus one
  activation rounding step — a net win end-to-end and strictly better
  numerics: pooled-vs-oracle NMSE **5.12e-07** against the simdgroup path's
  2.09e-06, matrix-vs-vector 2.78e-06 against 1.52e-05.
- **matmul2d is bitwise stable where it matters.** Its output measured
  bitwise invariant to the dispatch token count (26/32/96/512) and bitwise
  deterministic across repeated runs. That is what allows a wholesale
  kernel-class swap at the same ≥8-token threshold: the paged/pooled
  bitwise placement gates keep comparing a single class — and now cover the
  tensor path on M5-class hardware.

Because CI runners cannot run (or even compile) this path, it carries more
tripwires than any other kernel class: a runtime compile-and-probe ladder
behind a device allow-list, `KIPP_METAL_REQUIRE_TENSOR=1` turning silent
fallback into a load failure (it caught a real kernel bug during
development), a bench harness that refuses degraded-build numbers, an
affirmative "tensor path disabled (reason)" line asserted by CI, tensor
slots in the shader geometry probe, and an exact-integer operator test at
tolerance 0.0 on a ragged 72×133 shape — integer FP32 accumulation is
order-independent, so a wrong `matmul2d` descriptor cannot hide inside a
tolerance. Per-kernel-class tripwires (M5 Max): tensor `--prefill-metal`
fingerprint **794f1799c6a7326a** / pooled NMSE **5.12341885e-07**; the
frozen simdgroup values (**49e2ada96bce2804** / **2.08738076e-06**) remain
verifiable with `KIPP_METAL_TENSOR_DISABLE=1`, and both must be checked on
their own path.

### Panel-flash attention: prefill GQA on the tensor units, and the ~6× headroom it converts (2026-08-08)

v0.0.7 moves batched-prefill grouped-query attention off the simdgroup
flash-attention kernel and onto the M5 `mpp::tensor_ops` matmul path the
previous campaign built for projections, attacking the long-context cost
that section named. The win grows with context, tracking attention's share
of the wall:

| Qwen3-4B BF16, M5 Max, same-session A/B, cool-start matched | simdgroup (`KIPP_METAL_TENSOR_DISABLE=1`) | panel-flash (default on M5-class) | ratio |
|---|---|---|---|
| Prefill @2,048 | 4,069 | **4,530** | 1.11× |
| Prefill @6,400 | 2,740 | **3,451** | 1.26× |
| Prefill @12,800 | 1,754 | **2,410** | **1.37×** |
| Decode | 60.6 | 60.7 | 1.00× |
| Q8_0-KV control | equal | equal | streaming, unchanged |

Per 1,024-KV-position panel: a gather copies the round-visible KV range
through the block table into contiguous K + transposed-V panels; Q·Kᵀ runs
as one `matmul2d` per head into a float score panel; a panel-granular
online softmax reuses the streaming kernel's exact merge expressions (bf16
probabilities, float denominators); P·V runs as a second `matmul2d`; an
accumulate kernel folds each panel into the running output with the
cross-panel rescale. It is tensor-state + BF16-KV only — the simdgroup
flash-attention kernel stays the path for every other device, for
`KIPP_METAL_TENSOR_DISABLE`, and for Q8_0 KV, so its fingerprints stay
frozen. The committed records (`4b-bf16-prefill2k.json`,
`4b-bf16-prefill12k8.json`) read 4,443 tok/s at 2,048 tokens (sustained)
and 2,410 at 12,800 (cool-start); same-session, llama.cpp reads 4,100 and
1,561, so Kipp leads it 1.54× at 12,800 — llama.cpp's flash-attention is
simdgroup-only even on M5. CUDA was not revalidated this campaign.

A measurement note that shapes everything below: the M5 laptop chassis
**cannot hold steady state under sustained long-context prefill**. bf16
@2,048 tensor holds ~4,450 tok/s across a dozen back-to-back runs (MAD
0.66%), but @12,800 the same kernel decays run-over-run as the chassis
heat-soaks (2,304 → 1,098 within one session). This is thermal, not power
(battery 100%, adapter steady). So the long-context A/B is measured
**cool-start matched** — an idle-GPU cooldown before every run, both arms
interleaved — which samples both kernels at the same thermal state and
gives a fair, reproducible cold-prompt ratio. Absolute long-context
numbers are cold-prompt medians, labelled as such.

Why attention was worth its own kernel-class swap: at a 12,800-token
prefill the attention FLOPs are 2·2·(N²/2)·32 heads·128 dim·36 layers ≈
**48.3 TFLOP** — about half the projection FLOPs and growing linearly with
context. A two-point quadratic fit T(N)=aN+bN² across both kernel classes
agrees within 11% and attributes **~85% of the 12.8k wall to attention**.
The simdgroup FA kernel ran ~3 TFLOP/s effective (`--fa-bench-metal`) while
`matmul2d` at attention panel shapes sustains 20–25 TFLOP/s (probe E4) —
the ~6× headroom panel-flash exists to convert.

Design findings, several of them kill records:

- **Panels, not per-32-block matmul2d.** The tensor units need
  panel-scale operands, so attention is tiled at 1,024 KV positions rather
  than the projection tile. The gather's compulsory traffic is ≈ 23 GB per
  full 12.8k prefill ≈ 0.1 s — cheap against the 48.3 TFLOP it feeds.
- **The panel width is workspace-bounded, and that boundary is real.**
  `matmul2d` P·V at panel width 4,096 collapses to **6.5 TFLOP/s** (256
  threadgroups, occupancy) versus 19.5–25.3 at ≤2,048; the workspace budget
  already bounds the panel at 1,024, on the fast side of the cliff.
- **KV-head sharing across simdgroups lost −7.3%.** Dividing nominal KV
  traffic by 4 would have won ~4× if attention were traffic-bound; it did
  not, so prefill attention is **issue-bound, not traffic-bound** — which
  is why moving it to the tensor units, not to a cleverer memory schedule,
  is the win.
- **Round-width for attention is inert.** Total KV position-reads = N²/2
  regardless of how a round partitions them.
- **Two ~5%-ceiling levers deliberately skipped.** Wider-KV-tile and
  `staged_out`-deletion each cap at ~5% and were not worth re-baselining
  two kernel classes once panel-flash landed.
- **matmul2d truncates f32 sources to ~bf16 internally** (~2.5e-3 abs err
  on 128-dim dot products, measured) — below the softmax's own rounding, so
  fine for attention scores.
- **The zero-mean-V test-data trap, and why the operator test carries a DC
  offset.** Attention output is a probability-weighted average, so over
  1,500 zero-mean positions the reference cancels toward |v|/√N while the
  bf16-probability rounding both matrix-class kernels share stays absolute;
  a CPU simulation of the exact rounding class measured **8e-4 NMSE on
  zero-mean data with provably-correct algebra, and 3e-9 with a DC
  offset** — so the panel operator test carries the offset.

Per-kernel-class tripwires (M5 Max): tensor-state `--prefill-metal` bf16
fingerprint **721ea327c1facaed** / NMSE **7.46e-07**, pooled NMSE
**6.05480037e-07**; quant tensor-state q8_0 **0c251632a7a38a3d**, affine4
**52161133c69d0d76**; the frozen simdgroup values
(**49e2ada96bce2804** / **2.08738076e-06**) remain recoverable with
`KIPP_METAL_TENSOR_DISABLE=1`, and both classes must be checked on their
own path. Every tensor-state value is **tighter** than the class it
replaced, and paged placement invariance is bitwise through the panel
gather.

### Quantized projections on the tensor units: Q8_0/affine4 prefill 2.3× (2026-08-09)

v0.0.6 put BF16 projections on the Metal 4 `mpp::tensor_ops` path; v0.0.8
does the same for the quantized schemes, the last prefill path still on the
simdgroup kernels.

| Qwen3-4B, M5 Max, same-session A/B (both tensor-state) | simdgroup (`KIPP_METAL_TENSOR_DISABLE=1`) | quant-tensor (default) | ratio |
|---|---|---|---|
| Q8_0 prefill @2,048 | 1,247 | **2,875** | **2.31×** |
| affine4 prefill @2,048 | 1,250 | **2,857** | 2.29× |
| decode (both schemes) | — | — | unchanged (matvec) |

Committed steady-state records (tensor path): Q8_0 **2,882 / 1,940** at
2,048 / 348 tokens, affine4 **2,868 / 1,960** (MAD ≤0.3% at 2,048).

Unlike the BF16 tensor kernel — which feeds `matmul2d` bf16 weights straight
from device memory and stages nothing — a quant weight must be dequantized
first. `kipp_matmul_q8_0_tensor` and `kipp_matmul_affine4_tensor` reuse the
simdgroup kernels' cooperative dequant prologue (`kipp_fp16_bytes`; Q8_0
`d*q`, affine4 `scale*q+bias`) to expand each K-block into a threadgroup
bf16 tile (64 rows × 32), then run one `matmul2d` per block. The output tile
geometry (64 × 128, K-chunk 32), transposed grid, and geometry probe are
shared verbatim with the BF16 tensor kernel — nothing else moved.

Design notes:
- **The threadgroup-source `matmul2d` was the one open risk, and it holds.**
  Every prior tensor kernel feeds `matmul2d` from device memory; the quant
  design feeds a dequantized *threadgroup* tile. A Phase-0 spike confirmed
  M5's `matmul2d` accepts a threadgroup-address-space weight operand and
  returns bitwise-exact results; `kipp_tensor_probe_bf16` was extended to
  exercise it so a device that can't build it demotes cleanly to the
  simdgroup path rather than hard-failing.
- **Selection is a pure function of token count** (quant-tensor at ≥ one
  token tile, simdgroup below), so the paged/pooled bitwise gates keep
  comparing a single kernel class. Q8_0-KV and non-M5 / `TENSOR_DISABLE`
  stay on the simdgroup kernels, whose fingerprints are frozen.
- **Decode is untouched** — the tensor path is prefill-only.
- **No llama.cpp head-to-head this release.** llama-bench's quantized
  prefill is thermally unstable on this laptop chassis: in one session the
  same Q8_0 `-p 2048` measured 2,102 tok/s back-to-back and 4,336 cold (a
  2× swing; the latter implausibly above llama.cpp's own bf16 4,100),
  because llama-bench has no steady-state protocol. A fair same-session
  quant comparison could not be taken, so the result above is Kipp-internal
  (tensor vs the simdgroup kernel it replaces), steady-state, same-session.

Per-kernel-class tripwires (M5 Max): tensor-state `--prefill-metal` q8_0
fingerprint **33b605f4f5a05243** (nmse vs CPU oracle 9.2e-07), affine4
**0953f2975a2692d5** (2.2e-06); the frozen simdgroup values
(**7421c99f0af71365** q8_0, **0709c0eb78978e61** affine4) remain recoverable
with `KIPP_METAL_TENSOR_DISABLE=1`, and both classes must be checked on their
own path. bf16 (721ea327c1facaed) and pooled (6.05480037e-07) reprint
unchanged. CUDA was not revalidated this campaign.

### Decode is weight-bandwidth-bound: the dequant-vectorization kill (2026-08-12)

With every prefill path on the tensor units (v0.0.6–v0.0.8), decode is the
last axis where Kipp trails llama.cpp (~3–6% BF16/Q8_0, ~17% 4-bit). Decode
generates one token at a time and **streams every model weight per token**, so
it is memory-bandwidth-bound. Two independent signals confirm the floor:

- **The model-size sweep is a clean inverse-with-weight-bytes law** — BF16
  decode 0.6B **269**, 4B **60.7**, 8B **33.5** tok/s. Halving the streamed
  bytes (Q8_0 vs BF16) buys ~1.6×, not the ~2× a compute-bound kernel would
  give — decode tracks bytes, not FLOPs.
- **A roofline check**: 4B BF16 at ~60 tok/s streams ~8 GB/token × 60 ≈
  480 GB/s, a large fraction of the M5 Max's unified-memory bandwidth. The
  BF16 matvec already reads each weight once via `ushort4`/`float4` loads with
  thousands of threadgroups resident — there is no occupancy or vectorization
  slack to reclaim.

**The one lever with apparent headroom — measured, and killed.** The Q8_0 and
affine4 decode matvec *inner loops* were scalar (`float(qs[j])*in[j]` per
weight, per-byte nibble unpack) while BF16 was vectorized, and Q8_0 decoding
only ~1.6× faster than BF16 despite moving ~half the bytes suggested the quant
kernels were partly ALU-bound on dequant. We vectorized both inner loops
(`float4` activation loads + FMA, mirroring the BF16 kernel; correctness-gated:
operator suite 38/0, oracle NMSE 9.2e-07/2.2e-06, tensor fingerprints frozen)
and ran a same-session A/B (vectorized vs scalar). A naive sequential A/B read
Q8_0 +2.3%, but the **BF16 control — an untouched, identical kernel — read
+2.8% between the two binaries**, i.e. the +2.3% was measurement drift, not a
win. An interleaved, drift-cancelling A/B settled it: **drift-corrected Q8_0
−2.6%** (vectorized is if anything *slightly slower* — extra register pressure,
no fewer bytes) and affine4 neutral. The change was reverted. Quant decode is
byte-bound, not ALU-bound; the dequant loop is not the bottleneck.

Conclusion: the BF16/Q8_0 decode gap vs llama.cpp is a **weight-bandwidth
floor**, not an implementation defect — both engines stream the same bytes and
land within a few percent. The larger 4-bit gap is **partly the format**:
Kipp's affine4 carries a per-group scale *and* bias (higher quality), which
Q4_0's scale-only zero-point (`−8`) avoids, costing an extra term per block;
closing it would mean a new scale-only 4-bit scheme, not a kernel tweak.

Measurement note: decode is immune to GPU-clock DVFS and other-GPU
contention, but **not to Low Power Mode**, which throttles unified-memory
bandwidth SoC-wide — under LPM a BF16 decode control read 12.5 vs 59.7 (5×
low). Every decode A/B here carries an untouched BF16 control as the
validity gate.

## Optimized Metal kernels on Apple M5 (v0.0.1)

Measured on 2026-07-13 with Kipp v0.0.1's batched Metal path: one serial compute
encoder per command buffer, up to 32 prefill tokens per batch,
simdgroup-matrix BF16 projections for batched prefill (16-token tiles, FP32
accumulation, with a vector-kernel fallback on devices without bfloat
simdgroup matrices), a simdgroup matvec for single-token decode, and split-K
streaming online-softmax attention (eight partial softmaxes per head merged
in threadgroup memory). Same hardware, model artifact, build flags, and
harness as the baseline below; one discarded warm-up and five measured
subprocesses per configuration.

Commands:

```bash
python3 tools/bench.py --warmup 1 --runs 5 --decode 8
python3 tools/bench.py --warmup 1 --runs 5 --decode 32 --prompt "<265-token prompt>"
```

Results (median, with median absolute deviation):

- 3-token prompt, greedy decode of 8:
  prefill **9.375 tokens/s** (MAD 0.939, range 8.436–12.557);
  decode **13.776 tokens/s** (MAD 0.063, range 13.259–13.985)
- 265-token prompt, greedy decode of 32:
  prefill **147.769 tokens/s** (MAD 2.219, range 123.937–149.988);
  decode **13.072 tokens/s** (MAD 0.802, range 12.133–13.969)
- Single spot check at a 1,981-token prompt: prefill 136.6 tokens/s and
  decode 13.0 tokens/s (decode was 11.0 tokens/s before split-K
  attention — long contexts are where the partitioned kernel pays).
- Peak process RSS: **~40 MB median** in both configurations (see the RSS
  caveat below; mapped GPU-touched weights are not charged to the process)

Short-prompt prefill is dominated by fixed per-evaluation cost; the
265-token configuration reflects the batched matrix-kernel prefill path.
Decode remains memory-bandwidth-bound: at ~13.8 tokens/s the engine streams
roughly 115 GB/s of BF16 weights through the M5. Prefill at this revision was
limited by per-layer KV re-reads in attention rather than by the projection
matmuls. **That attribution is specific to the 32-token round this section
measured and no longer describes the engine** -- see "Prefill round shape"
below for the current model and the measurement that corrected it.

Batched decoding through the server (48 sampled tokens per sequence,
temperature 0.8, single measurements). Multi-choice requests (`n`) and
independent concurrent clients batch the same way — each step reads the
weights once for every active sequence:

- one request, `n` = 1/2/4/8: 12.7 / 18.9 / 29.7 / 41.4 tokens/s aggregate
- concurrent clients (distinct prompts and seeds), 1/2/4/8 connections:
  12.7 / 20.3 / 29.1 / 39.7 tokens/s aggregate

Aggregate throughput approaches compute limits instead of scaling the
bandwidth cost with the number of sequences.

## Phase 3 Apple M5 baseline (superseded)

Measured on 2026-07-10 with the original one-token-per-command-buffer
kernels, on:

- Apple M5, 10-core GPU, 24 GiB unified memory
- macOS 26.4, Apple clang 21.0.0
- dirty development tree at `0eac2059cfa138c8e56ce2664c9213c9e4340261`
- `Qwen/Qwen3-4B-Base` revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`
- 8,049,127,680-byte BF16 Kipp GGUF-v3 artifact
- `-std=c11 -O2 -Wall -Wextra -Wpedantic -Werror`
- prompt `Hi 世界` (3 tokens), greedy decode of 8 tokens, batch 1,
  concurrency 1
- one discarded warm-up and five measured subprocesses

Command:

```bash
python3 tools/bench.py --warmup 1 --runs 5 --decode 8
```

Results:

- Prefill: **6.405 tokens/s median**, 0.027 tokens/s median absolute
  deviation (range 6.379–6.719)
- Decode: **8.193 tokens/s median**, 0.004 tokens/s median absolute
  deviation (range 8.189–8.207)
- Peak process RSS: **47,775,744 bytes median** (45.6 MiB), 32,768-byte
  median absolute deviation

Peak process RSS is not total unified-memory residency. Metal accesses the
7.50 GiB model through a shared no-copy mmap, and macOS does not charge those
GPU-touched mapped pages to the CLI's reported peak RSS in the same way as
ordinary process allocations. The artifact size and cache layout therefore
remain necessary context; the RSS number must not be presented as the total
memory needed to run the model.

This was a readable fixed-shape baseline, not an optimized performance claim.
At that revision the implementation intentionally excluded fusion,
quantization, batching, and private weight copies; later results above
supersede those constraints.
