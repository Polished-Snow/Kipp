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

- Decode tok/s (64 tokens, median of 5): BF16 **60.7**, Q8_0 **97.9**,
  affine4 **130**.
- Wikitext-2 perplexity (full test set, 2,048-token windows): BF16
  **7.731**, Q8_0 **7.733** (+0.02%), affine4 **8.171** (+5.7%) — Q8_0 is
  effectively lossless; affine4 is Q4-class.
- Prefill tok/s (348-token / 2,048-token prompt): BF16 **528 / 481**,
  Q8_0 **488 / 441**, affine4 **509 / 466**. The simdgroup-matrix kernels
  dequantize once per 16-token tile, so quantized prefill stays near BF16
  parity; the O(n²) attention tail brings Q8_0 prefill from ~485 tok/s at
  short context to **177 tok/s** at 12,800 tokens.
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
- llama.cpp A/B (same weights/host, pinned commit, same session,
  `llamacpp-qwen3-4b.json`): Kipp decode 60.7/97.9 vs llama.cpp 35.0/56.5
  (BF16/Q8_0, ~1.7× in Kipp's favor); llama.cpp Q4_0 decodes 89.1 vs
  affine4's 130 (schemes differ). llama.cpp prefill is ~4.5× faster at a
  matched 2,048-token prompt (2,174 vs 481 BF16).
- CUDA revalidation (`cuda-h100-gates.json`): all four default checkpoints
  pass `--model` and `--phase4-cuda` on an ephemeral NVIDIA H100 80GB
  (worst observed NMSE 5.9e-7 against the CPU oracle).

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
roughly 115 GB/s of BF16 weights through the M5. Long-context prefill is
now limited by per-layer KV re-reads in attention rather than by the
projection matmuls.

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
