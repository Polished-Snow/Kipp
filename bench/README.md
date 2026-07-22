# Kipp benchmarks

Reproducible throughput and speculative-decoding measurements. Numbers must
be reported per the policy in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md):
median + dispersion, separate prefill/decode timers, matched weights/context/
hardware, and the exact commands.

## Harness

- `../tools/bench.py` — subprocess harness: discarded warm-up, median + MAD +
  min/max over N runs, separate prefill/decode timers, peak RSS (and CUDA
  device bytes). Writes one JSON per configuration into `results/`.
- `spec_bench.py` — prompt-lookup speculative decoding: acceptance rate `α`,
  block efficiency, drafting duty cycle (adaptive gate), and wall-clock
  speedup vs. non-speculative greedy across a fixed workload set. Greedy
  speculative output is token-identical to greedy decode (gated), so only
  the wall clock differs.
- `server_bench.py` — aggregate server throughput vs. batch size: one
  request with `n` choices, and `n` concurrent connections.
- `load_bench.py` — open-loop Poisson serving load: streamed TTFT/TPOT
  percentiles and goodput under an SLO, per arrival rate.
- `prefix_bench.py` — cross-request prefix reuse: cold-vs-warm prefill time
  for the same long prompt against the pooled server.
- `ppl_bench.py` — wikitext-2 perplexity per weight scheme via the CLI's
  `--ppl` mode, with a CPU-vs-Metal agreement check.
- `_provenance.py` — shared self-description module. Every script records
  the same engine block (commit, dirty, binary, compiler, build flags), the
  full hardware block (chip, GPU cores, memory), and the model's pinned
  repository/revision read from its `*.gguf.manifest.json` — never
  hardcoded.

Every result file records the engine commit, binary, and hardware — **a
number quoted anywhere (docs, paper) must trace to a committed
`results/*.json`**, and a hardware change must be visible in that file, not
inferred.

The recorded `dirty` flag ignores `bench/results/` itself and the paper
artifacts derived from it (`paper/generated/`, `paper/data/`): a run whose
only working-tree changes are freshly written results or stale derived
paper data records `dirty: false`. Any modification outside those paths
still marks the run dirty.

## Measurement protocol: GPU steady state

Apple-silicon GPU clocks are aggressively demand-scaled. A short dispatch
burst (a one-second prefill) issued from an idle GPU runs well below the
sustained clock and can measure 2–5x low with run-to-run swings of 3x;
the same measurement taken back-to-back with other GPU work is stable to a
fraction of a percent. Other GPU clients (browsers, music players, IDE
rendering) depress compute-bound kernels — quantized prefill hardest,
because in-kernel dequantization is ALU-heavy — while leaving
bandwidth-bound decode almost untouched. A cool chassis additionally
allows a short-lived boost regime above the sustained level.

All committed numbers are therefore **sustained steady-state** figures:
the machine otherwise idle, measurements taken back-to-back in one session
after a multi-minute GPU warm-up workload, external baselines (llama.cpp)
measured in the same session under the same conditions. Result files with
tight dispersion (MAD well under 1%) indicate the protocol held; a wide
MAD means the session was contaminated and should be re-run.

One more tripwire, learned the hard way: the Metal bridge falls back to
vector kernels when its matrix-pipeline compile fails, with only a stderr
warning. In July 2026 a reserved MSL keyword silently disabled every MMA
kernel for two days — all correctness gates still passed, and a full
benchmark campaign recorded the degraded build. `tools/bench.py` now
refuses to record Metal numbers when the fallback warning is present.

## Results (`results/*.json`)

Committed per configuration. Naming: `<model>-<scheme>-<decode|prefill>.json`,
`ctx-<tokens>.json` (context scaling), `spec.json` / `spec-gated.json`
(speculation without/with the adaptive gate), `server-batch.json`,
`llamacpp-*.json` (external A/B), `faults.json` (mutation study), and
`cuda-h100-gates.json` for cloud CUDA gate runs.

## Reference numbers — Apple M5 Max (Qwen3-4B, Metal, greedy, median of 5)

Measured on a MacBook Pro M5 Max (40-core GPU, 128 GB unified memory) under
the steady-state protocol above. Numbers from earlier releases were measured
on a base M5 (10-core GPU, 24 GB) and are roughly 4× lower across the board;
see each result file's recorded hardware.

| Weight scheme | Decode tok/s | Prefill tok/s (348 / 2,048 tokens) | Wikitext-2 PPL |
|---|---|---|---|
| BF16 | 60.7 | 528 / 481 | 7.731 |
| Q8_0 (8-bit) | 97.9 | 488 / 441 | 7.733 |
| affine4 (4-bit) | 130 | 509 / 466 | 8.171 |

Peak process RSS ≈ 41 MB (weights are mmap-ed; RSS understates GPU-touched
residency). Decode is bandwidth-bound, so quantization scales it with the
weight-byte reduction at unmeasurable quality cost for Q8_0 (+0.02% PPL)
and a Q4-class cost for affine4 (+5.7% PPL). Prefill runs on the
simdgroup-matrix (MMA) kernels — matmuls and the tiled flash-attention
kernel — so quantized prefill stays near BF16 parity; the O(n²) attention
tail brings Q8_0 prefill from ~485 tok/s at short context down to
177 tok/s at 12.8k tokens.

Speculative decoding (Q8_0, 256-token decode, paired-baseline A/B via
`--gate both`, median of 3):

| Workload | α | Ungated | Gated | Duty |
|---|---|---|---|---|
| Repetitive / copy-heavy | 1.00 | 2.10× | 2.27× | 1.00 |
| Code | 0.19 | 0.73× | 1.24× | 0.06 |
| Grounded QA | 0.00 | 0.27× | 0.90× | 0.04 |
| Open-ended | 0.09 | 0.62× | 0.84× | 0.10 |

Prompt-lookup only pays at high acceptance; the adaptive gate suspends
drafting when a short acceptance EMA collapses and probes periodically, which
holds every low-acceptance workload at or near parity (0.84× floor, and
above parity on code) while keeping the copy-heavy win. Speedups are
smaller than earlier drafts because the sampling fast path made the plain
decode baseline itself faster. Gated speculative output remains
token-identical to greedy.

## Reproduce

```bash
python3 tools/bench.py --backend metal --model <gguf> \
  --prompt "The capital of France is" --decode 64 --warmup 1 --runs 5 \
  --output bench/results/4b-<scheme>-decode.json
python3 bench/spec_bench.py --model <q8_0 gguf> --decode 256 --runs 3 --gate both
python3 bench/server_bench.py --model <gguf>
python3 bench/ppl_bench.py --output bench/results/ppl-4b.json
```

## Submitting results for new hardware

Run the commands above on your hardware and open a PR adding the `results/`
JSON files. Keep the model revision, quantization, context, and build flags
identical to the reference so numbers are comparable.
