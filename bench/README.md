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

Every result file records the engine commit, binary, and hardware. A number
quoted in project documentation must trace to a committed `results/*.json`,
and a hardware change must be visible in that file rather than inferred.

The recorded `dirty` flag ignores `bench/results/` itself so a run whose only
working-tree changes are freshly written results records `dirty: false`.
Any modification outside that directory still marks the run dirty.

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

## Published results

[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) is the single human-readable
record of reference results. The JSON files in `results/` are the underlying
machine-readable evidence; this document intentionally describes only the
harness and measurement protocol.

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
identical to the reference so numbers are comparable. See
[`docs/REPRODUCING.md`](../docs/REPRODUCING.md) for the complete build and
gate sequence.
