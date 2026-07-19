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
  block efficiency (accepted tokens per target forward), and wall-clock
  speedup vs. non-speculative greedy, across a fixed workload set (repetitive,
  code, grounded, chat). Greedy speculative output is token-identical to
  greedy decode, so only the wall clock differs.

## Results (`results/*.json`)

Committed per configuration. Naming: `<model>-<scheme>-<decode|prefill>.json`
and `cuda-a100-...` for cloud CUDA runs. Each file records the engine commit,
hardware, model revision, and full run distribution.

## Reference numbers — Apple M5 (Qwen3-4B, Metal, greedy, median of 5)

| Weight scheme | Decode tok/s | Prefill tok/s |
|---|---|---|
| BF16 | 12.6 | 143 |
| Q8_0 (8-bit) | 20.7 | 35 |
| Affine (4-bit) | 26.9 | 32 |

Peak process RSS ≈ 41 MB (weights are mmap-ed). Decode is bandwidth-bound, so
quantization scales it with the weight-byte reduction; prefill is
compute-bound and quantization currently *slows* it (the quantized prefill
uses a vector kernel, not the simdgroup-matrix path).

Speculative decoding (Q8_0, controlled A/B, median of 3):

| Workload | α | Block eff. | Speedup |
|---|---|---|---|
| Repetitive / copy-heavy | 1.00 | 4.00 | 1.93× |
| Code | 0.17 | 1.39 | 0.88× |
| Open-ended / grounded | 0.04 | 1.10 | 0.55× |

Prompt-lookup helps only at high acceptance; below that the multi-token verify
(which does not amortize weight streaming at small draft batch on this
bandwidth-bound engine) makes it a net loss.

## Reproduce

```bash
python3 tools/bench.py --backend metal --model <gguf> \
  --prompt "The capital of France is" --decode 64 --warmup 1 --runs 5 \
  --output bench/results/4b-<scheme>-decode.json
python3 bench/spec_bench.py --model <q8_0 gguf> --decode 64 --runs 3
```

## Submitting results for new hardware

Run the commands above on your hardware and open a PR adding the `results/`
JSON files. Keep the model revision, quantization, context, and build flags
identical to the reference so numbers are comparable.
