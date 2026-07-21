# Reproducing Kipp's results

This document lists the exact commands to build Kipp, run its correctness
gates, and regenerate the benchmark numbers and the paper. Every claim in
`paper/` and `docs/BENCHMARKS.md` is produced by one of these commands.

## System requirements

- **CPU / any host**: a C11 compiler (`clang` or `gcc`), `make`, Python 3.12+.
- **Apple Metal**: macOS with the Xcode command-line tools (Apple Silicon).
- **NVIDIA CUDA**: `nvcc` (CUDA 12.x); build with `make cuda-generic`.
- **Model weights**: pinned Qwen3 checkpoints, downloaded and converted with
  `tools/download_model.sh` + `tools/convert_to_gguf.py`. Weights are **not**
  committed (see `.gitignore`); only the `qwen3-4b-base` golden test vectors
  are, under `tests/test-vectors/`.
- **Paper**: any LaTeX toolchain. We compile with `tectonic` (single binary):
  `cd paper && tectonic main.tex`.

## Build

```bash
make cpu            # scalar CPU reference/oracle  -> build/kipp
make metal          # Apple Metal backend          -> build/kipp-metal
make server-metal   # OpenAI-compatible server      -> build/kipp-server-metal
make cuda-generic   # NVIDIA CUDA (on an NVIDIA host)
```

## Correctness gates (the paper's contribution)

Hermetic unit tests and the address/UB sanitizer run with no model:

```bash
make test            # unit tests
make test-sanitize   # ASan + UBSan
```

Model gates run the built test binary directly against the pinned vectors
(do **not** use `make test-model`'s vectors->convert chain for a quick gate —
it rewrites the multi-GB GGUF). With a converted `qwen3-4b-base` BF16 GGUF at
`$M` and vectors at `$V`:

```bash
build/kipp_test        --model      $M $V   # CPU logits == pinned reference (bitwise)
build/kipp_test        --paged-cpu  $M $V   # paged KV == contiguous, scrambled block table (bitwise)
build/kipp_test_metal  --paged-metal $M $V  # same, Metal backend
build/kipp_test_metal  --phase3-metal $M $V # Metal vs CPU oracle (argmax exact, NMSE <= 1e-4)
build/kipp_test_metal  --metal-operators    # per-kernel operator tests
make test-server                            # OpenAI Completions + Chat Completions
```

Convenience targets that wire the same gates to the default checkpoint:
`make test-paged-cpu`, `make test-multilogit`, `make test-paged-metal`,
`make test-metal`.

## Mutation study (the paper's fault-detection matrix)

`make build/kipp_test_fault` builds the CPU oracle with seeded KV-addressing
bugs selected by the `KIPP_FAULT` environment variable (1 read-block,
2 read-slot, 3 rollover, 4 swap-kv; 0/unset = no fault). For each fault, run
the identity-mapped tolerance gate and the scramble gate on the same
multi-block sequence:

```bash
KIPP_FAULT=<n> build/kipp_test_fault --fault-reference $M $V  # tolerance + argmax
KIPP_FAULT=<n> build/kipp_test_fault --paged-cpu       $M $V  # scramble, bitwise
```

The fault build is research tooling only; production targets never define
`KIPP_FAULT_INJECT`.

## Benchmarks

The harness (`tools/bench.py`) runs isolated subprocesses with a discarded
warm-up, reports median + MAD + min/max, and separates prefill from decode
timers. Results land in `bench/results/` as JSON.

**Measure in GPU steady state.** Apple-silicon GPU clocks are demand-scaled:
short bursts from an idle GPU measure far below the sustained clock (2-5x,
with 3x run-to-run swings), other GPU clients depress compute-bound prefill
while barely touching bandwidth-bound decode, and a cool chassis briefly
boosts above the sustained level. Run benches on an otherwise idle machine,
back-to-back in one session, after a few minutes of continuous GPU work;
check the recorded MAD is well under 1% before trusting a file (see
`bench/README.md`, "Measurement protocol").

```bash
# Decode throughput by weight scheme (Apple M5 Metal), Qwen3-4B:
python3 tools/bench.py --backend metal --model models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf \
  --prompt "The capital of France is" --decode 64 --warmup 1 --runs 5 \
  --output bench/results/4b-bf16-decode.json
# (repeat for -q8_0 and -affine4_gs32 artifacts)

# Prefill throughput (long prompt):
python3 tools/bench.py --backend metal --model <gguf> --prompt "<~300-word prompt>" \
  --decode 8 --warmup 1 --runs 5 --output bench/results/4b-<scheme>-prefill.json

# Speculative decoding A/B (run both in one session for coherent baselines):
python3 bench/spec_bench.py --model models/qwen3-4b-base/kipp-qwen3-4b-base-q8_0.gguf \
  --decode 256 --runs 3 --gate off --output bench/results/spec.json
python3 bench/spec_bench.py --model models/qwen3-4b-base/kipp-qwen3-4b-base-q8_0.gguf \
  --decode 256 --runs 3 --gate on --output bench/results/spec-gated.json

# Serving: batch scaling, cross-request prefix reuse, open-loop load:
python3 bench/server_bench.py --model models/qwen3-4b-base/kipp-qwen3-4b-base-q8_0.gguf \
  --output bench/results/server-batch.json
python3 bench/prefix_bench.py --model models/qwen3-4b-base/kipp-qwen3-4b-base-q8_0.gguf \
  --output bench/results/prefix-reuse.json
python3 bench/load_bench.py --model models/qwen3-4b-base/kipp-qwen3-4b-base-q8_0.gguf \
  --rates 0.5,1,2,4 --requests 32 --output bench/results/load.json

# Quantization quality (wikitext-2 perplexity; CPU-vs-Metal verify included):
python3 bench/ppl_bench.py --output bench/results/ppl-4b.json
```

CUDA numbers use the same harness with `--backend cuda --binary build/kipp-cuda-generic`
on an NVIDIA host; see `tools/ops/verda_cuda_gate.sh` for the ephemeral-cloud
workflow used to produce them.

## The paper

```bash
cd paper && tectonic main.tex   # -> paper/main.pdf
```

The numbers in `paper/main.tex` are LaTeX macros generated from committed
`bench/results/*.json` by `make paper-data` (into `paper/generated/` and
`paper/data/`), each annotated with its source file and field.
`make paper-check` fails if any consumed result file is missing, untracked,
modified, or if the generated files drift from regeneration — run it after
re-running the harness on new hardware.
