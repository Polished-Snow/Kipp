# Reproducing Kipp

This guide covers builds, correctness gates, and benchmark reproduction.
Benchmark claims in `BENCHMARKS.md` trace to committed JSON records under
`bench/results/`.

## Requirements

- **CPU:** a C11 compiler (`clang` or `gcc`), `make`, and Python 3.12+
- **Apple Metal:** Apple Silicon with the Xcode command-line tools
- **NVIDIA CUDA:** `nvcc` from CUDA 12.x or newer
- **Tooling:** [`uv`](https://docs.astral.sh/uv/) for the locked Python
  environment
- **Model artifacts:** converted pinned Qwen3 checkpoints under `models/`

Model weights are downloaded by `tools/download_model.sh` and converted by
`tools/convert_to_gguf.py`. They are intentionally not committed. The pinned
`qwen3-4b-base` golden vectors live under `tests/test-vectors/`.

## Build

```bash
make cpu             # scalar CPU reference CLI
make server          # CPU server
make metal           # Apple Metal CLI
make server-metal    # Apple Metal server
make cuda-generic    # portable NVIDIA CUDA CLI
make server-cuda     # portable NVIDIA CUDA server
make cuda-spark      # NVIDIA DGX Spark (sm_121)
```

Backend selection is explicit. A backend initialization or compilation
failure is fatal; Kipp never silently falls back to CPU.

## Model-free gates

```bash
make test            # native units, tooling tests, docs drift
make test-sanitize   # ASan + UBSan
make docs-check      # generated HTML matches canonical Markdown
```

## Model-backed CPU and Metal gates

For a quick local gate, run the built test binaries against an existing GGUF
and vector directory. Avoid the `vectors` target unless you intend to
download, convert, and regenerate the multi-gigabyte artifact.

```bash
M=models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf
V=tests/test-vectors/qwen3-4b-base

build/kipp_test --model "$M" "$V"
build/kipp_test --phase2-model "$M" "$V"
build/kipp_test --multilogit "$M" "$V"
build/kipp_test --paged-cpu "$M" "$V"
build/kipp_test --pooled-cpu "$M" "$V"

build/kipp_test_metal --metal-operators
build/kipp_test_metal --phase3-metal "$M" "$V"
build/kipp_test_metal --multilogit-metal "$M" "$V"
build/kipp_test_metal --paged-metal "$M" "$V"
build/kipp_test_metal --pooled-metal "$M" "$V"
build/kipp_test_metal --longctx-metal "$M" "$V"
```

The convenience targets `test-cpu-model` and `test-metal` run the complete
CPU and Metal suites for `CHECKPOINT` (default `qwen3-4b-base`).
Additional integration gates:

```bash
make test-server
make test-chat
make test-draft-spec
```

CPU reference logits require the vector-specific bound and exact argmax.
Cached CPU decode must match recomputation within `1e-6`; Metal must preserve
argmax and remain within `1e-4` NMSE of the CPU oracle.

## CUDA gates

On an NVIDIA host:

```bash
make cuda-generic test-cuda
```

`tools/ops/verda_cuda_gate.sh` provisions a disposable Verda instance,
runs the registered checkpoint gates, and removes the VM and volume. It
requires explicit cost acknowledgement. Parse a captured log with
`tools/ops/collect_cuda_gates.py` to produce a self-describing result file.

## Mutation gate

The optional fault-injection binary validates that the identity-layout
tolerance gate and adversarial block-table gate catch different classes of
KV addressing defects:

```bash
make build/kipp_test_fault
KIPP_FAULT=1 build/kipp_test_fault --fault-reference "$M" "$V"
KIPP_FAULT=1 build/kipp_test_fault --paged-cpu "$M" "$V"
```

Values 1–4 select read-block, read-slot, rollover, and K/V-swap faults.
Production targets never define `KIPP_FAULT_INJECT`.

## Benchmarks

All benchmark harnesses record commit, dirty state, binary, compiler,
hardware, and model manifest metadata. Follow the sustained GPU steady-state
protocol in [`../bench/README.md`](../bench/README.md).

```bash
python3 tools/bench.py \
  --backend metal \
  --model models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf \
  --prompt "The capital of France is" \
  --decode 64 --warmup 1 --runs 5 \
  --output bench/results/4b-bf16-decode.json

python3 bench/spec_bench.py --model <gguf> --decode 256 --runs 3 --gate both
python3 bench/server_bench.py --model <gguf>
python3 bench/prefix_bench.py --model <gguf>
python3 bench/load_bench.py --model <gguf>
python3 bench/ppl_bench.py --output bench/results/ppl-4b.json
```

CUDA uses the same subprocess harness with `--backend cuda --binary
build/kipp-cuda-generic`. Only commit results from a clean, stable run with
tight dispersion.
