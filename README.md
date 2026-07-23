<p align="center">
  <a href="https://polished-snow.github.io/Kipp/">
    <img src="docs/assets/kipp-logo.svg" alt="Kipp" width="420">
  </a>
</p>

<p align="center">
  <a href="https://polished-snow.github.io/Kipp/">
    <img src="https://img.shields.io/badge/documentation-github%20pages-6366F1?style=for-the-badge" alt="Documentation">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/Polished-Snow/Kipp?style=for-the-badge&color=22C55E" alt="MIT License">
  </a>
  <a href="https://polished-snow.github.io/Kipp/architecture.html">
    <img src="https://img.shields.io/badge/core-C11-00599C?style=for-the-badge&logo=c&logoColor=white" alt="C11 core">
  </a>
</p>

<p align="center">
  <a href="https://x.com/polished_snow">
    <img src="https://img.shields.io/badge/follow-%40polished__snow-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X">
  </a>
  <a href="https://www.linkedin.com/company/plsn">
    <img src="https://img.shields.io/badge/follow-Polished%20Snow-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="Follow on LinkedIn">
  </a>
</p>

## Status

Kipp v0.0.3 targets the pinned **Qwen3 dense family** (0.6B–32B, base and
instruct) through a strict compiled-in checkpoint registry.

- **Backends:** scalar CPU oracle, Metal on Apple M5-class hardware, and CUDA
  revalidated on H100 for the current default checkpoints. Larger 14B/32B
  CUDA gates were previously validated on A100.
- **Weights:** BF16, near-lossless Q8_0, and Q4-class affine 4-bit projection
  weights. Quantization improves decode by roughly 1.6–2.2× on the measured
  4B setup while Metal prefill remains near BF16 parity.
- **Serving:** OpenAI Completions and Chat Completions, SSE, 32-way batched
  decode, full sampling, latency metrics, and cross-request KV prefix reuse
  through a content-addressed CPU/Metal block pool.
- **CLI:** greedy or sampled generation, multi-turn chat, perplexity scoring,
  prompt-lookup speculation, and draft-model speculation.
- **Current work:** split-K long-context Metal decode and all-lane prefill
  softmax are implemented and awaiting the next release.

## What is Kipp?

Kipp is a small, **hand-written Qwen3 inference engine** for local hardware.
Most engines try to run everything; Kipp deliberately runs one pinned model
family and aims to run it well enough that you can read the engine in an
afternoon.

- **Readable by design** — follow one token from embeddings to logits through
  the linear, model-specific reference pass in `src/kipp.c`.
- **Native all the way down** — C11 core, a narrow Objective-C Metal bridge,
  and isolated CUDA kernels for NVIDIA.
- **No frameworks in the hot path** — Python and shell are tooling only;
  PyTorch, JAX, and similar frameworks never enter inference.
- **One model family, deeply** — closer to `whisper.cpp` and `antirez/ds4`
  than a generic runner such as llama.cpp or vLLM.
- **Correctness first** — every GPU backend must match the CPU oracle on real
  hardware before it is called working.
- **mmap-backed loading** — weights are mapped instead of eagerly copied into
  RAM wherever the backend permits.
- **Registry, not runner** — every supported checkpoint is pinned to one
  revision in `src/kipp_checkpoints.h`; anything else is rejected at load.

## Quickstart

```bash
git clone https://github.com/Polished-Snow/Kipp.git
cd Kipp
make test         # warning-clean native unit tests; no model download
make cpu          # builds the session-backed CPU CLI
make metal        # builds the Apple Metal CLI
make server-metal # builds the completions/chat/metrics server on Metal
```

The real-model correctness gates are explicit because they download and
convert gigabytes of pinned BF16 weights. `CHECKPOINT` selects any id from
the supported registry (default `qwen3-4b-base`):

```bash
make test-cpu-model                   # complete CPU model suite
make test-metal
make test-server
make CHECKPOINT=qwen3-0.6b-base test-model   # any registry checkpoint
```

Run a short Metal decode — greedy by default, or sampled with a seed:

```bash
build/kipp-metal --backend metal \
  --model models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf \
  --prompt "Hi 世界" --decode 8

build/kipp-metal --backend metal \
  --model models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf \
  --prompt "Once upon a time" --decode 64 \
  --temperature 0.8 --top-p 0.95 --seed 7
```

Or serve the OpenAI Completions subset on loopback:

```bash
build/kipp-server-metal --backend metal --port 8080 \
  --model models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf

curl -s http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}'
```

The Metal binary embeds standalone MSL and compiles it through Metal's runtime
API. Backend selection is explicit and never falls back to CPU.

More runnable CLI, chat, streaming, and metrics requests live in
[`examples/`](examples/).

## Repository tour

```
src/kipp.c              strict GGUF loading, tokenizer, sampling, CPU oracle
src/kipp.h              public model, tokenizer, sampling, KV-session C API
src/kipp_backend.h      fixed internal backend boundary + runtime config
src/kipp_checkpoints.h  the supported-checkpoint registry (the growth point)
src/kipp_chat.c         native Qwen3 ChatML rendering
src/kipp_spec.c         prompt-lookup speculative drafting
src/kipp_kv_pool.c      production KV block allocation and prefix sharing
src/kipp_cli.c          greedy/sampled decode CLI with timing metrics
src/kipp_server.c       completions/chat server: batching, SSE, metrics
src/metal/kipp_metal.m  the only Objective-C in the project (Metal bridge)
metal/kipp_kernels.metal standalone, build-time-embedded MSL kernels
src/cuda/kipp_cuda.cu   CUDA bridge, isolated from every other backend
cuda/kipp_kernels.cu    CUDA kernels validated on NVIDIA A100/H100
tests/                  native + server tests, generated reference vectors
tools/                  Python/shell tooling; checkpoints.py mirrors the registry
examples/               copy-pasteable CLI and HTTP requests
docs/                   design docs + the documentation site
AGENT.md                contribution constraints, for humans and AI agents alike
```

## Roadmap

| Phase | Goal |
|-------|------|
| **0** | Pin and specify Qwen3-4B-Base |
| **1** | CPU reference path: loader, tokenizer, forward pass, test vectors |
| **2** | Correct incremental decoding with a contiguous KV cache |
| **3** | Metal backend, validated against CPU reference outputs |
| **4** | Isolated CUDA backend, validated on NVIDIA hardware |
| **5** | Continuous batching, scheduler, and cross-session KV sharing |
| **6** | OpenAI Completions and Chat Completions subset |
| **7** | Reviewed extensions: quantization and speculative decoding |

Full details are in the
[roadmap](https://polished-snow.github.io/Kipp/roadmap.html). Deferred work
includes ROCm/HIP, extra model families, and arbitrary-GGUF compatibility.

## Documentation

The full documentation lives at **[polished-snow.github.io/Kipp](https://polished-snow.github.io/Kipp/)**:

- [Architecture](https://polished-snow.github.io/Kipp/architecture.html) —
  design constraints, backends, and correctness boundaries
- [Roadmap](https://polished-snow.github.io/Kipp/roadmap.html) —
  correctness-gated implementation phases
- [Model support](https://polished-snow.github.io/Kipp/model-support.html) —
  what supporting a checkpoint means
- [Benchmarks](https://polished-snow.github.io/Kipp/benchmarks.html) —
  measurement policy and reference results
- [Reproducing](https://polished-snow.github.io/Kipp/reproducing.html) —
  exact build, gate, and benchmark commands
- [Releasing](https://polished-snow.github.io/Kipp/releasing.html) —
  release checklist and hardware policy
- [Research](https://polished-snow.github.io/Kipp/research/inspiration-notes.html) —
  reference repositories and design notes
- [Contributing](https://polished-snow.github.io/Kipp/contributing.html) —
  ground rules for human and AI contributors

## Contributing

Read [`AGENT.md`](AGENT.md) first. It is the source of truth for every
contributor: keep a C11 core, isolate backends, preserve mmap-backed loading,
test on real hardware before claiming success, and never vendor code from the
[reference repositories](docs/research/inspiration-notes.md).

## Enjoying Kipp?

Kipp is built in the open by [Polished Snow](https://github.com/Polished-Snow).
If you find it useful, star the repository and follow project updates:

<p align="center">
  <a href="https://x.com/polished_snow">
    <img src="https://img.shields.io/badge/X-follow%20%40polished__snow%20for%20updates-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow @polished_snow on X">
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/company/plsn">
    <img src="https://img.shields.io/badge/LinkedIn-follow%20Polished%20Snow-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="Follow Polished Snow on LinkedIn">
  </a>
</p>

<p align="center">
  <a href="https://www.star-history.com/#Polished-Snow/Kipp&Date">
    <img src="https://api.star-history.com/svg?repos=Polished-Snow/Kipp&type=Date" alt="Star History Chart" width="600">
  </a>
</p>

## License

Kipp is open source under the [MIT License](LICENSE).

<p align="center">
  <sub>Built by <a href="https://github.com/Polished-Snow">Polished Snow</a> · <a href="https://x.com/polished_snow">X</a> · <a href="https://www.linkedin.com/company/plsn">LinkedIn</a></sub>
</p>
