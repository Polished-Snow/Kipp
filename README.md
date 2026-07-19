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

> **Status (v0.0.2):** Kipp targets the **pinned Qwen3 dense family**
> (0.6B–32B, base + instruct) through a compiled-in checkpoint registry —
> strict validation, per-checkpoint pinned revisions and hashes, one shared
> forward pass. Validated checkpoints run on the CPU oracle plus **Metal** on
> Apple M5 or **CUDA** on an NVIDIA A100; registry-only entries are not
> presented as validated. Kipp runs
> **BF16, Q8_0, and 4-bit affine** weights — from near-lossless Q8 to a
> smaller Q4-class option, with ~1.6–2× faster decode and larger checkpoints
> under Apple's Metal single-buffer cap. The server speaks the OpenAI
> **Completions and Chat Completions** subset (SSE streaming, batched
> multi-sequence decode, serial prefix reuse), a full sampling surface
> (temperature, top-p, top-k, min-p, penalties, logit_bias, seed), and a
> Prometheus `/metrics` endpoint.

## What is Kipp?

Kipp is a small, **hand-written Qwen3 inference engine** for local hardware. Most engines try to run everything; Kipp deliberately runs *one pinned model family*, and aims to run it well enough that you can read the entire engine in an afternoon.

- **Readable by design** — the reference forward pass stays linear and model-specific, so you can follow one token from embeddings to logits in `src/kipp.c`.
- **Native all the way down** — a C11 core, a narrow Objective-C bridge with Metal kernels for Apple Silicon, and isolated CUDA kernels for NVIDIA.
- **No frameworks in the hot path** — Python and shell are reserved for tooling; PyTorch, JAX, and friends never enter the inference process.
- **One model family, deeply** — in the lineage of `whisper.cpp` and `antirez/ds4`, not a generic runner like llama.cpp or vLLM.
- **Correctness first** — the CPU path is the reference; every GPU backend must match it, on real hardware, before it's called working.
- **mmap-backed loading** — weights are mapped, not eagerly copied into RAM, wherever the format permits.
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
make test-model                       # Phase 1 gate for the default 4B
make test-phase2
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
src/kipp_kv_pool.c      tested, not-yet-integrated KV block bookkeeping
src/kipp_cli.c          greedy/sampled decode CLI with timing metrics
src/kipp_server.c       completions/chat server: batching, SSE, metrics
src/metal/kipp_metal.m  the only Objective-C in the project (Metal bridge)
metal/kipp_kernels.metal standalone, build-time-embedded MSL kernels
src/cuda/kipp_cuda.cu   CUDA bridge, isolated from every other backend
cuda/kipp_kernels.cu    CUDA kernels validated on NVIDIA A100
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
| **5** | Continuous batching and scheduler; cross-session KV sharing remains |
| **6** | OpenAI Completions and Chat Completions subset |
| **7** | Explicitly reviewed extensions; quantization and speculative decode delivered |

Full details in the [roadmap](https://polished-snow.github.io/Kipp/roadmap.html) · deferred: ROCm/HIP, extra model families, arbitrary-GGUF compatibility.

## Documentation

The full documentation lives at **[polished-snow.github.io/Kipp](https://polished-snow.github.io/Kipp/)**:

- [Architecture](https://polished-snow.github.io/Kipp/architecture.html) — design constraints, backends, correctness boundaries
- [Roadmap](https://polished-snow.github.io/Kipp/roadmap.html) — correctness-gated implementation phases
- [Model support](https://polished-snow.github.io/Kipp/model-support.html) — what "supporting a checkpoint" means here
- [Benchmarks](https://polished-snow.github.io/Kipp/benchmarks.html) — the reporting rules future numbers must follow
- [Contributing](https://polished-snow.github.io/Kipp/contributing.html) — ground rules, including for AI agents

## Contributing

Read [`AGENT.md`](AGENT.md) first — it is the source of truth for every contributor, human or AI. The short version: C11 core, isolated backends, mmap-backed loading, test on real hardware before claiming success, and never vendor code from the [reference repos](docs/research/inspiration-notes.md).

## Enjoying Kipp?

Kipp is built in the open by [Polished Snow](https://github.com/Polished-Snow). If you like where this is going, the single best way to support the project is to **star the repo** and **follow along** — model selection, first kernels, and benchmarks will be announced there first:

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
  <sub>Built with ❄️ by <a href="https://github.com/Polished-Snow">Polished Snow</a> · <a href="https://x.com/polished_snow">X</a> · <a href="https://www.linkedin.com/company/plsn">LinkedIn</a></sub>
</p>
