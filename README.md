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

> **Status (v0.0.1):** Phases 1 through 3 for the pinned `Qwen/Qwen3-4B-Base`
> BF16 checkpoint are implemented, plus native sampling, session truncation,
> and the OpenAI Completions server with SSE streaming and serial prefix
> reuse. The CPU correctness oracle and explicit Metal backend both support
> contiguous BF16 KV sessions; Metal batches prefill through simdgroup-matrix
> BF16 kernels, streams attention with a split-K online softmax, and passes
> the full-logit gate on Apple M5 with no CPU fallback. CUDA kernels exist but stay unvalidated until the Phase 4
> gate runs on NVIDIA hardware. Continuous batching (Phase 5) is not
> implemented; the server handles one request at a time.

## What is Kipp? ⚡

Kipp is a small, **hand-written Qwen3-4B-Base inference engine** for local hardware. Most engines try to run everything; Kipp deliberately runs *one pinned checkpoint*, and aims to run it well enough that you can read the entire engine in an afternoon.

- 📜 **Readable by design** — the reference forward pass stays linear and model-specific, so you can follow one token from embeddings to logits in `src/kipp.c`.
- ⚡ **Native all the way down** — a C11 core, a narrow Objective-C bridge with Metal kernels for Apple Silicon, and isolated CUDA kernels for NVIDIA.
- 🚫 **No frameworks in the hot path** — Python and shell are reserved for tooling; PyTorch, JAX, and friends never enter the inference process.
- 🧩 **One model family, deeply** — in the lineage of `whisper.cpp` and `antirez/ds4`, not a generic runner like llama.cpp or vLLM.
- 🔐 **Correctness first** — the CPU path is the reference; every GPU backend must match it, on real hardware, before it's called working.
- 💾 **mmap-backed loading** — weights are mapped, not eagerly copied into RAM, wherever the format permits.

## Quickstart 💿

```bash
git clone https://github.com/Polished-Snow/Kipp.git
cd Kipp
make test         # warning-clean native unit tests; no model download
make cpu          # builds the session-backed CPU CLI
make metal        # builds the Apple Metal CLI
make server-metal # builds the OpenAI Completions server (Metal backend)
```

The real-model correctness gates are explicit because they download and
convert roughly 8 GB of pinned BF16 weights:

```bash
make test-model
make test-phase2
make test-metal
make test-server
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

## Repository tour 🗺️

```
src/kipp.c              strict GGUF loading, tokenizer, sampling, CPU oracle
src/kipp.h              public model, tokenizer, sampling, KV-session C API
src/kipp_backend.h      fixed internal backend boundary
src/kipp_cli.c          greedy/sampled decode CLI with timing metrics
src/kipp_server.c       OpenAI Completions server: SSE + prefix reuse
src/metal/kipp_metal.m  the only Objective-C in the project (Metal bridge)
metal/kipp_kernels.metal standalone, build-time-embedded MSL kernels
src/cuda/kipp_cuda.cu   CUDA bridge, isolated from every other backend
cuda/kipp_kernels.cu    CUDA kernels (unvalidated until Phase 4 hardware)
tests/                  native + server tests, generated reference vectors
tools/                  Python/shell tooling only — never part of inference
docs/                   design docs + the documentation site
AGENT.md                contribution constraints, for humans and AI agents alike
```

## Roadmap 🧭

| Phase | Goal |
|-------|------|
| **0** | Pin and specify Qwen3-4B-Base |
| **1** | CPU reference path: loader, tokenizer, forward pass, test vectors |
| **2** | Correct incremental decoding with a contiguous KV cache |
| **3** | Metal backend, validated against CPU reference outputs |
| **4** | Isolated CUDA backend, validated on NVIDIA hardware |
| **5** | Continuous batching and scheduler |
| **6** | Minimal OpenAI Completions-compatible serving |
| **7** | Explicitly reviewed optimizations such as quantization |

Full details in the [roadmap](https://polished-snow.github.io/Kipp/roadmap.html) · deferred: ROCm/HIP, extra model families, arbitrary-GGUF compatibility.

## Documentation 📚

The full documentation lives at **[polished-snow.github.io/Kipp](https://polished-snow.github.io/Kipp/)**:

- [Architecture](https://polished-snow.github.io/Kipp/architecture.html) — design constraints, backends, correctness boundaries
- [Roadmap](https://polished-snow.github.io/Kipp/roadmap.html) — correctness-gated implementation phases
- [Model support](https://polished-snow.github.io/Kipp/model-support.html) — what "supporting a checkpoint" means here
- [Benchmarks](https://polished-snow.github.io/Kipp/benchmarks.html) — the reporting rules future numbers must follow
- [Contributing](https://polished-snow.github.io/Kipp/contributing.html) — ground rules, including for AI agents

## Contributing 🤝

Read [`AGENT.md`](AGENT.md) first — it is the source of truth for every contributor, human or AI. The short version: C11 core, isolated backends, mmap-backed loading, test on real hardware before claiming success, and never vendor code from the [reference repos](context/inspiration-notes.md).

## Enjoying Kipp? ⭐

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

## License 📄

Kipp is open source under the [MIT License](LICENSE).

<p align="center">
  <sub>Built with ❄️ by <a href="https://github.com/Polished-Snow">Polished Snow</a> · <a href="https://x.com/polished_snow">X</a> · <a href="https://www.linkedin.com/company/plsn">LinkedIn</a></sub>
</p>
