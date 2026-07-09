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

> **Status:** Kipp is at the repository-scaffold stage. The first target model family has not been selected yet, and no build target should be interpreted as working inference. Everything below describes the design — honestly labeled, built in the open.

## What is Kipp? ⚡

Kipp is a small, **hand-written inference engine for a single model family** on local hardware. Most engines try to run everything; Kipp deliberately runs *one thing*, and aims to run it well enough that you can read the entire engine in an afternoon.

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
make cpu   # builds the CPU reference-path placeholders (warning-clean C11)
```

Each backend has an explicit build target — there is no auto-detection, by design:

```bash
make metal         # Apple Silicon backend   — not implemented yet
make cuda-generic  # generic NVIDIA backend  — not implemented yet
make cuda-spark    # tuned NVIDIA config     — not implemented yet
```

## Repository tour 🗺️

```
src/kipp.c              core engine: loading, tensors, the reference forward pass
src/kipp_server.c       future OpenAI- & Anthropic-compatible HTTP server
src/metal/kipp_metal.m  the only Objective-C in the project (Metal bridge)
src/cuda/kipp_cuda.cu   CUDA kernels, isolated from every other backend
tests/                  native test harness + generated reference vectors
tools/                  Python/shell tooling only — never part of inference
docs/                   design docs + the documentation site
AGENT.md                contribution constraints, for humans and AI agents alike
```

## Roadmap 🧭

| Phase | Goal |
|-------|------|
| **0** | Select and specify the first model family |
| **1** | CPU reference path: loader, tokenizer, forward pass, test vectors |
| **2** | Metal backend, validated against CPU reference outputs |
| **3** | Isolated CUDA backends (`cuda-generic`, then `cuda-spark`) |
| **4** | Minimal OpenAI- & Anthropic-compatible serving with batching |

Full details in the [roadmap](https://polished-snow.github.io/Kipp/roadmap.html) · deferred: ROCm/HIP, extra model families, arbitrary-GGUF compatibility.

## Documentation 📚

The full documentation lives at **[polished-snow.github.io/Kipp](https://polished-snow.github.io/Kipp/)**:

- [Architecture](https://polished-snow.github.io/Kipp/architecture.html) — design constraints, backends, correctness boundaries
- [Roadmap](https://polished-snow.github.io/Kipp/roadmap.html) — the five phases, correctness-gated
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
