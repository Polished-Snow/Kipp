# Kipp — Agent Instructions

Kipp is a small, hand-written inference engine for the pinned Qwen3 dense
checkpoints registered in `src/kipp_checkpoints.h` and documented in
`docs/ARCHITECTURE.md`. It is not a generic Qwen or GGUF runner — keep it
narrow: model support grows by adding registry entries and running the
gates, never by loosening validation.

- Core language is C. Objective-C only where Metal requires a bridge. Metal
  Shading Language for Apple GPU kernels. CUDA for NVIDIA kernels. Keep these
  paths isolated — a change to one backend must never silently break another.
- Keep model loading mmap-backed where possible; avoid eager full-weight
  copies into RAM.
- Any change that could affect a backend (Metal, CUDA, CPU reference path)
  must be tested against that backend before being called done. If you don't
  have access to test a backend, say so explicitly and ask before claiming
  it works.
- Prefer readability over cleverness. A contributor should be able to read
  `src/kipp.c` top to bottom and understand the whole forward pass.
- The CPU backend exists for reference/debugging only — never optimize for
  it at the expense of the GPU paths.
- Do not vendor code from reference repos (see
  `docs/research/inspiration-notes.md`).
  Read them for architecture, reimplement cleanly.

## Building and gating

Build targets: `make cpu`, `make server`, `make metal`, `make server-metal`,
`make cuda-generic`, `make server-cuda`, and `make cuda-spark` for DGX Spark.
Every feature is gated against the CPU oracle on real hardware before it is
called done:

- `make test` / `make test-sanitize` — hermetic unit tests (+ ASan/UBSan).
- Model gates run the built `build/kipp_test[_metal]` binary directly against
  the pinned vectors in `tests/test-vectors/` — do **not** go through the
  `vectors`→`convert` chain for a quick gate, it rewrites the multi-GB GGUF:
  `--model` (CPU logits vs pinned), `--paged-cpu` / `--paged-metal` (paged KV
  bitwise-equal to contiguous under a scrambled block table), `--pooled-cpu` /
  `--pooled-metal` (shared-prefix KV), `--multilogit`,
  `--phase3-metal` (CPU-vs-Metal within `1e-4` NMSE, identical argmax), and
  `--longctx-metal` (split-K decode partitioning).
- `make test-server` — OpenAI Completions + Chat Completions server tests.
- `make test-chat` / `make test-draft-spec` — chat and draft-model integration
  gates against converted model artifacts.

State plainly which backends you actually ran. CUDA is validated on ephemeral
cloud GPUs at phase milestones, not on this machine. The complete command
matrix lives in `docs/REPRODUCING.md`.
