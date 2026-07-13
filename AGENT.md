# Kipp — Agent Instructions

Kipp is a small, hand-written inference engine for the pinned
`Qwen/Qwen3-4B-Base` checkpoint documented in `docs/ARCHITECTURE.md`. It is not
a generic Qwen or GGUF runner — keep it narrow.

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
- Do not vendor code from reference repos (see `context/inspiration-notes.md`).
  Read them for architecture, reimplement cleanly.
