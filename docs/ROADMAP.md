# Kipp Roadmap

This sequence is binding unless `ARCHITECTURE.md` is explicitly revised.
Correctness gates every phase; dates and speculative performance claims are
intentionally absent.

**Status (v0.0.1):** Phases 0 through 3 are complete and gated on Apple M5 hardware.
Native sampling, session truncation, and the Phase 6 Completions subset
(with SSE streaming, prefix reuse, and multi-choice requests) are
implemented and tested. Phase 5 continuous batching is live: batched
multi-sequence evaluation (`kipp_eval_batch`, gated batched-equals-isolated)
plus a FIFO event-loop scheduler in the server with chunked prefill,
admission control, and disconnect cancellation. KV blocks and prefix
sharing across sessions remain deferred. Phase 4 CUDA code exists but stops
short of a success claim until it runs on NVIDIA hardware.

## Phase 0 — Specify the model

- Pin `Qwen/Qwen3-4B-Base` and its exact revision.
- Confirm model, tokenizer, and weight licenses.
- Fix tensor shapes, tokenizer behavior, precision, memory, and API boundaries.

## Phase 1 — Build the CPU reference path

- Convert the pinned BF16 weights into Kipp's strict GGUF subset.
- Implement mmap-backed loading, native tokenization, scalar operators, and
  the readable full-prompt forward pass.
- Pass deterministic full-logit vectors with exact argmax and NMSE at most
  `1e-5`.

## Phase 2 — Add the KV cache

- Add bounded contiguous BF16 K/V storage and session lifecycle.
- Match no-cache CPU recomputation for incremental decoding.

## Phase 3 — Add the Metal backend

- Implement Apple Silicon kernels behind the fixed backend boundary.
- Validate prefill and decode against CPU on actual Metal hardware.

## Phase 4 — Add the CUDA backend

- Implement isolated resident-weight CUDA execution.
- Validate on the provided NVIDIA machine; stop if hardware is unavailable.

## Phase 5 — Continuous batching

- Introduce KV blocks and a small FIFO scheduler.
- Require batched requests to reproduce isolated execution.

## Phase 6 — Serve

- Add the approved non-streaming OpenAI Completions subset.
- Keep HTTP, scheduling, and backend execution separate.

## Phase 7 — Explicitly reviewed optimizations

- ROCm/HIP support, on a separate community-maintained branch
- Additional model families
- Generic tensor-runtime or arbitrary-GGUF compatibility
- Quantization, prefix/radix caching, SSD streaming, and speculative decoding
- Broad API parity with llama.cpp, vLLM, or SGLang
