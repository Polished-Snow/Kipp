# Kipp Roadmap

The order below is provisional until the first target model family is chosen.
Correctness gates each phase; dates and performance claims are intentionally
absent.

## Phase 0 — Select and specify the model

- Choose one model family and a deliberately small initial checkpoint set.
- Confirm model and weight licenses permit the intended distribution.
- Document tensor shapes, tokenizer behavior, precision, and hardware needs.
- Replace target-model placeholders and write Kipp's one-line description.

## Phase 1 — Build the CPU reference path

- Define the minimal native model format and mmap-backed loader.
- Implement tokenizer, forward pass, sampling, and deterministic test vectors.
- Add end-to-end correctness tests for supported checkpoints.

## Phase 2 — Add the Metal backend

- Define a narrow C-to-Objective-C backend boundary.
- Implement and validate Apple Silicon kernels against CPU reference outputs.
- Measure memory use, prefill throughput, and token-generation latency.

## Phase 3 — Add isolated CUDA backends

- Implement a generic CUDA path without coupling it to Metal internals.
- Add a separately tuned `cuda-spark` configuration if target hardware
  justifies it.
- Validate both CUDA targets on actual NVIDIA hardware.

## Phase 4 — Serve and schedule

- Add minimal OpenAI- and Anthropic-compatible streaming APIs.
- Introduce batching and KV-cache management only with correctness tests.
- Define failure behavior, resource limits, and reproducible benchmarks.

## Deferred

- ROCm/HIP support, on a separate community-maintained branch
- Additional model families
- Generic tensor-runtime or arbitrary-GGUF compatibility
- Broad API parity with llama.cpp, vLLM, or SGLang
