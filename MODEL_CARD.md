# Kipp — Model Support Card

Kipp is a narrow inference engine: it runs a fixed, compiled-in registry of
pinned checkpoints, not arbitrary GGUF or Qwen models. Support grows only by
adding a registry entry (`src/kipp_checkpoints.h`) and passing the gates.

## Supported models

The **Qwen3 dense family**, base and instruct, pinned by repository revision:

| Checkpoint | Params | Layers | Hidden | tie-embed | Notes |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.6B | 28 | 1024 | yes | 16 Q-heads (attn width ≠ hidden) |
| Qwen3-1.7B-Base | 1.7B | 28 | 2048 | yes | |
| Qwen3-4B-Base | 4B | 36 | 2560 | yes | reference checkpoint |
| Qwen3-4B (instruct, hybrid think) | 4B | 36 | 2560 | yes | ChatML |
| Qwen3-4B-Instruct-2507 | 4B | 36 | 2560 | yes | rope θ 5e6, 262144 ctx |
| Qwen3-8B (Base/Instruct) | 8B | 36 | 4096 | no | separate `lm_head` |
| Qwen3-14B-Base | 14B | 40 | 5120 | no | quant needed for Metal |
| Qwen3-32B-Instruct | 32B | 64 | 5120 | no | 64 Q-heads; quant needed |

Family constants: head_dim 128, 8 KV heads, vocab 151936, RMS eps 1e-6.

## Weight schemes

- **BF16** — the reference; byte-identical forward pass to Kipp v0.0.1.
- **Q8_0** — 8-bit, 34-byte blocks, near-lossless.
- **Affine (4-bit, group size 32)** — 20-byte groups, `w = scale·q + bias`.

Only the seven per-layer projections are quantized; embeddings, `lm_head`,
and norms stay BF16.

## Backends actually tested

Per project policy, we state which backends were run on real hardware:

- **CPU** (scalar reference oracle) — all checkpoints.
- **Apple Metal** (Apple M5, 24 GB unified memory) — 0.6B/4B/8B and
  quantized 4B/8B; 14B/32B BF16 exceed the single-buffer cap and need quant.
- **NVIDIA CUDA** (A100, via ephemeral cloud runs) — correctness gates for
  the family incl. 14B/32B; kernels are compiled and validated for argmax
  and NMSE against the CPU oracle. CUDA throughput numbers are measured
  separately and disclosed only when run.

## Correctness contract

Every backend is gated against the CPU oracle: identical argmax and
NMSE ≤ 1e-4 at prefill and every checked decode position. The paged KV cache
is additionally gated **bitwise-equal to the contiguous layout under an
adversarially scrambled block table** (`--paged-cpu`, `--paged-metal`).

## Limitations

- Not a general model runner; unpinned checkpoints are rejected by design.
- Quantized **prefill** on Metal uses the vector kernel (no quantized
  simdgroup-matrix path yet), so quantization trades faster decode for slower
  prefill on long prompts.
- CUDA paged addressing and the shared cross-request KV block pool are not yet
  wired into the serving path.
