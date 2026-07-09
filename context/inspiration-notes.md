# Inspiration Notes

These repositories are architecture references only. Kipp must not vendor or
copy their code. Before adapting anything beyond a general idea, inspect the
source file and its repository license, preserve required notices, and record
the provenance explicitly.

## antirez/ds4

Study the narrow, model-specific native-engine structure; readable C core;
isolated Metal and CUDA paths; mmap-oriented loading; KV-cache design; and
model-aware quantization decisions.

## ggml-org/llama.cpp and ggml-org/ggml

Study GGUF metadata and tensor layout, quantization formats and kernels,
backend boundaries, and portable low-level tensor representations. Kipp will
support only the subset required by its chosen model.

## ggml-org/whisper.cpp

Study how a deliberately narrow C engine keeps model loading, inference, and
platform acceleration understandable without becoming a generic runtime.

## karpathy/llm.c

Study naming, file organization, direct C/CUDA implementations, and the use
of small reference paths to make low-level numerical code approachable.

## GeeeekExplorer/nano-vllm

Study the concepts behind paged attention, request lifecycle, continuous
batching, and KV-cache ownership. Reimplement any selected ideas natively.

## vllm-project/vllm and sgl-project/sglang

Study scheduler behavior, memory management, prefix reuse, and serving
semantics. Treat their breadth as research material, not Kipp's initial scope.

## Research workflow

Once Nia is configured manually, index every repository listed in
`.cursor/commands/nia.md`. Notes derived from source research should name the
repository, file, revision, and applicable license.
