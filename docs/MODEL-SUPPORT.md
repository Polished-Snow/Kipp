# Kipp Model Support

No model family is supported yet.

Kipp will initially support one explicitly selected family and a small,
documented set of checkpoints. This file will record:

- exact upstream model identifiers and revisions;
- model, tokenizer, and weight licenses;
- supported parameter counts and architectural variants;
- accepted source and runtime weight formats;
- quantization and numeric-format support by backend;
- tokenizer and chat-template compatibility;
- minimum memory and hardware requirements; and
- known correctness or quality limitations.

Supporting a new checkpoint will require a reproducible conversion procedure,
reference test vectors, and validation on every backend claimed to support it.
Kipp is not intended to accept arbitrary GGUF files.
