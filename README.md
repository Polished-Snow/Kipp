# Kipp

Kipp is a small, hand-written inference engine for a single model family on
local hardware. The first target model family has not been selected yet.

The inference path will be native code: C for the core, a narrow Objective-C
bridge and Metal kernels for Apple Silicon, and isolated CUDA kernels for
NVIDIA GPUs. Python and shell are reserved for tooling.

## Status

Kipp is at the repository-scaffold stage. No model or backend is implemented,
and no build target should be interpreted as working inference.

See `docs/ROADMAP.md` for the planned sequence and `AGENT.md` for contribution
constraints.

## License

Kipp is open source under the MIT License. See `LICENSE`.
