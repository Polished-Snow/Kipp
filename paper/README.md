# Kipp paper

LaTeX source for the Kipp paper, *"Oracle-Gated, Placement-Invariant Paged
Attention for Verifiable LLM Inference."*

## Files

- `main.tex` — the paper. Measured numbers are `\newcommand` macros at the top
  of the file, each annotated with the `bench/results/` JSON it came from.
- `references.bib` — bibliography (verified titles/authors/arXiv IDs).
- `figures/`, `data/` — figures and derived data.

## Compile

Any LaTeX toolchain compiles it (standard packages, pdfLaTeX). We use
[`tectonic`](https://tectonic-typesetting.github.io/) locally:

```bash
tectonic main.tex        # -> main.pdf
# or:  make
```

On **Overleaf**: upload `main.tex` + `references.bib`, set the compiler to
pdfLaTeX, and compile. No custom class files are required.

## Updating numbers

Re-run the harness (see `../bench/README.md`), then update the macros at the
top of `main.tex` — the prose and tables read them, so numbers stay
single-sourced.

**The rule: a macro may cite only a committed `bench/results/*.json`.** Every
macro carries a trailing comment naming its source file; a macro whose source
is uncommitted (or marked `STALE`) must not survive to a released draft.
Result files record the engine commit and hardware, so a hardware change
(e.g. the M5 → M5 Max move) is visible in the provenance chain rather than
silently mixed into the tables.
