# Kipp paper

LaTeX source for the Kipp paper, *"Oracle-Gated, Placement-Invariant Paged
Attention for Verifiable LLM Inference."*

## Files

- `main.tex` — the paper (`acmart`, `sigconf`/`nonacm`). It `\input`s the
  generated macros and reads its plot data from `data/*.dat`; it contains no
  hand-typed measured numbers.
- `references.bib` — bibliography (verified titles/authors/arXiv IDs).
- `generated/results-macros.tex` — every measured number, one macro each,
  produced by `make paper-data` from committed `bench/results/*.json`.
- `data/*.dat` — pgfplots tables, likewise generated. All figures are vector
  (pgfplots/tikz); `figures/` holds no raster assets.
- `SUBMITTING.md` — arXiv and Zenodo submission steps.

## Compile

Any LaTeX toolchain compiles it (`acmart` + `pgfplots`, standard TeXLive).
We use [`tectonic`](https://tectonic-typesetting.github.io/) locally:

```bash
tectonic main.tex        # -> main.pdf
# or:  make
```

On **Overleaf**: upload `main.tex`, `references.bib`, `generated/`, and
`data/`; set the compiler to pdfLaTeX.

## Updating numbers

Re-run the harness (see `../bench/README.md`), then regenerate:

```bash
make -C .. paper-data     # bench/results/*.json -> generated/ + data/
make                      # rebuild main.pdf
make -C .. paper-check    # fail unless every input is committed and current
```

**The rule: a macro may cite only a committed `bench/results/*.json`.** Each
generated macro carries a trailing comment naming its source file and field;
`make paper-check` (run in CI via `make test`) refuses to pass if any consumed
result is missing, untracked, or modified, or if the generated files drift
from a fresh regeneration. Result files record the engine commit and hardware,
so a hardware change is visible in the provenance chain rather than silently
mixed into the tables.

## Submitting

`make arxiv` writes a self-contained `arxiv-submission.tar.gz`; see
`SUBMITTING.md` for the arXiv upload and the Zenodo software-DOI flow.
