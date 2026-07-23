# Submitting the Kipp paper

The paper builds with `tectonic main.tex` (or `make`). This note covers the
two archival steps — an arXiv preprint and a Zenodo software DOI — that
still require the author's own accounts.

## arXiv preprint

`make arxiv` writes `paper/arxiv-submission.tar.gz`: a self-contained source
bundle of `main.tex`, `references.bib`, a pre-built `main.bbl`,
`generated/results-macros.tex`, and `data/*.dat`. Everything else the paper
uses — the `acmart` class, `pgfplots`, and the Linux Libertine / newtx fonts
— ships in arXiv's TeXLive, and every figure is vector (pgfplots/tikz), so
no fonts or images are bundled.

To submit:

1. Regenerate numbers and the PDF from committed benchmarks first:
   `make -C .. paper-data && make` (then `make -C .. paper-check` must pass).
2. `make arxiv`, then upload `arxiv-submission.tar.gz` at
   <https://arxiv.org/submit>. Primary class **cs.DC** (Distributed,
   Parallel, and Cluster Computing); cross-list **cs.LG**.
3. arXiv runs `pdflatex` + the bundled `.bbl` (no `bibtex` pass needed).
   If it reports an `acmart` version skew, bump the local build's TeXLive
   and rebuild the `.bbl`; the sources use no version-specific features.
4. The `\acmDOI`/conference metadata is suppressed by the `nonacm` class
   option, so the preprint carries no spurious ACM footer.

## Zenodo software DOI

`.zenodo.json` at the repository root describes the software artifact. To
mint a DOI:

1. Link the GitHub repository to Zenodo (<https://zenodo.org/account/settings/github/>).
2. Cut a GitHub release (the release flow in `docs/RELEASING.md` already
   tags `vX.Y.Z`); Zenodo archives the tagged tree and mints a version DOI
   plus a concept DOI that always resolves to the latest.
3. Add the concept DOI to `CITATION.cff` (`doi:` field) and to the paper's
   Artifact Availability section, replacing the placeholder.

The paper's artifact appendix already lists the exact build and gate
commands; the Zenodo archive is the citable snapshot those commands run
against.
