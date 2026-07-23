# Build the paper with tectonic (single-binary LaTeX). Falls back to latexmk.
PAPER := main

.PHONY: all clean arxiv

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic $(PAPER).tex; \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf $(PAPER).tex; \
	else \
		echo "install tectonic or latexmk (or use Overleaf)"; exit 1; \
	fi

# Self-contained arXiv source bundle. arXiv runs pdflatex+bibtex on its own
# TeXLive (which ships acmart, pgfplots, and the libertine/newtx fonts), so
# the tarball carries only the sources that are not in a standard TeXLive:
# main.tex, the bibliography (.bib and a pre-built .bbl so arXiv needs no
# bibtex pass), the generated results macros, and the pgfplots data files.
# Every figure is vector (pgfplots/tikz) so no images are bundled.
arxiv: $(PAPER).tex references.bib
	@command -v tectonic >/dev/null 2>&1 || { \
		echo "arxiv bundle needs tectonic to pre-build the .bbl"; exit 1; }
	tectonic --keep-intermediates $(PAPER).tex >/dev/null
	@rm -rf arxiv-build && mkdir -p arxiv-build/generated arxiv-build/data
	cp $(PAPER).tex references.bib $(PAPER).bbl arxiv-build/
	cp generated/results-macros.tex arxiv-build/generated/
	cp data/*.dat arxiv-build/data/
	cp SUBMITTING.md arxiv-build/00README.txt
	cd arxiv-build && tar czf ../arxiv-submission.tar.gz \
		$(PAPER).tex references.bib $(PAPER).bbl \
		generated/results-macros.tex data/*.dat 00README.txt
	@rm -rf arxiv-build
	@echo "wrote paper/arxiv-submission.tar.gz"
	@tar tzf arxiv-submission.tar.gz

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).pdf $(PAPER).fdb_latexmk $(PAPER).fls \
	      arxiv-submission.tar.gz
	rm -rf arxiv-build
