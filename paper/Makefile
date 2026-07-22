# Build the paper with tectonic (single-binary LaTeX). Falls back to latexmk.
PAPER := main

.PHONY: all clean
all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic $(PAPER).tex; \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf $(PAPER).tex; \
	else \
		echo "install tectonic or latexmk (or use Overleaf)"; exit 1; \
	fi

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
	      $(PAPER).out $(PAPER).pdf
