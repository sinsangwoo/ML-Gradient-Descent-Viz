# Makefile for LaTeX compilation
# Optimization Primitive Library Documentation

# Main document
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compiler flags
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Output directory
OUTDIR = .

# Source files
SOURCES = main.tex algorithms.tex experiments.tex numerical_analysis.tex conclusion.tex
BIB = references.bib

# Default target
.PHONY: all
all: $(MAIN).pdf

# Full compilation with bibliography
.PHONY: full
full: clean
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "Full compilation completed: $(MAIN).pdf"

# Quick compilation (no bibliography)
$(MAIN).pdf: $(SOURCES) $(BIB)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "Quick compilation completed: $(MAIN).pdf"

# Clean auxiliary files
.PHONY: clean
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz
	rm -f *.fdb_latexmk *.fls
	@echo "Clean completed"

# Clean everything including PDF
.PHONY: distclean
distclean: clean
	rm -f $(MAIN).pdf
	@echo "All files cleaned"

# View PDF
.PHONY: view
view: $(MAIN).pdf
	@if command -v xdg-open > /dev/null 2>&1; then \
		xdg-open $(MAIN).pdf; \
	elif command -v open > /dev/null 2>&1; then \
		open $(MAIN).pdf; \
	else \
		echo "No PDF viewer found"; \
	fi

# Count words (approximate)
.PHONY: wordcount
wordcount:
	@echo "Approximate word count:"
	@texcount -total -q $(MAIN).tex

# Check for errors
.PHONY: check
check:
	@echo "Checking LaTeX syntax..."
	@$(LATEX) -draftmode $(MAIN).tex > /dev/null 2>&1 && echo "No errors found" || echo "Errors detected"

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make          - Quick compilation (2 passes)"
	@echo "  make full     - Full compilation with bibliography"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean - Remove all generated files"
	@echo "  make view     - Open PDF in viewer"
	@echo "  make wordcount - Count words in document"
	@echo "  make check    - Check for LaTeX errors"
	@echo "  make help     - Show this help message"
