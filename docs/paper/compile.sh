#!/bin/bash
# Compilation script for LaTeX documentation
# Optimization Primitive Library

set -e  # Exit on error

MAIN="main"

echo "======================================"
echo "Compiling LaTeX Documentation"
echo "======================================"
echo ""

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found"
    echo "Please install TeX Live or MiKTeX"
    exit 1
fi

echo "[1/4] First pass (pdflatex)..."
pdflatex -interaction=nonstopmode -halt-on-error $MAIN.tex > /dev/null 2>&1 || {
    echo "Error in first LaTeX pass"
    pdflatex -interaction=nonstopmode $MAIN.tex | tail -20
    exit 1
}

echo "[2/4] Processing bibliography (bibtex)..."
if [ -f "references.bib" ]; then
    bibtex $MAIN > /dev/null 2>&1 || echo "Warning: Bibliography processing failed (this is okay if no citations)"
fi

echo "[3/4] Second pass (pdflatex)..."
pdflatex -interaction=nonstopmode -halt-on-error $MAIN.tex > /dev/null 2>&1 || {
    echo "Error in second LaTeX pass"
    pdflatex -interaction=nonstopmode $MAIN.tex | tail -20
    exit 1
}

echo "[4/4] Final pass (pdflatex)..."
pdflatex -interaction=nonstopmode -halt-on-error $MAIN.tex > /dev/null 2>&1 || {
    echo "Error in final LaTeX pass"
    pdflatex -interaction=nonstopmode $MAIN.tex | tail -20
    exit 1
}

echo ""
echo "======================================"
echo "✓ Compilation successful!"
echo "======================================"
echo ""
echo "Output: $MAIN.pdf"
echo "Pages: $(pdfinfo $MAIN.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo 'unknown')"
echo "Size: $(du -h $MAIN.pdf | awk '{print $1}')"
echo ""

# Clean auxiliary files
echo "Cleaning auxiliary files..."
rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz
rm -f *.fdb_latexmk *.fls
echo "✓ Clean completed"
echo ""

# Try to open PDF
if command -v xdg-open &> /dev/null; then
    echo "Opening PDF with default viewer..."
    xdg-open $MAIN.pdf &
elif command -v open &> /dev/null; then
    echo "Opening PDF with default viewer..."
    open $MAIN.pdf
else
    echo "PDF viewer not found. Open $MAIN.pdf manually."
fi
