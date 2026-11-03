#!/bin/bash
# ==============================================================================
# Build Script for AI & ML Lab Report 04
# ==============================================================================

set -e

cd "$(dirname "$0")"

echo "================================================="
echo " Building AI & ML Lab Report 04 PDF..."
echo "================================================="

if [ "$1" == "--regen" ]; then
    echo "[1/3] Re-running classifier script to regenerate visuals..."
    python3 ../src/cancer_ensemble_classifier.py
else
    echo "[1/3] Using existing figures in screenshots/ (pass --regen to re-run)"
fi

echo "[2/3] Compiling LaTeX document (main.tex)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null

cp main.pdf 2201032_Report_4.pdf
rm -f *.aux *.log *.out main.pdf

echo "[3/3] Done!"
echo "================================================="
echo " SUCCESS! Output generated:"
echo " -> $(pwd)/2201032_Report_4.pdf"
echo "================================================="

