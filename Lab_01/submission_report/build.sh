#!/bin/bash
# ==============================================================================
# 1-Click Build Script for AI & ML Lab Report 01
# ==============================================================================

set -e

# Change directory to where this script is located
cd "$(dirname "$0")"

echo "================================================="
echo " Building AI & ML Lab Report 01 PDF..."
echo "================================================="

# Step 1: Update screenshots / models if needed
if [ "$1" == "--regen" ]; then
    echo "[1/3] Re-running spam classifier to update visuals..."
    python3 ../src/spam_classifier.py
else
    echo "[1/3] Using existing screenshots in screenshots/ (pass --regen to re-run pipeline)"
fi

# Step 2: Compile LaTeX with pdflatex
echo "[2/3] Compiling LaTeX document (main.tex)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null

# Step 3: Copy to final output and clean auxiliary files
cp main.pdf 2201032_Report_1.pdf
rm -f *.aux *.log *.out main.pdf

echo "[3/3] Done!"
echo "================================================="
echo " SUCCESS! Output generated:"
echo " -> $(pwd)/2201032_Report_1.pdf"
echo "================================================="
