#!/bin/bash

# Decompress model files script for REBL project
echo "Decompressing model files..."

# Decompress main model files
if [ -f "biobert_model.pt.gz" ]; then
    echo "Decompressing biobert_model.pt.gz..."
    gunzip -k biobert_model.pt.gz
fi

if [ -f "baseline_model.pt.gz" ]; then
    echo "Decompressing baseline_model.pt.gz..."
    gunzip -k baseline_model.pt.gz
fi

# Decompress saved models
if [ -f "saved_models/biobert_model.pt.gz" ]; then
    echo "Decompressing saved_models/biobert_model.pt.gz..."
    gunzip -k saved_models/biobert_model.pt.gz
fi

if [ -f "saved_models/improved_biobert_model.pt.gz" ]; then
    echo "Decompressing saved_models/improved_biobert_model.pt.gz..."
    gunzip -k saved_models/improved_biobert_model.pt.gz
fi

echo "Decompression complete!"
echo "Note: Large BioBERT models (413MB each) are still too large for GitHub."
echo "You may need to download them separately or use Git LFS." 