#!/bin/bash

# Build script for DCNv4 CUDA extension
# This script compiles the DCNv4 CUDA extension for use in the S-VRT project

echo "Building DCNv4 CUDA extension..."

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit not found. Please install CUDA toolkit first."
    exit 1
fi

# Check if PyTorch is available
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyTorch not found. Please install PyTorch first."
    exit 1
fi

# Build the extension using setup.py
echo "Compiling CUDA extension..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "DCNv4 CUDA extension built successfully!"
    echo "You can now use DCNv4 in your models by setting 'dcn_type': 'DCNv4' in your config."
else
    echo "Error: Failed to build DCNv4 CUDA extension."
    exit 1
fi

