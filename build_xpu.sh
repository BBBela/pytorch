#!/bin/bash

# Build script for XPU oneDNN AddMM example
# This requires Intel oneAPI toolkit with SYCL and oneDNN

echo "Building XPU oneDNN AddMM example..."

# Check if Intel oneAPI is available
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "Setting up Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
elif [ -f "$HOME/intel/oneapi/setvars.sh" ]; then
    echo "Setting up Intel oneAPI environment from home directory..."
    source $HOME/intel/oneapi/setvars.sh
else
    echo "Intel oneAPI not found. Please install Intel oneAPI toolkit."
    echo "Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkit.html"
    exit 1
fi

# Check if icpx compiler is available
if ! command -v icpx &> /dev/null; then
    echo "Intel DPC++ compiler (icpx) not found."
    echo "Make sure Intel oneAPI DPC++/C++ Compiler is installed and sourced."
    exit 1
fi

# Check if we can find oneDNN
if [ ! -z "$DNNLROOT" ]; then
    ONEDNN_ROOT="$DNNLROOT"
elif [ -d "/opt/intel/oneapi/dnnl/latest" ]; then
    ONEDNN_ROOT="/opt/intel/oneapi/dnnl/latest"
else
    echo "oneDNN not found. Using default paths..."
    ONEDNN_ROOT=""
fi

echo "Using Intel DPC++ compiler: $(icpx --version | head -n 1)"

# Compile with Intel DPC++ compiler
if [ ! -z "$ONEDNN_ROOT" ]; then
    echo "Using oneDNN from: $ONEDNN_ROOT"
    INCLUDE_FLAGS="-I$ONEDNN_ROOT/include"
    LIBRARY_FLAGS="-L$ONEDNN_ROOT/lib -ldnnl"
else
    echo "Using system oneDNN paths..."
    INCLUDE_FLAGS=""
    LIBRARY_FLAGS="-ldnnl"
fi

icpx -fsycl -std=c++17 -O2 -Wall \
    $INCLUDE_FLAGS \
    standalone_addmm_xpu_onednn.cpp \
    -o standalone_addmm_xpu_onednn \
    $LIBRARY_FLAGS

if [ $? -eq 0 ]; then
    echo "Build successful! Run with: ./standalone_addmm_xpu_onednn"
    echo ""
    echo "Note: This program requires:"
    echo "  - Intel XPU (GPU) hardware"
    echo "  - Intel GPU drivers"
    echo "  - Intel Level Zero runtime"
else
    echo "Build failed!"
    echo ""
    echo "Make sure you have:"
    echo "  1. Intel oneAPI DPC++/C++ Compiler"
    echo "  2. Intel oneDNN with SYCL support"
    echo "  3. Intel GPU drivers and Level Zero runtime"
    echo ""
    echo "Install with:"
    echo "  sudo apt install intel-oneapi-toolkit"
    echo "  # or download from Intel's website"
fi