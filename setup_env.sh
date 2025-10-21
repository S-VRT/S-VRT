#!/bin/bash
# Setup script for Deblur project
# Usage: source setup_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Add project root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "Environment setup complete!"
echo "PYTHONPATH set to: ${PYTHONPATH}"
echo ""
echo "You can now run training with:"
echo "  python src/train.py --config configs/deblur/vrt_spike_baseline.yaml"

