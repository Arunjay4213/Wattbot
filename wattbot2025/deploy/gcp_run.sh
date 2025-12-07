#!/bin/bash
# Optimized run script for GCP VM
# Run this script on the GCP VM after uploading files

set -e

echo "======================================"
echo "WattBot 2025 - GCP Pipeline Runner"
echo "======================================"
echo ""

# Move to project directory
cd ~/wattbot2025

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please ensure you uploaded the .env file with your API keys"
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    export CUDA_VISIBLE_DEVICES=0
else
    echo "Running on CPU (no GPU detected)"
    echo ""
fi

# Set optimal settings for performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Show system resources
echo "System Resources:"
echo "  CPUs: $(nproc)"
echo "  RAM: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Run pipeline in tmux session (so it continues if SSH disconnects)
if command -v tmux &> /dev/null; then
    echo "Starting pipeline in tmux session 'wattbot'..."
    echo "You can detach with Ctrl+B, then D"
    echo "Reattach with: tmux attach -t wattbot"
    echo ""
    sleep 2

    tmux new-session -d -s wattbot "python3 run.py 2>&1 | tee pipeline_log.txt"
    tmux attach -t wattbot
else
    # Run directly if tmux not available
    echo "Starting pipeline..."
    python3 run.py 2>&1 | tee pipeline_log.txt
fi

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo ""
echo "Results saved to: data/processed/submission.csv"
echo ""
echo "To download results to your local machine, run:"
echo "  gcloud compute scp --project=\$GCP_PROJECT_ID --zone=\$GCP_ZONE $INSTANCE_NAME:~/wattbot2025/data/processed/submission.csv ."
