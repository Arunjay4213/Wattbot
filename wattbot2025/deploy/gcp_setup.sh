#!/bin/bash
# GCP Setup Script for WattBot 2025
# This script sets up a GCP VM and runs the pipeline

set -e

echo "======================================"
echo "WattBot 2025 - GCP Deployment Script"
echo "======================================"
echo ""

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-wattbot-2025}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-wattbot-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"  # 8 vCPUs, 30GB RAM
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"

echo "Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  Instance: $INSTANCE_NAME"
echo "  Machine Type: $MACHINE_TYPE"
echo "  GPU: $GPU_TYPE x $GPU_COUNT"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user wants GPU
read -p "Use GPU? (y/n, default: y): " USE_GPU
USE_GPU=${USE_GPU:-y}

if [[ "$USE_GPU" != "y" ]]; then
    echo "Using CPU-only configuration..."
    MACHINE_TYPE="c2-standard-8"
    GPU_TYPE=""
    GPU_COUNT="0"
fi

# Create instance
echo ""
echo "Creating GCP VM instance..."

if [[ "$USE_GPU" == "y" ]]; then
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-standard \
        --maintenance-policy=TERMINATE \
        --metadata-from-file startup-script=deploy/gcp_startup.sh \
        --scopes=https://www.googleapis.com/auth/cloud-platform
else
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --image-family=pytorch-latest-cpu \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-standard \
        --metadata-from-file startup-script=deploy/gcp_startup.sh \
        --scopes=https://www.googleapis.com/auth/cloud-platform
fi

echo ""
echo "✓ VM instance created successfully!"
echo ""
echo "Next steps:"
echo "1. Wait ~2 minutes for the VM to fully start"
echo "2. Copy your code and data to the VM:"
echo "   ./deploy/gcp_upload.sh"
echo "3. SSH into the VM:"
echo "   gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE"
echo "4. Run the pipeline:"
echo "   cd wattbot2025 && python3 run.py"
echo ""
echo "To delete the VM when done:"
echo "   gcloud compute instances delete $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE"
