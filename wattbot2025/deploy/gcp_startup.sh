#!/bin/bash
# GCP VM Startup Script
# This runs automatically when the VM starts

set -e

echo "=== WattBot 2025 VM Startup Script ==="
date

# Update system
echo "Updating system packages..."
apt-get update -qq

# Install required system packages
echo "Installing system dependencies..."
apt-get install -y -qq \
    python3-pip \
    python3-dev \
    git \
    htop \
    tmux \
    vim

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install Python packages globally
echo "Installing Python packages..."
python3 -m pip install --upgrade \
    pandas==2.0.3 \
    numpy==1.24.3 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    anthropic==0.7.0 \
    google-generativeai==0.3.2 \
    sentence-transformers==2.2.2 \
    rank-bm25==0.2.2 \
    torch==2.1.2 \
    transformers==4.35.0 \
    scikit-learn==1.3.0

# Create working directory
mkdir -p /home/wattbot
cd /home/wattbot

echo "=== Startup complete ==="
echo "VM is ready for WattBot deployment"
date
