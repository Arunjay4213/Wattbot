#!/bin/bash
# Commands to run on your VM (ml-research-vm)
# You're already SSH'd in, so just copy-paste these!

echo "=========================================="
echo "WattBot 2025 - VM Setup"
echo "=========================================="
echo ""

# Create directory structure
echo "Creating directories..."
mkdir -p ~/wattbot2025/data/{raw,chunks,processed,cache}
cd ~/wattbot2025

echo "✅ Directories created"
echo ""

# Update system packages
echo "Updating system..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git -qq

echo "✅ System updated"
echo ""

# Install Python dependencies
echo "Installing Python packages (this may take a few minutes)..."
python3 -m pip install --upgrade pip --quiet

python3 -m pip install \
    pandas==2.0.3 \
    numpy==1.24.3 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    google-generativeai==0.3.2 \
    sentence-transformers==2.2.2 \
    rank-bm25==0.2.2 \
    torch==2.1.2 \
    transformers==4.35.0 \
    scikit-learn==1.3.0

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next: Upload your WattBot files to ~/wattbot2025/"
echo ""
echo "You can use:"
echo "  1. gcloud compute scp (from your Mac)"
echo "  2. Upload via Cloud Console SSH window"
echo "  3. Git clone (if you have a repo)"
echo ""
