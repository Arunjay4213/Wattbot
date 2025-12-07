#!/bin/bash
# Deploy WattBot to your existing GCP VM
# VM: ml-research-vm (35.224.94.133)

set -e

PROJECT_ID="wattbot-2025"
ZONE="us-central1-f"
INSTANCE_NAME="ml-research-vm"
EXTERNAL_IP="35.224.94.133"

echo "=========================================="
echo "WattBot Deployment to Existing VM"
echo "=========================================="
echo ""
echo "VM Details:"
echo "  Name: $INSTANCE_NAME"
echo "  IP: $EXTERNAL_IP"
echo "  Zone: $ZONE"
echo "  Type: n2-standard-4 (4 vCPUs, 16GB RAM)"
echo ""

# Check if we're in the right directory
if [ ! -f "run.py" ]; then
    echo "Error: Please run from wattbot2025 directory"
    exit 1
fi

echo "Step 1: Connecting to VM and setting up..."
echo ""

# Create directory structure
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --command="mkdir -p ~/wattbot2025/data/{raw,chunks,processed,cache}"

echo "✅ Directory structure created"
echo ""

echo "Step 2: Uploading source code..."
gcloud compute scp --recurse \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    src configs run.py .env \
    $INSTANCE_NAME:~/wattbot2025/

echo "✅ Code uploaded"
echo ""

echo "Step 3: Uploading data files..."
gcloud compute scp --recurse \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    data/raw data/chunks \
    $INSTANCE_NAME:~/wattbot2025/data/

echo "✅ Data uploaded"
echo ""

echo "Step 4: Installing dependencies..."
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --command="
        cd ~/wattbot2025
        python3 -m pip install --upgrade pip --quiet
        python3 -m pip install pandas==2.0.3 numpy==1.24.3 \
            python-dotenv==1.0.0 pyyaml==6.0.1 tqdm==4.66.1 \
            google-generativeai==0.3.2 sentence-transformers==2.2.2 \
            rank-bm25==0.2.2 torch==2.1.2 transformers==4.35.0 \
            scikit-learn==1.3.0 --quiet
        echo '✅ Dependencies installed'
    "

echo ""
echo "Step 5: Running pipeline..."
echo ""
echo "This will take 15-30 minutes on CPU (no GPU on n2-standard-4)"
echo ""

gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --command="
        cd ~/wattbot2025
        python3 run.py 2>&1 | tee pipeline_log.txt
    "

echo ""
echo "Step 6: Downloading results..."
mkdir -p data/processed
gcloud compute scp \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    $INSTANCE_NAME:~/wattbot2025/data/processed/submission.csv \
    data/processed/

gcloud compute scp \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    $INSTANCE_NAME:~/wattbot2025/pipeline_log.txt \
    ./ 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Results: data/processed/submission.csv"
echo "Log: pipeline_log.txt"
echo ""
echo "Note: Your VM is still running."
echo "To stop it (and save costs):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
