#!/bin/bash
# WattBot 2025 - Cloud Shell One-Click Deployment
# Run this script in Google Cloud Shell for fully automated deployment

set -e

# Configuration
PROJECT_ID="wattbot-2025"
ZONE="us-central1-a"
INSTANCE_NAME="wattbot-vm"
MACHINE_TYPE="n1-standard-8"

echo "=========================================="
echo "WattBot 2025 - Automated Cloud Deployment"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Create a GPU VM instance"
echo "  2. Upload your code and data"
echo "  3. Install dependencies"
echo "  4. Run the pipeline"
echo "  5. Download results"
echo "  6. Delete the VM"
echo ""
echo "Estimated time: 10-15 minutes"
echo "Estimated cost: \$0.10-0.25"
echo ""

# Set project
gcloud config set project $PROJECT_ID

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    echo "⚠️  Instance '$INSTANCE_NAME' already exists!"
    read -p "Delete and recreate? (y/n): " RECREATE
    if [ "$RECREATE" = "y" ]; then
        echo "Deleting existing instance..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    else
        echo "Exiting..."
        exit 1
    fi
fi

# Create VM instance
echo ""
echo "📦 Creating VM instance with GPU..."
echo ""

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=install-nvidia-driver=True

echo ""
echo "✅ VM created successfully!"
echo ""
echo "⏳ Waiting 90 seconds for VM to fully boot..."
sleep 90

# Create directory structure on VM
echo ""
echo "📁 Setting up directory structure..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="mkdir -p ~/wattbot2025/data/{raw,chunks,processed,cache}"

# Check if we're running from the wattbot2025 directory
if [ ! -f "run.py" ]; then
    echo ""
    echo "⚠️  Error: Please run this script from the wattbot2025 directory"
    echo "   cd /path/to/wattbot2025"
    echo "   ./deploy/cloud_shell_deploy.sh"
    exit 1
fi

# Upload code
echo ""
echo "📤 Uploading source code..."
gcloud compute scp --recurse --zone=$ZONE \
    src configs run.py .env \
    $INSTANCE_NAME:~/wattbot2025/

# Upload data
echo ""
echo "📤 Uploading data files (this may take a few minutes)..."
gcloud compute scp --recurse --zone=$ZONE \
    data/raw data/chunks \
    $INSTANCE_NAME:~/wattbot2025/data/

# Create requirements.txt on VM and install
echo ""
echo "📦 Installing Python dependencies on VM..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd ~/wattbot2025
    cat > requirements.txt << 'EOF'
pandas==2.0.3
numpy==1.24.3
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1
google-generativeai==0.3.2
sentence-transformers==2.2.2
rank-bm25==0.2.2
torch==2.1.2
transformers==4.35.0
scikit-learn==1.3.0
EOF
    python3 -m pip install --upgrade pip --quiet
    python3 -m pip install -r requirements.txt --quiet
    echo '✅ Dependencies installed'
"

# Run the pipeline
echo ""
echo "=========================================="
echo "🚀 Starting WattBot Pipeline on GPU VM"
echo "=========================================="
echo ""
echo "This will take approximately 10-15 minutes."
echo "You can monitor progress in real-time..."
echo ""

# Run in a way that streams output
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd ~/wattbot2025
    python3 run.py 2>&1 | tee pipeline_log.txt
" || {
    echo ""
    echo "⚠️  Pipeline failed or was interrupted!"
    echo "VM is still running. To debug:"
    echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    echo ""
    read -p "Delete VM anyway? (y/n): " DELETE
    if [ "$DELETE" != "y" ]; then
        exit 1
    fi
}

# Download results
echo ""
echo "📥 Downloading results..."
mkdir -p data/processed
gcloud compute scp --zone=$ZONE \
    $INSTANCE_NAME:~/wattbot2025/data/processed/submission.csv \
    data/processed/ || echo "⚠️  Could not download submission.csv"

gcloud compute scp --zone=$ZONE \
    $INSTANCE_NAME:~/wattbot2025/pipeline_log.txt \
    ./ 2>/dev/null || echo "No pipeline log found"

# Cleanup
echo ""
echo "=========================================="
echo "🗑️  Cleaning Up"
echo "=========================================="
echo ""
read -p "Delete the VM instance to stop charges? (y/n): " DELETE_VM

if [ "$DELETE_VM" = "y" ]; then
    echo "Deleting VM..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    echo "✅ VM deleted successfully!"
else
    echo "⚠️  VM is still running and WILL INCUR CHARGES!"
    echo "   To delete later: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Results saved to: data/processed/submission.csv"
echo ""
echo "Pipeline log saved to: pipeline_log.txt"
echo ""
