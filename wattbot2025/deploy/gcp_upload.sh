#!/bin/bash
# Upload WattBot code and data to GCP VM

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-wattbot-2025}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-wattbot-vm}"

echo "======================================"
echo "Uploading WattBot to GCP VM"
echo "======================================"
echo ""

# Check if instance exists
if ! gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE &> /dev/null; then
    echo "Error: Instance '$INSTANCE_NAME' not found."
    echo "Please run ./deploy/gcp_setup.sh first"
    exit 1
fi

echo "Uploading files to $INSTANCE_NAME..."
echo ""

# Create remote directory
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --command="mkdir -p ~/wattbot2025"

# Upload code
echo "Uploading source code..."
gcloud compute scp --recurse \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    src configs run.py .env \
    $INSTANCE_NAME:~/wattbot2025/

# Upload data
echo "Uploading data files..."
gcloud compute scp --recurse \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    data/raw data/chunks \
    $INSTANCE_NAME:~/wattbot2025/data/

# Create processed directory
gcloud compute ssh $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --command="mkdir -p ~/wattbot2025/data/processed ~/wattbot2025/data/cache"

echo ""
echo "✓ Upload complete!"
echo ""
echo "Next steps:"
echo "1. SSH into the VM:"
echo "   gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE"
echo ""
echo "2. Run the pipeline:"
echo "   cd wattbot2025"
echo "   python3 run.py"
echo ""
echo "3. Download results when complete:"
echo "   gcloud compute scp --project=$PROJECT_ID --zone=$ZONE $INSTANCE_NAME:~/wattbot2025/data/processed/submission.csv ."
