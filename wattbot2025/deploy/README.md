# WattBot 2025 - GCP Deployment Guide

This guide will help you deploy and run WattBot on Google Cloud Platform for faster processing.

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
3. **GCP Project** created

## Quick Start

### 1. Set Environment Variables

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"  # Or your preferred zone
export INSTANCE_NAME="wattbot-vm"
```

### 2. Create GCP VM

```bash
cd deploy
chmod +x *.sh
./gcp_setup.sh
```

**Choose your configuration:**
- **With GPU** (Recommended for fastest processing): Press `y`
  - Machine: n1-standard-8 (8 vCPUs, 30GB RAM)
  - GPU: 1x NVIDIA Tesla T4
  - Cost: ~$0.50-1.00/hour
  - Estimated time: 5-10 minutes

- **CPU Only** (Budget option): Press `n`
  - Machine: c2-standard-8 (8 vCPUs, 32GB RAM)
  - Cost: ~$0.30/hour
  - Estimated time: 15-20 minutes

### 3. Upload Your Code and Data

Wait ~2 minutes for the VM to fully start, then:

```bash
./gcp_upload.sh
```

This will upload:
- Source code (`src/`, `configs/`, `run.py`)
- Environment variables (`.env`)
- Data files (`data/raw/`, `data/chunks/`)

### 4. SSH into the VM

```bash
gcloud compute ssh $INSTANCE_NAME --project=$GCP_PROJECT_ID --zone=$GCP_ZONE
```

### 5. Run the Pipeline

Once connected to the VM:

```bash
cd wattbot2025
chmod +x ../deploy/gcp_run.sh
../deploy/gcp_run.sh
```

The pipeline will run in a `tmux` session:
- **Detach**: Press `Ctrl+B`, then `D`
- **Reattach**: `tmux attach -t wattbot`
- **View log**: `tail -f pipeline_log.txt`

### 6. Download Results

From your local machine:

```bash
gcloud compute scp \
  --project=$GCP_PROJECT_ID \
  --zone=$GCP_ZONE \
  $INSTANCE_NAME:~/wattbot2025/data/processed/submission.csv \
  ./data/processed/
```

### 7. Delete the VM (Important!)

**Don't forget to delete the VM to avoid charges:**

```bash
gcloud compute instances delete $INSTANCE_NAME \
  --project=$GCP_PROJECT_ID \
  --zone=$GCP_ZONE
```

## Advanced Options

### Monitor GPU Usage

While connected to the VM:

```bash
watch -n 1 nvidia-smi
```

### Check Progress

```bash
cd ~/wattbot2025
tail -f pipeline_log.txt
```

### Run Specific Steps

```python
# SSH into VM and start Python
cd ~/wattbot2025
python3

# In Python:
from src.vector_pipeline import VectorDBPipeline

pipeline = VectorDBPipeline()

# Build index only
pipeline.build_or_load_index()

# Test on training data
pipeline.test_on_training_data(n_samples=10)

# Process test questions
pipeline.process_test_questions()
```

## Cost Estimation

| Configuration | Cost/Hour | Estimated Total | Time |
|--------------|-----------|-----------------|------|
| GPU (T4) | $0.50-1.00 | $0.10-0.20 | 5-10 min |
| CPU (c2-standard-8) | $0.30 | $0.10-0.15 | 15-20 min |

**Always delete the VM when done to avoid ongoing charges!**

## Troubleshooting

### VM Creation Fails
- Check quota limits in GCP Console
- Try a different zone
- Ensure billing is enabled

### Upload Fails
- Ensure VM is fully started (wait 2-3 minutes)
- Check instance name and zone

### Out of Memory
- Use a larger machine type: `n1-standard-16` or `n1-highmem-8`
- Edit `gcp_setup.sh` and change `MACHINE_TYPE`

### GPU Not Detected
- Ensure you selected GPU option during setup
- Check GPU quotas in your project
- Try different zone (GPU availability varies)

## Alternative: No-GPU Budget Setup

For minimal cost, use preemptible CPU instance:

```bash
# Edit gcp_setup.sh and add this flag:
--preemptible

# This reduces cost by ~70% but VM may be terminated
```

## Support

For issues, check:
- GCP Console: https://console.cloud.google.com
- Pipeline logs: `~/wattbot2025/pipeline_log.txt`
- System logs: `sudo journalctl -u google-startup-scripts`
