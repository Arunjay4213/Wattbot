# WattBot 2025 - Quick GCP Setup

## Your Project: wattbot-2025

## Option 1: Use Google Cloud Console (Web Interface) - EASIEST

### Step 1: Open Cloud Console
Go to: https://console.cloud.google.com/?project=wattbot-2025

### Step 2: Create VM via Web
1. Go to **Compute Engine** > **VM Instances**
2. Click **CREATE INSTANCE**
3. Configure:
   - **Name**: `wattbot-vm`
   - **Region**: `us-central1`
   - **Zone**: `us-central1-a`
   - **Machine type**: `n1-standard-8` (8 vCPUs, 30GB RAM)
   - **GPU**: Click "Add GPU" → Select `NVIDIA T4` (1 GPU)
   - **Boot disk**:
     - Click "CHANGE"
     - Select "Deep Learning on Linux"
     - Choose "Deep Learning VM with CUDA 11.8 M122"
     - Size: 100 GB
   - **Firewall**: Allow HTTP/HTTPS traffic

4. Click **CREATE**

### Step 3: Connect via Browser
1. Wait 2-3 minutes for VM to start
2. Click **SSH** button next to your VM
3. A browser window will open with terminal access

### Step 4: Upload Files (via Cloud Console)
1. In Cloud Console, go to **Cloud Storage** > **Browser**
2. Create a bucket: `wattbot-2025-data`
3. Upload these folders:
   - `wattbot2025/src/`
   - `wattbot2025/configs/`
   - `wattbot2025/data/`
   - `wattbot2025/run.py`
   - `wattbot2025/.env`

### Step 5: Download and Run (in SSH terminal)
```bash
# Download from Cloud Storage
gsutil -m cp -r gs://wattbot-2025-data/* ~/wattbot2025/

# Or use Cloud Shell editor to upload directly
# Click "Upload File" in the SSH window menu

# Once files are uploaded, run:
cd ~/wattbot2025
python3 -m pip install -r requirements.txt
python3 run.py
```

---

## Option 2: Install gcloud CLI (Recommended for Automation)

### Install gcloud CLI:

**macOS:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

**Or via Homebrew:**
```bash
brew install --cask google-cloud-sdk
gcloud init
```

### After Installing gcloud:

```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project wattbot-2025

# Create VM with GPU
gcloud compute instances create wattbot-vm \
  --project=wattbot-2025 \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# Wait 2 minutes, then upload files
cd /Users/revanshphull/watt/wattbot2025

gcloud compute scp --recurse \
  --zone=us-central1-a \
  src configs run.py .env requirements.txt \
  wattbot-vm:~/wattbot2025/

gcloud compute scp --recurse \
  --zone=us-central1-a \
  data/raw data/chunks \
  wattbot-vm:~/wattbot2025/data/

# SSH into VM
gcloud compute ssh wattbot-vm --zone=us-central1-a

# On the VM, install dependencies and run
cd ~/wattbot2025
python3 -m pip install -r requirements.txt
python3 run.py
```

---

## Option 3: Use Cloud Shell (No Installation Needed)

### Step 1: Open Cloud Shell
Go to: https://console.cloud.google.com/?cloudshell=true&project=wattbot-2025

### Step 2: Upload Your Code
In Cloud Shell, click the 3 dots (...) → "Upload" → Upload your wattbot2025 folder

### Step 3: Create VM from Cloud Shell
```bash
cd wattbot2025

gcloud compute instances create wattbot-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# Upload files
gcloud compute scp --recurse \
  --zone=us-central1-a \
  src configs run.py .env requirements.txt \
  wattbot-vm:~/wattbot2025/

gcloud compute scp --recurse \
  --zone=us-central1-a \
  data/raw data/chunks \
  wattbot-vm:~/wattbot2025/data/

# Connect and run
gcloud compute ssh wattbot-vm --zone=us-central1-a
cd ~/wattbot2025
python3 -m pip install -r requirements.txt
python3 run.py
```

---

## After Running - Download Results

### Via Cloud Console:
1. SSH into VM
2. Run: `cat ~/wattbot2025/data/processed/submission.csv`
3. Copy the output

### Via gcloud:
```bash
gcloud compute scp \
  --zone=us-central1-a \
  wattbot-vm:~/wattbot2025/data/processed/submission.csv \
  ./data/processed/
```

---

## IMPORTANT: Delete VM When Done!

### Via Cloud Console:
1. Go to **Compute Engine** > **VM Instances**
2. Check the box next to `wattbot-vm`
3. Click **DELETE**

### Via gcloud:
```bash
gcloud compute instances delete wattbot-vm --zone=us-central1-a
```

---

## Cost Estimate
- **n1-standard-8 + T4 GPU**: ~$0.50-1.00/hour
- **Expected runtime**: 5-10 minutes
- **Total cost**: ~$0.10-0.20

**Don't forget to delete the VM to avoid ongoing charges!**
