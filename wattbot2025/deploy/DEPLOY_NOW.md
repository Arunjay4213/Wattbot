# 🚀 Deploy WattBot to GCP - Step-by-Step Guide

## Your Project: **wattbot-2025**

Follow these steps to get WattBot running on Google Cloud in ~10 minutes.

---

## Step 1: Open Google Cloud Console (1 minute)

**Click this link:**
👉 **https://console.cloud.google.com/?project=wattbot-2025**

---

## Step 2: Create VM Instance (3 minutes)

1. Click **☰ menu** → **Compute Engine** → **VM instances**

2. Click **CREATE INSTANCE**

3. Fill in these settings:

   **Basic Configuration:**
   - **Name**: `wattbot-vm`
   - **Region**: `us-central1`
   - **Zone**: `us-central1-a`

   **Machine Configuration:**
   - Click **GENERAL PURPOSE**
   - **Series**: N1
   - **Machine type**: `n1-standard-8` (8 vCPUs, 30 GB)

   **GPU:**
   - Scroll down and click **CPU PLATFORM AND GPU**
   - Click **ADD GPU**
   - **GPU type**: NVIDIA Tesla T4
   - **Number of GPUs**: 1

   **Boot Disk:**
   - Click **CHANGE**
   - Click **Marketplace** tab
   - Search for: "Deep Learning VM"
   - Select: **Deep Learning VM for PyTorch 1.13 with CUDA 11.3**
   - **Boot disk type**: Standard persistent disk
   - **Size**: 100 GB
   - Click **SELECT**

   **Firewall:**
   - ✅ Allow HTTP traffic
   - ✅ Allow HTTPS traffic

4. Click **CREATE** at the bottom

5. **Wait 2-3 minutes** for the green checkmark ✅

---

## Step 3: Connect to VM (30 seconds)

1. Find your VM instance `wattbot-vm` in the list

2. Click the **SSH** button next to it

3. A browser terminal will open

---

## Step 4: Prepare VM (2 minutes)

In the SSH terminal that just opened, copy-paste these commands:

```bash
# Create directory structure
mkdir -p ~/wattbot2025/data/{raw,chunks,processed,cache}
cd ~/wattbot2025

# Install Python dependencies
python3 -m pip install --upgrade pip --quiet

python3 -m pip install pandas==2.0.3 numpy==1.24.3 \
    python-dotenv==1.0.0 pyyaml==6.0.1 tqdm==4.66.1 \
    google-generativeai==0.3.2 sentence-transformers==2.2.2 \
    rank-bm25==0.2.2 torch==2.1.2 transformers==4.35.0 \
    scikit-learn==1.3.0 --quiet

echo "✅ VM is ready!"
```

---

## Step 5: Upload Your Files (2 minutes)

### Method A: Use Cloud Shell (EASIEST)

1. In Cloud Console, click the **>_** icon (top right) to open Cloud Shell

2. Click **⋮** (three dots) → **Upload** → **Upload folder**

3. Select your `/Users/revanshphull/watt/wattbot2025` folder

4. After upload completes, in Cloud Shell run:

```bash
cd ~/wattbot2025

# Copy files to VM
gcloud compute scp --recurse \
  --zone=us-central1-a \
  src configs run.py .env requirements.txt \
  wattbot-vm:~/wattbot2025/

gcloud compute scp --recurse \
  --zone=us-central1-a \
  data/raw data/chunks \
  wattbot-vm:~/wattbot2025/data/
```

### Method B: Manual Upload (if Method A doesn't work)

1. In the VM SSH window, click **⚙️ (gear icon)** → **Upload file**

2. Upload these files/folders one by one:
   - Upload `src/` folder
   - Upload `configs/` folder
   - Upload `data/raw/` folder
   - Upload `data/chunks/` folder
   - Upload `run.py`
   - Upload `.env`

---

## Step 6: Run the Pipeline! (5-10 minutes)

In the VM SSH terminal:

```bash
cd ~/wattbot2025

# Verify files are there
ls -la

# Run the pipeline
python3 run.py
```

**The pipeline will now run!** You'll see:
- Step 1: Building Vector Index (5-7 minutes with GPU)
- Step 2: Testing on Training Data (~1 minute)
- Step 3: You'll be asked if you want to process test set - type `y`

**You can safely close your browser** - the VM will keep running!

To reconnect later: Click SSH button again in Cloud Console

---

## Step 7: Download Results (1 minute)

### After pipeline completes:

**Method A: Via Cloud Shell**
```bash
# In Cloud Shell
gcloud compute scp \
  --zone=us-central1-a \
  wattbot-vm:~/wattbot2025/data/processed/submission.csv \
  ~/submission.csv

# Then download from Cloud Shell to your Mac
# Click ⋮ → Download file → submission.csv
```

**Method B: Via SSH Window**
```bash
# In VM SSH terminal
cat ~/wattbot2025/data/processed/submission.csv

# Copy the output and save locally
```

---

## Step 8: DELETE THE VM! ⚠️ IMPORTANT

**Don't forget this step to avoid charges!**

1. Go to **Compute Engine** → **VM instances**

2. Check the box next to `wattbot-vm`

3. Click **DELETE** (trash icon at top)

4. Click **DELETE** again to confirm

---

## 💰 Cost Estimate

- **Machine**: n1-standard-8 + Tesla T4 GPU
- **Rate**: ~$0.50-1.00 per hour
- **Expected runtime**: 10-15 minutes
- **Total cost**: **~$0.10 - $0.25**

---

## 🆘 Troubleshooting

### "Quota exceeded" error
- Go to **IAM & Admin** → **Quotas**
- Request increase for GPUs in us-central1

### GPU not available
- Try zone `us-west1-b` instead
- Or use CPU-only: skip the GPU step (will take 20-30 minutes)

### Upload failed
- Use Cloud Shell method instead
- Make sure VM is fully started (green checkmark)

### Can't SSH
- Wait 2-3 minutes after VM creation
- Check firewall rules allow SSH

---

## 📊 Monitor Progress

While pipeline runs, you can monitor:

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Check progress:**
```bash
tail -f ~/wattbot2025/pipeline_log.txt  # if you started with logging
```

---

## ✅ Quick Checklist

- [ ] Opened Cloud Console
- [ ] Created VM with GPU
- [ ] SSH'd into VM
- [ ] Installed dependencies
- [ ] Uploaded files
- [ ] Started pipeline (`python3 run.py`)
- [ ] Downloaded results
- [ ] **DELETED THE VM**

---

**Ready to start? Open this link now:**
👉 **https://console.cloud.google.com/?project=wattbot-2025**
