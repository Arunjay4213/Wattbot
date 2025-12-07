# Deploy WattBot to Your Existing VM

## Your VM Info:
- **Name**: ml-research-vm
- **IP**: 35.224.94.133
- **Zone**: us-central1-f
- **Specs**: 4 vCPUs, 16 GB RAM (CPU only - no GPU)
- **Status**: ✅ Running

---

## 🚀 EASIEST METHOD - Use Cloud Shell (5 minutes)

### Step 1: Open Cloud Shell
Click here: **https://console.cloud.google.com/?project=wattbot-2025&cloudshell=true**

### Step 2: Upload Your Code to Cloud Shell

In Cloud Shell, click **⋮** (three dots) → **Upload** → **Upload folder**

Select: `/Users/revanshphull/watt/wattbot2025`

(Wait 2-3 minutes for upload)

### Step 3: Deploy to Your VM

Copy-paste these commands in Cloud Shell:

```bash
cd wattbot2025

# Set up directories on VM
gcloud compute ssh ml-research-vm \
  --zone=us-central1-f \
  --command="mkdir -p ~/wattbot2025/data/{raw,chunks,processed,cache}"

# Upload code
gcloud compute scp --recurse \
  --zone=us-central1-f \
  src configs run.py .env \
  ml-research-vm:~/wattbot2025/

# Upload data
gcloud compute scp --recurse \
  --zone=us-central1-f \
  data/raw data/chunks \
  ml-research-vm:~/wattbot2025/data/

# Install dependencies and run
gcloud compute ssh ml-research-vm --zone=us-central1-f << 'ENDSSH'
cd ~/wattbot2025

# Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install pandas==2.0.3 numpy==1.24.3 \
    python-dotenv==1.0.0 pyyaml==6.0.1 tqdm==4.66.1 \
    google-generativeai==0.3.2 sentence-transformers==2.2.2 \
    rank-bm25==0.2.2 torch==2.1.2 transformers==4.35.0 \
    scikit-learn==1.3.0

# Run pipeline
python3 run.py
ENDSSH
```

### Step 4: Download Results

After the pipeline completes (15-30 minutes), download results:

```bash
gcloud compute scp \
  --zone=us-central1-f \
  ml-research-vm:~/wattbot2025/data/processed/submission.csv \
  ./
```

Then download from Cloud Shell to your Mac:
- Click **⋮** → **Download**
- Enter: `submission.csv`

---

## ⚠️ Important Notes

### Performance
Your VM has **CPU only** (no GPU). The pipeline will take:
- **Embedding indexing**: ~20-30 minutes
- **Question processing**: ~15-20 minutes
- **Total**: ~30-50 minutes

### To Speed Up (Optional)
You could add a GPU to your existing VM:
1. Stop the VM
2. Edit → Add GPU (NVIDIA Tesla T4)
3. Restart

This would reduce time to 10-15 minutes, but costs more.

### Cost
- n2-standard-4: ~$0.15/hour
- Expected runtime: ~1 hour
- Total: ~$0.15

---

## 🔧 Alternative: Direct SSH Method

If you prefer, you can SSH directly from your Mac terminal:

```bash
# SSH to VM (one-time setup will ask for SSH key generation)
gcloud compute ssh ml-research-vm \
  --project=wattbot-2025 \
  --zone=us-central1-f

# Then manually upload files using the Web UI or scp
```

---

## 📊 Monitor Progress

While pipeline runs:

```bash
# In another Cloud Shell tab or SSH session
gcloud compute ssh ml-research-vm --zone=us-central1-f

# Then on the VM:
cd ~/wattbot2025
tail -f pipeline_log.txt  # if you redirected output
# or
ps aux | grep python  # check if running
```

---

## ✅ Quick Checklist

- [ ] Open Cloud Shell
- [ ] Upload wattbot2025 folder to Cloud Shell
- [ ] Run the deployment commands above
- [ ] Wait for pipeline to complete (~30-50 min)
- [ ] Download submission.csv
- [ ] (Optional) Stop VM to save costs

---

**Ready to start?**
👉 Click: **https://console.cloud.google.com/?project=wattbot-2025&cloudshell=true**
