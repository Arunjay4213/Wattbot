# 🎉 You're Already on the VM! Let's Deploy WattBot

You're connected to: **ml-research-vm** (35.224.94.133)

## 🚀 Quick Deploy - Just 3 Steps

### Step 1: Set Up the VM (2 minutes)

**In your VM terminal, copy-paste this entire block:**

```bash
# Create directory structure
mkdir -p ~/wattbot2025/data/{raw,chunks,processed,cache}
cd ~/wattbot2025

# Update system
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv -qq

# Install Python dependencies
python3 -m pip install --upgrade pip

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

echo "✅ Setup complete!"
```

---

### Step 2: Upload Files (2 minutes)

**Option A - From Cloud Shell (Easiest):**

1. Open a NEW browser tab: https://console.cloud.google.com/?project=wattbot-2025&cloudshell=true

2. In Cloud Shell, upload your folder:
   - Click **⋮** → **Upload** → **Upload folder**
   - Select: `/Users/revanshphull/watt/wattbot2025`

3. After upload, run in Cloud Shell:
```bash
cd wattbot2025

gcloud compute scp --recurse \
  --zone=us-central1-f \
  src configs run.py .env \
  ml-research-vm:~/wattbot2025/

gcloud compute scp --recurse \
  --zone=us-central1-f \
  data/raw data/chunks \
  ml-research-vm:~/wattbot2025/data/
```

**Option B - Via SSH Window Upload:**

In your current SSH window:
- Look for **⚙️** (gear icon) or **⋮** menu
- Click **Upload file**
- Upload these files/folders to `~/wattbot2025/`:
  - `src/` folder
  - `configs/` folder
  - `data/raw/` folder
  - `data/chunks/` folder
  - `run.py`
  - `.env`

---

### Step 3: Run the Pipeline (30-40 minutes)

**Back in your VM terminal (where you're connected):**

```bash
cd ~/wattbot2025

# Verify files are there
ls -la

# Run the pipeline!
python3 run.py 2>&1 | tee pipeline_log.txt
```

That's it! The pipeline will now run.

---

## 📊 What You'll See

```
wattbot!!!!
🚀 Initializing Vector DB Pipeline...
Initializing Hybrid Retriever...

Step 1: Building Vector Index...
📦 Building vector index...
📚 Loaded 4498 chunks from 32 documents
Indexing 4498 chunks for hybrid search...
Batches:   0%|          | 0/141 [00:00<?, ?it/s]
```

### Timeline (CPU - No GPU):
- **Building index**: ~20-30 minutes
- **Testing**: ~2 minutes
- **Processing questions**: ~15-20 minutes
- **Total**: ~30-50 minutes

You can leave this running and check back later!

---

## 📥 After Completion - Download Results

### Option 1: View in VM
```bash
cat ~/wattbot2025/data/processed/submission.csv
```

### Option 2: Download via Cloud Shell
In Cloud Shell (new tab):
```bash
gcloud compute scp \
  --zone=us-central1-f \
  ml-research-vm:~/wattbot2025/data/processed/submission.csv \
  ./

# Then download from Cloud Shell:
# Click ⋮ → Download → submission.csv
```

### Option 3: Download via SSH Window
- In SSH window, click **⚙️** → **Download file**
- Enter: `wattbot2025/data/processed/submission.csv`

---

## 🔍 Monitor Progress

### Check if it's running:
```bash
ps aux | grep python3
```

### View live output:
```bash
tail -f ~/wattbot2025/pipeline_log.txt
```

### Check system resources:
```bash
htop  # Press q to exit
```

---

## 💡 Tips

### Run in Background (Optional)
If you want to disconnect and let it run:
```bash
# Install tmux
sudo apt-get install -y tmux

# Start pipeline in tmux
cd ~/wattbot2025
tmux new -s wattbot
python3 run.py 2>&1 | tee pipeline_log.txt

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -s wattbot
```

### Stop the Pipeline
If you need to stop it:
```bash
# Press Ctrl+C
# Or find and kill the process:
pkill -f "python3 run.py"
```

---

## ✅ Current Status

- ✅ VM Running (ml-research-vm)
- ✅ Connected via SSH
- ⏳ Waiting for: Dependencies installation
- ⏳ Waiting for: File upload
- ⏳ Waiting for: Pipeline execution

---

**Next:** Run the setup commands above in your VM terminal!
