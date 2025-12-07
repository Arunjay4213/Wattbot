# 🚀 WattBot GCP Deployment - START HERE

## The EASIEST Way - 3 Steps, 10 Minutes Total

### Step 1: Open Google Cloud Shell (30 seconds)

**Click this link:** 👉 **https://console.cloud.google.com/?project=wattbot-2025&cloudshell=true**

This opens Cloud Shell - a free browser-based terminal with all tools pre-installed.

---

### Step 2: Upload Your Code (2 minutes)

In Cloud Shell (the black terminal at the bottom of the page):

**Click the ⋮ (three dots) menu → Upload → Upload folder**

Select your wattbot2025 folder: `/Users/revanshphull/watt/wattbot2025`

Wait for upload to complete (~2 minutes).

---

### Step 3: Run ONE Command (7-10 minutes)

In Cloud Shell, copy-paste this ENTIRE command:

```bash
cd wattbot2025 && chmod +x deploy/cloud_shell_deploy.sh && ./deploy/cloud_shell_deploy.sh
```

**That's it!** The script will:
- ✅ Create GPU VM
- ✅ Upload your files
- ✅ Install dependencies
- ✅ Run pipeline with Gemini
- ✅ Download results
- ✅ Delete VM (to stop charges)

---

## What You'll See:

```
==========================================
WattBot 2025 - Automated Cloud Deployment
==========================================

This will:
  1. Create a GPU VM instance
  2. Upload your code and data
  3. Install dependencies
  4. Run the pipeline
  5. Download results
  6. Delete the VM

Estimated time: 10-15 minutes
Estimated cost: $0.10-0.25

[... automated deployment starts ...]
```

---

## After Completion:

Your results will be in:
- `data/processed/submission.csv` (in Cloud Shell)

To download to your Mac:
1. In Cloud Shell, click ⋮ → Download
2. Enter: `wattbot2025/data/processed/submission.csv`

---

## Alternative: Manual Web UI Method

If Cloud Shell upload fails, see `deploy/DEPLOY_NOW.md` for point-and-click web UI instructions.

---

## 💰 Cost:

- GPU VM: ~$0.50/hour
- Runtime: 10-15 minutes
- **Total: ~$0.10-0.25**

The script automatically deletes the VM when done!

---

## 🆘 Having Issues?

### Cloud Shell upload too slow?
Use Cloud Shell's git clone instead:
```bash
# In Cloud Shell:
git clone YOUR_REPO_URL
cd wattbot2025
./deploy/cloud_shell_deploy.sh
```

### Upload folder not working?
Upload files individually:
```bash
# In Cloud Shell:
mkdir -p wattbot2025
cd wattbot2025
# Then use "Upload file" button to upload:
# - run.py
# - .env
# - Folders: src/, configs/, data/
```

### Script fails?
Check `deploy/DEPLOY_NOW.md` for manual step-by-step instructions.

---

## ✅ That's It!

**Ready? Click here to start:**
👉 **https://console.cloud.google.com/?project=wattbot-2025&cloudshell=true**

Then upload your folder and run the script. Done!
