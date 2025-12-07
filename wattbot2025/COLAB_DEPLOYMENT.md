# 🚀 Run WattBot on Google Colab (FREE GPU!)

The **easiest** way to run WattBot - no setup, no VM costs, free GPU!

## Quick Start (2 clicks!)

1. **Open the notebook**: 
   👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arunjay4213/Wattbot/blob/main/wattbot2025/WattBot_Colab.ipynb)

2. **Enable GPU**:
   - Runtime → Change runtime type → GPU (T4) → Save

3. **Run all cells**:
   - Runtime → Run all

That's it! ✅

---

## What You Get

- ✅ **FREE GPU** (NVIDIA T4)
- ✅ **15GB RAM**
- ✅ **No disk space issues**
- ✅ **No installation** required
- ✅ **Run in browser**
- ✅ **~10-15 minutes** total runtime

---

## Step-by-Step

### 1. Open Notebook
Click the "Open in Colab" badge above or go to:
https://colab.research.google.com/github/Arunjay4213/Wattbot/blob/main/wattbot2025/WattBot_Colab.ipynb

### 2. Enable GPU (Important!)
- Click **Runtime** menu
- Select **Change runtime type**
- Hardware accelerator: **GPU**
- GPU type: **T4**
- Click **Save**

### 3. Run the Notebook
- Click **Runtime** → **Run all**
- Or run cells one by one with Shift+Enter

### 4. Enter API Key
When prompted, enter your Google API key:
- Get it from: https://makersuite.google.com/app/apikey
- Paste in the input field

### 5. Wait for Completion
- **Building index**: ~5-7 minutes
- **Processing questions**: ~2-3 minutes  
- **Total**: ~10-15 minutes

### 6. Download Results
The last cell will automatically download `submission.csv` to your computer!

---

## Timeline with GPU

| Step | Time |
|------|------|
| Clone repo | 10 seconds |
| Install dependencies | 2-3 minutes |
| Build vector index | 5-7 minutes |
| Process questions | 2-3 minutes |
| **Total** | **~10-15 minutes** |

---

## Advantages vs Other Methods

| Method | Time | Cost | Setup |
|--------|------|------|-------|
| **Colab (GPU)** | **10-15 min** | **FREE** | **None** |
| Local Mac | 1-2 hours | Free | Complex |
| GCP VM (CPU) | 30-40 min | $0.15 | Medium |
| GCP VM (GPU) | 10-15 min | $0.50 | Complex |

**Winner: Colab!** 🏆

---

## Tips

### Keep Session Active
- Colab sessions timeout after ~12 hours of inactivity
- Keep the tab open while running
- The pipeline completes in 10-15 minutes, so no problem!

### Save to Google Drive (Optional)
Uncomment the last cell to save results to your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Monitor Progress
Watch the progress bars in real-time as the pipeline runs!

### Free Tier Limits
- Colab free tier: ~12 hours per session
- GPU access: May be limited during peak times
- Our pipeline: Completes in 10-15 minutes ✅

---

## Troubleshooting

### "No GPU available"
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. GPU type → T4
4. Save
5. Runtime → Restart runtime

### "Out of memory"
- Already using T4 (15GB VRAM) - should be plenty
- If still issues, clear outputs: Edit → Clear all outputs

### "Session disconnected"
- Keep browser tab active
- Don't let computer sleep
- Pipeline is only 10-15 min, shouldn't disconnect

### "API quota exceeded"
- Gemini 2.5 Flash has high free limits
- Wait a bit or use different API key

---

## Why Colab is Perfect for This

1. **No Setup**: Zero installation, just click and run
2. **Free GPU**: T4 GPU makes embedding generation fast
3. **No Costs**: Completely free (vs $0.15-0.50 for GCP)
4. **Easy Sharing**: Share the notebook link with teammates
5. **Reproducible**: Same environment every time

---

## Next Steps

After you get your results:

1. Download `submission.csv` 
2. Check the results
3. Submit to competition!

---

## Need Help?

- **GitHub Issues**: https://github.com/Arunjay4213/Wattbot/issues
- **Full Docs**: See `deploy/` folder for other deployment options

---

**Happy WattBot-ing! 🤖⚡**
