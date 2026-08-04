# Kaggle notebook: full offline pipeline (Qwen2.5-VL-7B caption+OCR, OWLv2, CLIP, Whisper) on GPU T4x2.
#
# Copy each "# %%" block below into its own Kaggle notebook cell, in order.
# Before running: Notebook settings -> Accelerator = GPU T4 x2, Internet = On.
# See docs/KAGGLE_OFFLINE_GUIDE.md for the full walkthrough and troubleshooting.

# %% [1] confirm GPU is attached
# !nvidia-smi

# %% [2] clone the repo (edit REPO_URL/BRANCH if you forked it)
import os

REPO_URL = "https://github.com/nguyennhan2006/AIC2026_Nam_thang_ay.git"
BRANCH = "Data_Section"

# If the repo is PRIVATE: add a GitHub token as a Kaggle Secret named GITHUB_TOKEN
# (Add-ons -> Secrets), then uncomment the next 3 lines instead of using REPO_URL as-is.
# from kaggle_secrets import UserSecretsClient
# token = UserSecretsClient().get_secret("GITHUB_TOKEN")
# REPO_URL = f"https://{token}@github.com/nguyennhan2006/AIC2026_Nam_thang_ay.git"

os.system(f"git clone -b {BRANCH} {REPO_URL} /kaggle/working/repo")
os.chdir("/kaggle/working/repo")
print(os.getcwd())

# %% [3] install deps -- do NOT reinstall torch, Kaggle already ships a CUDA-linked build
# !pip install -e ".[api,faiss]" -q
# !pip install -U "transformers>=4.49,<5" "accelerate>=0.34,<2" "qwen-vl-utils>=0.0.8,<1" "pillow>=10,<12" -q

# %% [4] copy the uploaded video dataset into storage/raw/videos
# Attach your video as a Kaggle Dataset first (Add Data), then fix the dataset slug below.
import pathlib
import shutil

DATASET_SLUG = "aic2026-l16-v001"  # <-- change to your actual Kaggle dataset folder name
VIDEO_FILENAME = "L16_V001.mp4"  # canonical id: ^L\d{2}_V\d{3}$

src = pathlib.Path(f"/kaggle/input/{DATASET_SLUG}/{VIDEO_FILENAME}")
dst = pathlib.Path("storage/raw/videos") / VIDEO_FILENAME
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
print(dst, dst.stat().st_size, "bytes")

# %% [5] start the GPU worker in-process (same architecture as Vast.ai, just co-located here)
import subprocess
import time
import urllib.request

os.environ["AIC_GPU_PROVIDER"] = "transformers"
os.environ["AIC_GPU_API_KEY"] = "kaggle-local-key"
os.environ["AIC_CAPTION_MODEL"] = "Qwen/Qwen2.5-VL-7B-Instruct"
os.environ["AIC_GPU_DEVICE"] = "0"  # OWLv2/CLIP/Whisper pin to GPU 0; Qwen shards across both via device_map=auto

worker = subprocess.Popen(
    ["uvicorn", "offline.worker:app", "--host", "127.0.0.1", "--port", "8010"],
    env=os.environ.copy(),
)
for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8010/v1/health", timeout=2)
        print("worker up")
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("worker did not come up -- check worker logs")

# %% [6] run the offline pipeline against the uploaded video
# First call downloads ~16GB of model weights (Qwen2.5-VL-7B + OWLv2 + CLIP + Whisper) --
# this can take 10-20+ minutes depending on Kaggle's network speed. Be patient.
os.environ["AIC_OFFLINE_PROVIDER"] = "remote"
os.environ["AIC_GPU_URL"] = "http://127.0.0.1:8010"
os.environ["AIC_GPU_TIMEOUT_SEC"] = "300"
os.system("python -m offline run")

# %% [7] build the local index artifact (optional, keeps parity with the local workflow)
os.system("python -m offline index --encoder remote")

# %% [8] validate the export and package everything for download
os.system("python -m datasection.cli storage/exports")
os.system("python -m scripts.preflight")

shutil.make_archive("/kaggle/working/offline_output", "zip", root_dir=".", base_dir="storage")
print("done -> /kaggle/working/offline_output.zip (download it from the notebook's Output panel)")

# %% [9] stop the worker (frees GPU memory; optional cleanup)
worker.terminate()
