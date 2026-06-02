from pathlib import Path
import subprocess
import sys

# ---- 1. Create virtual environment ----
subprocess.run([sys.executable, "-m", "venv", "venv"])

# ---- 2. Install package in editable mode ----
# NOTE: activation is NOT needed inside Python
subprocess.run([r"venv\Scripts\python.exe", "-m", "pip", "install", "-e", "."])

# ---- 3. Set root folder ----
ROOT = Path("data")

# ---- 4. Create directory structure ----
folders = [
    ROOT / "dataset" / "wavs",
    ROOT / "output" / "wavs",
    ROOT / "output" / "text",
    ROOT / "processed" / "chunks",
    ROOT / "processed" / "separated",
    ROOT / "processed" / "transcripts",
    ROOT / "temp" / "runs",
    ROOT / "raw" / "audio",
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

print("Setup complete.")