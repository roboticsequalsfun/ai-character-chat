import os
import time
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import sys
import torch

# -----------------------------
# Configuration
# -----------------------------
AUDIO_FOLDER = "wavs"                      # Folder with reference audio chunks
OUTPUT_FILE = "output/peppa_clone.wav"     # Where to save generated audio
TEXT_FILE = "cloner_text/response.txt"     # File with text to speak
DONE_FOLDER = "temp"                       # Folder for done.txt trigger
DONE_TXT = os.path.join(DONE_FOLDER, "almostdone.txt")
EXIT_TXT = os.path.join(DONE_FOLDER, "exit.txt")

os.makedirs(DONE_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# -----------------------------
# Initialize Chatterbox
# -----------------------------
print("[INFO] Initializing Chatterbox TTS...")
cb = ChatterboxTTS.from_pretrained(device="cpu")  # CPU mode

# -----------------------------
# Pick a valid audio chunk as prompt
# -----------------------------
audio_prompt_path = None
for file_name in sorted(os.listdir(AUDIO_FOLDER)):
    file_path = os.path.join(AUDIO_FOLDER, file_name)
    if file_name.lower().endswith((".wav", ".mp3")) and os.path.getsize(file_path) > 0:
        try:
            # Test loading to catch corrupted files
            ta.load(file_path)
            audio_prompt_path = file_path
            print(f"[INFO] Using {file_name} as audio prompt")
            break
        except Exception as e:
            print(f"[WARN] Skipping corrupted audio file: {file_name} ({e})")

if not audio_prompt_path:
    raise RuntimeError("No valid audio prompt found in folder!")

if os.path.exists(DONE_TXT):
    os.remove(DONE_TXT)

# -----------------------------
# Main loop
# -----------------------------
print("[INFO] Waiting for trigger file (response.txt) or exit.txt...")

while True:
    try:
        # Check for exit signal
        if os.path.exists(EXIT_TXT):
            print("[INFO] exit.txt found. Exiting gracefully...")
            os.remove(EXIT_TXT)
            sys.exit(0)

        if os.path.exists(TEXT_FILE):
            # Read text to speak
            try:
                with open(TEXT_FILE, "r", encoding="utf-8") as f:
                    TEXT_TO_SPEAK = f.read().strip()
            except Exception:
                TEXT_TO_SPEAK = "Hello! I'm Peppa Pig, and I love jumping in muddy puddles!"
                print(f"[WARN] {TEXT_FILE} not found, using default text.")
            # Generate speech
            print("[INFO] Generating speech...")
            wav = cb.generate(TEXT_TO_SPEAK, audio_prompt_path=audio_prompt_path)

            # Convert directly to tensor for torchaudio
            wav_tensor = wav.cpu()  # wav is already a torch.Tensor
            ta.save(OUTPUT_FILE, wav_tensor, cb.sr)
            print(f"[DONE] Cloned voice saved to {OUTPUT_FILE}")


            # Create done.txt if it doesn't exist
            if not os.path.exists(DONE_TXT):
                with open(DONE_TXT, "w") as f:
                    f.write("trigger")
                print(f"[INFO] Created done file: {DONE_TXT}")

            # Delete trigger file to reset loop
            print("[INFO] Waiting for next trigger (response.txt) or exit.txt...")
            os.remove(TEXT_FILE)
        else:
            time.sleep(0.5)

    except Exception as e:
        print(f"[ERROR] {e}, continuing loop...")
        time.sleep(0.5)
