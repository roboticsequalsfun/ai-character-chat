import torchaudio as ta
import simpleaudio as sa
import os
import time
import torch  # needed for tensor conversion

OUTPUT_FILE = "output/peppa_clone.wav"
DONE_TXT = "temp/done.txt"
ALMOSTDONE_TXT = "temp/almostdone.txt"

print("[INFO] Waiting for done.txt trigger...")

while True:
    # Wait until done.txt exists
    while not os.path.exists(ALMOSTDONE_TXT):
        time.sleep(0.5)

    try:
        # Load waveform and sample rate
        waveform, sr = ta.load(OUTPUT_FILE)  # waveform: [channels, samples], float32 [-1.0,1.0]

        # Convert to PCM16 for simpleaudio
        pcm16 = (waveform * 32767).to(torch.int16).T  # transpose to [samples, channels]

        # Interleave and convert to bytes
        audio_bytes = pcm16.contiguous().numpy().tobytes()

        # Play audio
        wave_obj = sa.WaveObject(audio_bytes, num_channels=pcm16.shape[1], bytes_per_sample=2, sample_rate=sr)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        print("[INFO] Playback finished.")
        if not os.path.exists(DONE_TXT):
                with open(DONE_TXT, "w") as f:
                    f.write("trigger")

    except Exception as e:
        print(f"[ERROR] Failed to play audio: {e}")

    # Remove done.txt so loop can wait for the next trigger
    os.remove(ALMOSTDONE_TXT)
    print("[INFO] Waiting for next done.txt trigger...")
