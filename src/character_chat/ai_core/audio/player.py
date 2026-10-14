import sounddevice as sd
import soundfile as sf
from character_chat.shared.paths import INPUT_AUDIO_DIR, OUTPUT_AUDIO_DIR
from pathlib import Path

def play(audio_path=OUTPUT_AUDIO_DIR / "output.wav"):
    # Load waveform and sample rate
    data, fs = sf.read(audio_path)  # waveform: [samples, channels], float32 [-1.0,1.0]

    sd.play(data, fs)
    sd.wait()

def main():
    while True:
        user_input = input("Enter name of audio file for playback: ")
        if user_input.lower() == "exit":
            break
        play(audio_path=str((INPUT_AUDIO_DIR / user_input).resolve()))

if __name__ == "__main__":
    main()