import argparse
import sounddevice as sd
import soundfile as sf
from character_chat.shared.paths import INPUT_AUDIO_DIR, OUTPUT_AUDIO_DIR, PROJECT_DIR
from pathlib import Path

class AudioPlayer:
    def play(self, audio_path=OUTPUT_AUDIO_DIR / "output.wav"):
        # Load waveform and sample rate
        data, fs = sf.read(audio_path)  # waveform: [samples, channels], float32 [-1.0,1.0]

        sd.play(data, fs)
        sd.wait()

def main():
    parser = argparse.ArgumentParser(description="Audio player for playing generated responses")

    parser.add_argument(
        "--file",
        type=str,
        help="Path to audio file for playback"
    )

    args = parser.parse_args()
    file_path = (PROJECT_DIR / args.file).resolve()

    player = AudioPlayer()

    if args.file:
        player.play(audio_path=str(file_path))
        return

    while True:
        user_input = input("Enter path to audio file for playback: ")
        if user_input.lower() == "exit":
            break
        player.play(audio_path=str((INPUT_AUDIO_DIR / user_input).resolve()))

if __name__ == "__main__":
    main()