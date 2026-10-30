from pydub import AudioSegment
from pathlib import Path
import sounddevice as sd
import soundfile as sf
from character_chat.shared.paths import INPUT_AUDIO_DIR, OUTPUT_AUDIO_DIR
from character_chat.shared.logger import log
from character_chat.shared.file_utils import path_exists

def play(audio_path: str | Path = OUTPUT_AUDIO_DIR / "output.wav"):
    """Play audio from file"""

    path_exists(audio_path)

    log("Playing audio...")
    data, fs = sf.read(audio_path)

    sd.play(data, fs)
    sd.wait()

if __name__ == "__main__":
    # Since we are entering by calling the script we are in testing mode
    while True:
        # We ask the user for the file
        user_input = input("Enter name of audio file for playback (or 'exit' to quit): ")

        # If the input is exit we stop the script
        if user_input.lower() == "exit":
            break

        # Else we play the audio
        play(audio_path=str(INPUT_AUDIO_DIR / user_input))
