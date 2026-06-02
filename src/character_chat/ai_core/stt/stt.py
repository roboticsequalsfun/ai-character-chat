import whisper
import argparse
from character_chat.shared.paths import PROJECT_DIR, INPUT_AUDIO_DIR

class STT:
    def __init__(self, model_size="large"):
        # Load larger model for accuracy (small is fast, large is most accurate)
        self.model_size = model_size
        self.model = whisper.load_model(model_size)  # options: tiny, base, small, medium, large

    def transcribe(self, audio_path=None):
        result = self.model.transcribe(audio_path)
        return result['text']

def main():
    parser = argparse.ArgumentParser(description="Speech-to-text runner")

    parser.add_argument(
        "--file",
        type=str,
        help="Path to audio file for transcription"
    )

    args = parser.parse_args()
    file_path = (PROJECT_DIR / args.file).resolve()

    stt = STT()

    if args.file:
        result = stt.transcribe(audio_path=str(file_path))
        print(result)
        return

    while True:
        user_input = input("Enter path to audio file for transcription: ")
        if user_input.lower() == "exit":
            break
        result = stt.transcribe(audio_path=str((INPUT_AUDIO_DIR / user_input).resolve()))
        print(result)

if __name__ == "__main__":
    main()
