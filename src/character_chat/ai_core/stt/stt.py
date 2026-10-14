import whisper
from character_chat.shared.paths import INPUT_AUDIO_DIR
from character_chat.shared.file_utils import load_config

class STT:
    def __init__(self, model_size="large"):
        # Load larger model for accuracy (small is fast, large is most accurate)
        self.model_size = model_size
        self.model = whisper.load_model(model_size)  # options: tiny, base, small, medium, large

    def transcribe(self, audio_path=None):
        result = self.model.transcribe(audio_path)
        return result['text']

def main():
    config = load_config("app")

    stt = STT(model_size=config.get("stt", {}).get("model", "large"))

    while True:
        user_input = input("Enter name of audio file for transcription (type exit to escape): ")
        if user_input.lower() == "exit":
            break
        result = stt.transcribe(audio_path=str((INPUT_AUDIO_DIR / user_input).resolve()))
        print(result)

if __name__ == "__main__":
    main()
