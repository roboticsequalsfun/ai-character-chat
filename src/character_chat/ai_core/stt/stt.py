import whisper
from character_chat.shared.paths import INPUT_AUDIO_DIR
from character_chat.shared.file_utils import load_config, path_exists
from character_chat.shared.logger import log
from pathlib import Path

class STT:
    def __init__(self, model_size: str = "large") -> None:
        """Convert audio to text"""

        match model_size:
            case "tiny":
                pass
            case "base":
                pass
            case "small":
                pass
            case "medium":
                pass
            case "large":
                pass
            case "turbo":
                pass
            case _:
                log(f"Unkown model {model_size} for whisper.", "ERROR")

        # Load the model
        # Options are tiny, base, small, medium, large, and turbo
        log("Loading STT model (This might take a while)...")
        self.model = whisper.load_model(model_size)

        if self.model is None:
            log("Failed to load model.", "ERROR")

    def transcribe(self, audio_path: str | Path = None) -> str:
        """Transcribe the audio and return result"""

        path_exists(audio_path)

        log("Transcribing...", "INFO")
        result = self.model.transcribe(audio_path)

        return result['text']


if __name__ == "__main__":
    # Since we are entering by calling the script we are in testing mode
    
    # Load the config
    config = load_config("app")

    # We load the model and pass the model size
    stt = STT(model_size=config.get("stt", {}).get("model", "large"))

    while True:
        # Now we ask the user for the file
        user_input = input("Enter name of audio file for transcription (or 'exit' to quit): ")

        # If the input is exit we stop the script
        if user_input.lower() == "exit":
            break

        # Else we transcribe the audio file and print the result
        result = stt.transcribe(audio_path=str((INPUT_AUDIO_DIR / user_input).resolve()))
        print("Transcription: ", result)
