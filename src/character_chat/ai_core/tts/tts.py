import torch
import argparse
from pathlib import Path
from character_chat.shared.paths import MODELS_DIR, OUTPUT_AUDIO_DIR
from TTS.utils.synthesizer import Synthesizer
import warnings

class TextToSpeech:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODELS_DIR / "tts" / "glow_tts"
        
        model_path = Path(model_path)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Check if a GPU is available for faster inference
        USE_CUDA = torch.cuda.is_available()

        # Initialize the synthesizer with your custom model paths
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path / "best_model.pth",
            tts_config_path=model_path / "config.json",
            use_cuda=USE_CUDA
        )
            
    def synthesize(self, text="Hello world", output_path=None):
        if output_path is None:
            output_path = OUTPUT_AUDIO_DIR / "output.wav"

        output_path = Path(output_path)

        wav = self.synthesizer.tts(text)

        # Save the generated raw waveform array directly to a file
        self.synthesizer.save_wav(wav, output_path)

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text using a trained TTS model.")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the generated audio file",
        default=str(OUTPUT_AUDIO_DIR / "output.wav")
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained TTS model directory",
        default=str(MODELS_DIR / "tts" / "glow_tts_speaker_0")
    )

    args = parser.parse_args()

    args.model_path = Path(args.model_path)
    args.output_path = Path(args.output_path)

    tts = TextToSpeech(model_path=args.model_path)

    while True:
        user_input = input("Enter text to synthesize (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        tts.synthesize(user_input, args.output_path)

if __name__ == "__main__":
    main()