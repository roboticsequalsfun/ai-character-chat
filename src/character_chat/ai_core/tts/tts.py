import torch
import argparse
import soundfile as sf
from pathlib import Path
from character_chat.shared.paths import MODELS_DIR, OUTPUT_AUDIO_DIR
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

class TTS:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODELS_DIR / "tts" / "glow_tts_speaker_0"
        self.model_path = model_path
        # Load config
        config = GlowTTSConfig()
        config.load_json(self.model_path / "config.json")
        self.config = config

        # Audio processor
        self.ap = AudioProcessor.init_from_config(self.config)

        # Tokenizer
        self.tokenizer, self.config = TTSTokenizer.init_from_config(self.config)

        # Load model
        self.model = GlowTTS(self.config, self.ap, self.tokenizer, speaker_manager=None)
        self.model.load_checkpoint(
            self.config,
            checkpoint_path=self.model_path / "best_model.pth"
        )
        self.model.eval()
    
    def synthesize(self, text="Hello world", output_path=None):
        if output_path is None:
            output_path = OUTPUT_AUDIO_DIR / "output.wav"
        else:
            output_path = Path(output_path)

        with torch.no_grad():
            output = self.model.inference(text)

        wav = output["wav"]
        sf.write(output_path, wav, self.ap.sample_rate)

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text using a trained TTS model.")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the generated audio file",
        default=OUTPUT_AUDIO_DIR / "output.wav"
    )

    args = parser.parse_args()

    tts = TTS()

    while True:
        user_input = input("Enter text to synthesize (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        tts.synthesize(user_input, args.output_path)

if __name__ == "__main__":
    main()