import torch
import soundfile as sf
import argparse
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from character_chat.shared.paths import OUTPUT_AUDIO_DIR, TTS_MODEL_DIR, DATASET_DIR
from character_chat.shared.file_utils import load_config


class TTS:
    def __init__(self, model_path=None, best_model="best_model.pth"):
        # Your fine-tuned model files
        config_path = str(model_path / "config.json")
        checkpoint_path = str(model_path / best_model)
        vocab_path = str(TTS_MODEL_DIR / "xtts_v2.0.1_checkpoints" / "vocab.json")
        dataset_config = load_config("tts_training")  # Load TTS training configuration

        dataset_path = DATASET_DIR / "wavs" / f"speaker_{dataset_config.get('dataset', {}).get('speaker')}"

        # A clean recording of the target speaker
        reference = [
            str(dataset_path / "chunk_0000.wav"),
            str(dataset_path / "chunk_0001.wav"),
            str(dataset_path / "chunk_0002.wav"),
            str(dataset_path / "chunk_0003.wav")
        ]

        # Load the fine-tuned XTTS model
        config = XttsConfig()
        config.load_json(config_path)

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            use_deepspeed=False,
        )

        # Use CPU so inference doesn't compete with training for GPU memory
        self.model.to("cuda")
        self.model.eval()

        # Prepare speaker conditioning
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=reference
        )

    def synthesize(self, text="Hello world", output_path=OUTPUT_AUDIO_DIR / "output.wav"):

        with torch.inference_mode():
            output = self.model.inference(
                text,
                "en",
                self.gpt_cond_latent,
                self.speaker_embedding,
                temperature=0.65,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=50,
                top_p=0.8,
            )

        # Save the generated audio
        sf.write(str(output_path), output["wav"], 24000)

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text using a trained TTS model.")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the generated audio file",
        default=str(OUTPUT_AUDIO_DIR / "output.wav")
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
        default=str(TTS_MODEL_DIR / "glow_tts_speaker_0")
    )

    args = parser.parse_args()

    args.model_name = Path(args.model_name)
    args.output_path = Path(args.output_path)

    tts = TTS(model_path=TTS_MODEL_DIR / args.model_name, best_model="best_model.pth")

    while True:
        user_input = input("Enter text to synthesize (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        tts.synthesize(user_input, args.output_path)

if __name__ == "__main__":
    main()