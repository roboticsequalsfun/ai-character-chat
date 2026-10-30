import torch
import argparse
import torchaudio
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from character_chat.shared.paths import OUTPUT_AUDIO_DIR, TTS_MODEL_DIR, DATASET_DIR
from character_chat.shared.file_utils import load_config, path_exists
from character_chat.ai_core.audio.player import play
from character_chat.shared.logger import log


class TTS:
    def __init__(self, model_path: str | Path = None, temperature: int | float = 0.65,
            length_penalty: int | float= 1.0, repetition_penalty: int | float = 10.0,
            top_k: int | float = 50, top_p: int | float = 0.8, sample_rate: int = 44100) -> None:
        """Convert inputed text to speech using pretrained model from model_path"""

        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.top_p = top_p
        path_exists(model_path)

        # -------------- Setup the model --------------

        # Get the raw path of the model since the actual model path contains /best_model.pth
        parent_path = model_path.parent

        # Get the two files for loading a pretrained model
        config_path = str(parent_path / "config.json")
        checkpoint_path = str(model_path)

        # Get the path to the reference audio
        dataset_config = load_config("tts_training")
        dataset_path = DATASET_DIR / "wavs" / f"speaker_{dataset_config.get('dataset', {}).get('speaker')}"
        
        # Catch invalid paths
        path_exists(config_path)
        path_exists(checkpoint_path)
        path_exists(dataset_path)

        # More is better! Since audio clips may vary a bit, having multiple references averages it out
        reference = [
            str(dataset_path / "chunk_0000.wav"),
            str(dataset_path / "chunk_0001.wav"),
            str(dataset_path / "chunk_0002.wav"),
            str(dataset_path / "chunk_0003.wav")
        ]

        log("Loading the TTS model (This may take a while)...", "INFO")

        # -------------- Load the model --------------

        # Load the original config
        config = XttsConfig()
        config.load_json(config_path)

        # Now load the model
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            use_deepspeed=False, # We don't use deepspeed since it breaks the venv
        )

        # Catch null model. Probably won't happen but better to be on the safe side
        if self.model is None:
            log(f"Failed to load pretrained model: {model_path}", "ERROR")

        self.model.cuda()

        # Prepare speaker conditioning
        log("Computing speaker latents...", "INFO")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=reference
        )

    def synthesize(self, text: str = "Hello world", output_path: str | Path = OUTPUT_AUDIO_DIR / "output.wav") -> None:
        """Convert the input to speech using the trained model"""

        log("Converting text to speech...", "INFO")
        with torch.inference_mode():
            # And now it actually takes the text and uses the model to magically make speech! 
            # Okay its actually a bunch of multi dimensional floating point arrays, loaded onto the GPU, but that's not as cool.
            output = self.model.inference(
                text,
                "en", # Currently at v0.1.0 I'm only supporting english. That will change though.
                self.gpt_cond_latent,
                self.speaker_embedding,
                temperature=self.temperature,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                top_p=self.top_p,
            )

        # Use torchaudio causes thats what the doc says. I mean, because it is more efficient.
        wav = torch.as_tensor(output["wav"], dtype=torch.float32).detach().cpu()
        torchaudio.save(output_path, wav.reshape(1, -1), 24000)
        log(f"Saved audio to: {output_path}")

def main() -> None:
    """Since we are entering by calling the script we are in testing mode"""

    # -------------- Argparse --------------

    # Since we are in testing mode and the TTS relies on more than just some config stuff, we use argparse.
    parser = argparse.ArgumentParser(description="Synthesize speech from text using a trained TTS model.")

    # The output path
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the generated audio file",
        default=str(OUTPUT_AUDIO_DIR / "output.wav")
    )

    # And the model path
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the best model e.g. voice/run/best_model.pth",
        default=str(TTS_MODEL_DIR / "xtts_speaker_0/best_model.pth")
    )

    # Inintialize the arguments
    args = parser.parse_args()
    args.model_path = Path(args.model_path) # Model path is critical, probably will fail it without it
    args.output_path = Path(args.output_path) # Output path is non critical

    path_exists(args.model_path)
    path_exists(args.output_path)

    # -------------- Load Config --------------

    config = load_config()

    temperature = config.get("tts", {}).get("temperature", 0.65)
    length_penalty = config.get("tts", {}).get("length_penalty", 1.0)
    repetition_penalty = config.get("tts", {}).get("repetition_penalty", 10.0)
    top_k = config.get("tts", {}).get("top_k", 50)
    top_p = config.get("tts", {}).get("top_p", 0.8)

    # Load the model
    tts = TTS(
        model_path=TTS_MODEL_DIR / args.model_path, 
        temperature=temperature, 
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p
    )

    # -------------- Start the TTS --------------
    while True:
        # Get the user input
        user_input = input("Enter text to synthesize (or 'exit' to quit): ")

        # If the input is exit we stop the script
        if user_input.lower() == "exit":
            break
         
        # Else we synthesize the audio
        tts.synthesize(user_input, args.output_path)

        # Then play it!
        play(args.output_path)

if __name__ == "__main__":
    main()