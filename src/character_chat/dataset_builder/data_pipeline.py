from character_chat.dataset_builder.speaker_separator import SpeakerSeparator
from character_chat.dataset_builder.audio_chunker import main as audio_chunker
from character_chat.dataset_builder.dataset_builder import main as dataset_builder
from character_chat.shared.file_utils import load_config, ensure_type
from character_chat.shared.paths import INPUT_AUDIO_DIR
from character_chat.shared.logger import log
import warnings
from pathlib import Path

def main() -> None:
    """Build the dataset for TTS in the LJSpeech. Including converting audio to wav, 
    cleaning, diarizing, transcribing, chunking intelligently, cleaning up its traced, 
    and creating the dataset in the LJSpeech format"""

    log("===================================", "INFO")
    log(" Building Dataset...", "INFO")
    log("===================================\n", "INFO")

    config = load_config("dataset_builder")
    dataset_name = config.get("name", "character")
    cache = config.get("cache_run", True)

    input_file = INPUT_AUDIO_DIR / config.get("diarization", {}).get("audio_file", "input_audio.wav")  # Must include extension
    device = config.get("diarization", {}).get("device", "cuda")
    batch_size = config.get("diarization", {}).get("batch_size", 16)  # Bigger can overflow or run out of VRAM memory
    compute_type = config.get("diarization", {}).get("compute_type", "float16")
    clean = config.get("clean_audio", True)  # I would leave it true. So the TTS doesn't sound like a old VHS tape
    target_level = config.get("diarization", {}).get("target_level", -20.0)

    max_chunk_duration = config.get("chunk_audio", {}).get("max_chunk_duration", 10.0)
    min_chunk_duration = config.get("chunk_audio", {}).get("min_chunk_duration", 5.0)
    ideal_chunk_duration = config.get("chunk_audio", {}).get("ideal_chunk_duration", 7.0)
    min_silence_duration = config.get("chunk_audio", {}).get("min_silence_duration", 0.01)
    weights = config.get("chunk_audio", {}).get("weights")

    ensure_type(cache, bool, "cache_run", "dataset_builder")

    ensure_type(input_file, str | Path, "audio_file", "dataset_builder")
    ensure_type(device, str, "device", "dataset_builder")
    ensure_type(batch_size, (int, float), "batch_size", "dataset_builder")
    ensure_type(compute_type, str, "compute_type", "dataset_builder")
    ensure_type(clean, bool, "clean_audio", "dataset_builder")
    ensure_type(target_level, (int, float), "target_level", "dataset_builder")

    ensure_type(max_chunk_duration, (int, float), "max_chunk_duration", "dataset_builder")
    ensure_type(min_chunk_duration, (int, float), "min_chunk_duration", "dataset_builder")
    ensure_type(ideal_chunk_duration, (int, float), "ideal_chunk_duration", "dataset_builder")
    ensure_type(min_silence_duration, (int, float), "min_silence_duration", "dataset_builder")

    SpeakerSeparator(input_file=input_file, 
        device=device, 
        batch_size=batch_size, 
        compute_type=compute_type,
        clean=clean,
        target_level=target_level
    )
    audio_chunker(max_chunk_duration=max_chunk_duration, 
        min_chunk_duration=min_chunk_duration, 
        ideal_chunk_duration=ideal_chunk_duration, 
        min_silence_duration=min_silence_duration,
        weights=weights
    )
    dataset_builder(cache=cache, dataset_name=dataset_name)

    log("===================================", "INFO")
    log("All Proccesing complete! Dataset ready for training", "INFO")
    log("===================================", "INFO")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="torchcodec")
    warnings.filterwarnings("ignore", module="libtorchcodec")
    main()