from character_chat.pipelines.train_data.speaker_separator import SpeakerSeparator
from character_chat.pipelines.train_data.audio_chunker import AudioChunker
from character_chat.pipelines.train_data.dataset_builder import DatasetBuilder
from character_chat.shared.file_utils import load_config
from character_chat.shared.logger import log
from character_chat.shared.paths import INPUT_AUDIO_DIR
import argparse

config = load_config("dataset_builder")
cache = config.get("cache_run", True)

input_file = INPUT_AUDIO_DIR / config.get("diarization", {}).get("audio_file", "input_audio.wav")
punctuate = config.get("diarization", {}).get("punctuate", True)
diarize = config.get("diarization", {}).get("enabled", True)
max_silence_ms = config.get("diarization", {}).get("max_silence_ms", 1000)
max_chunk_duration = config.get("chunk_audio", {}).get("chunk_duration", 5.0)
min_chunk_duration = config.get("chunk_audio", {}).get("min_chunk_duration", 2.0)
enable_padding = config.get("chunk_audio", {}).get("padding", {}).get("enabled", True)
start_padding = config.get("chunk_audio", {}).get("padding", {}).get("start", 0.5)
end_padding = config.get("chunk_audio", {}).get("padding", {}).get("end", 0.5)
if not enable_padding:
    start_padding = 0.0
    end_padding = 0.0

SpeakerSeparator(input_file=input_file, punctuate=punctuate, diarize=diarize, max_silence_ms=max_silence_ms)
AudioChunker(max_chunk_duration=max_chunk_duration, min_chunk_duration=min_chunk_duration, start_padding=start_padding, end_padding=end_padding)
DatasetBuilder(cache=cache)

log("All processing complete. Dataset is ready for training!", "INFO")