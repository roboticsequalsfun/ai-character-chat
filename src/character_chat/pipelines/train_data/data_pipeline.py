from character_chat.pipelines.train_data.speaker_separator import main as speaker_separator
from character_chat.pipelines.train_data.audio_chunker import main as audio_chunker
from character_chat.pipelines.train_data.dataset_builder import main as dataset_builder
from character_chat.shared.file_utils import load_config
from character_chat.shared.logger import log

log("===================================", "INFO")
log(" Building Dataset...", "INFO")
log("===================================\n", "INFO")

config = load_config("dataset_builder")
cache = config.get("cache_run", True)

input_file = config.get("diarization", {}).get("audio_file", "input_audio.wav")
device = config.get("diarization", {}).get("device", "cuda")
batch_size = config.get("diarization", {}).get("batch_size", 16)
compute_type = config.get("diarization", {}).get("compute_type", "float16")
max_chunk_duration = config.get("chunk_audio", {}).get("max_chunk_duration", 10.0)
min_chunk_duration = config.get("chunk_audio", {}).get("min_chunk_duration", 5.0)
ideal_chunk_duration = config.get("chunk_audio", {}).get("ideal_chunk_duration", 7.0)
min_silence_duration = config.get("chunk_audio", {}).get("min_silence_duration", 0.01)
weights = config.get("chunk_audio", {}).get("weights")

speaker_separator(input=input_file, device=device, batch_size=batch_size, compute_type=compute_type)
audio_chunker(max_chunk_duration=max_chunk_duration, 
    min_chunk_duration=min_chunk_duration, 
    ideal_chunk_duration=ideal_chunk_duration, 
    min_silence_duration=min_silence_duration,
    weights=weights
)
dataset_builder(cache=cache)

log("===================================", "INFO")
log("All Proccesing complete! Dataset ready for training", "INFO")
log("===================================", "INFO")