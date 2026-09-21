from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INPUT_AUDIO_DIR = DATA_DIR / "raw" / "audio"
OUTPUT_AUDIO_DIR = DATA_DIR / "output" / "audio"
OUTPUT_TEXT_DIR = DATA_DIR / "output" / "text"

DATASET_DIR = DATA_DIR / "dataset"
DATASET_WAV_DIR = DATASET_DIR / "wavs"

TRANSCRIPT_DIR = PROCESSED_DIR / "transcripts"
SEPARATED_DIR = PROCESSED_DIR / "separated"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
CHUNK_DATA_DIR = PROCESSED_DIR / "chunk_data"

CONFIG_DIR = PROJECT_DIR / "configs"
APP_CONFIG_PATH = CONFIG_DIR / "app.json"
DATASET_BUILDER_CONFIG_PATH = CONFIG_DIR / "dataset_builder.json"
TTS_TRAINING_CONFIG_PATH = CONFIG_DIR / "tts_training.json"
PROMPT_DIR = CONFIG_DIR / "prompts"
TEMP_DIR = DATA_DIR / "temp"
RUN_DIR = TEMP_DIR / "runs"


MODELS_DIR = PROJECT_DIR / "models"
LLM_MODELS_DIR = MODELS_DIR / "llm"