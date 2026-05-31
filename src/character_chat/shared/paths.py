import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_DIR / "data"
INPUT_AUDIO_DIR = DATA_DIR / "raw" / "audio"
SEPARATED_DIR = DATA_DIR / "processed" / "separated"
CHUNKS_DIR = DATA_DIR / "processed" / "chunks"
OUTPUT_AUDIO_DIR = DATA_DIR / "output" / "audio"
OUTPUT_TEXT_DIR = DATA_DIR / "output" / "text"
PROCESSED_DIR = DATA_DIR / "processed"

DATASET_DIR = DATA_DIR / "dataset"
DATASET_WAV_DIR = DATASET_DIR / "wavs"
TRANSCRIPT_DIR = PROCESSED_DIR / "transcripts"

CONFIG_DIR = PROJECT_DIR / "configs"
CONFIG_PATH = CONFIG_DIR / "config.json"
PROMPT_DIR = CONFIG_DIR / "prompts"
TEMP_DIR = DATA_DIR / "temp"
RUN_DIR = TEMP_DIR / "runs"


MODELS_DIR = PROJECT_DIR / "models"
LLM_MODELS_DIR = MODELS_DIR / "llm"