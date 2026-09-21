from pathlib import Path
import json
from character_chat.shared.logger import log
from character_chat.shared.paths import CONFIG_DIR, APP_CONFIG_PATH, DATASET_BUILDER_CONFIG_PATH, TTS_TRAINING_CONFIG_PATH

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def write_json(path, data):
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))

def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"JSON file {path} does not exist.")
    return json.loads(path.read_text())

def load_config(config_type):
    config = None

    match config_type:
        case "app":
            config = APP_CONFIG_PATH
        case "dataset_builder":
            config = DATASET_BUILDER_CONFIG_PATH
        case "tts_training":
            config = TTS_TRAINING_CONFIG_PATH
        case _:
            log("")

    with open(CONFIG_DIR / config, "r") as f:
        return json.load(f)

def write_file(path, content=None):
    ensure_dir(path.parent)
    if content:
        path.write_text(content)

def load_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def list_files_in_directory(directory, extension=None):
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a valid directory.")  
    if extension:
        return [f for f in directory.iterdir() if f.is_file() and f.suffix == extension]
    else:
        return [f for f in directory.iterdir() if f.is_file()]
    
