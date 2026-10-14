from pathlib import Path
import json
import os
import shutil
from natsort import natsorted
from character_chat.shared.logger import log
from character_chat.shared.paths import CONFIG_DIR, APP_CONFIG_PATH, DATASET_BUILDER_CONFIG_PATH, TTS_TRAINING_CONFIG_PATH

def path_exists(path):
    path = Path(path)
    if not path.exists():
        log(f"{path} does not exist", "ERROR")

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))

def load_json(path):
    path = Path(path)
    path_exists(path)
    return json.loads(path.read_text(encoding="utf-8"))

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
            log("Unkown config type passed", "WARNING")

    with open(CONFIG_DIR / config, "r") as f:
        return json.load(f)

def write_file(path, content=None):
    path = Path(path)
    ensure_dir(path.parent)
    if content:
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)

def load_file(path):
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def move_files(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        log(f"{src_dir} does not exist", "ERROR")
    ensure_dir(dst_dir)

    for item in src_dir.iterdir():
        if item.is_file():
            shutil.move(str(item), str(dst_dir / item.name))

def list_files_in_directory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        log(f"{directory} is not a valid directory.", "ERROR")
    # Get all files
    files = natsorted([f.name for f in directory.iterdir() if f.is_file()])
    if files is None:
        log(f"No files found in {directory}", "WARNING")
    return files

def list_folders_in_directory(directory):
    directory = Path(directory)
    if not directory.is_dir():
        log(f"{directory} is not a valid directory.", "ERROR")
    folders = [ f.path for f in os.scandir(directory) if f.is_dir() ]
    if folders is None:
        log(f"No folders found in {directory}", "WARNING")
    return folders

def name_to_int(name, breaker='_'):
    return int(name.split(breaker)[1])
    
