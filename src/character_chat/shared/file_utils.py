from pathlib import Path
import json
import os
import shutil
from natsort import natsorted
from character_chat.shared.logger import log
from character_chat.shared.paths import CONFIG_DIR, APP_CONFIG_PATH, DATASET_BUILDER_CONFIG_PATH, TTS_TRAINING_CONFIG_PATH
import character_chat.shared.paths as all_paths

"""A bunch of high level file operations"""

def path_exists(path: str | Path) -> None:
    """Check if path exists"""

    path = Path(path)
    if not path.exists():
        log(f"{path} does not exist", "ERROR")

def ensure_base_dirs() -> None:
    """Make sure that all the directories in shared/paths.py exist"""

    for name, value in vars(all_paths).items():
        if isinstance(value, Path):
            if not value.is_file():
                ensure_dir(value)

def ensure_dir(path: str | Path) -> None:
    """Create dir if it doesn't exist"""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def write_json(path: str | Path, data: str) -> None:
    """Write json data to file"""

    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))

def load_json(path: str | Path) -> None:
    """Load json from file"""

    path = Path(path)
    path_exists(path)
    return json.loads(path.read_text(encoding="utf-8"))

def load_config(config_type: str) -> None:
    """Load config based on the three types"""

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

    return load_json(CONFIG_DIR / config)

def write_file(path: str | Path, content: str = None) -> None:
    """Write data to file"""

    path = Path(path)
    ensure_dir(path.parent)
    if content:
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)

def load_file(path: str | Path) -> str:
    """Load data from file"""

    path = Path(path)
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def copy_files(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Copy only files"""

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    path_exists(src_dir)
    ensure_dir(dst_dir)

    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(str(item), str(dst_dir / item.name))

def move_directory(src_dir: str | Path, dst_dir: str | Path) -> None:
    """Move the folder and all its contents"""

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    path_exists(src_dir)
    ensure_dir(dst_dir)

    shutil.move(src_dir, dst_dir)

def delete_files(dir: str | Path) -> None:
    """Delete only files in directory"""

    dir = Path(dir)
    path_exists(dir)

    for item in dir.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True) 

def delete_directory(dir: str | Path) -> None:
    """Delete everything inside of directory"""

    dir = Path(dir)
    path_exists(dir)

    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def list_files_in_directory(directory: str | Path) -> list: 
    """List only files in directory (excludes .gitkeep)"""

    directory = Path(directory)
    if not directory.is_dir():
        log(f"{directory} is not a valid directory.", "ERROR")
    # Get all files
    files = natsorted([f.name for f in directory.iterdir() if f.is_file() and f.name() != ".gitkeep"])
    if files is None:
        log(f"No files found in {directory}", "WARNING")
    return files

def list_folders_in_directory(directory: str | Path) -> list:
    """List only folders in directory"""

    directory = Path(directory)
    if not directory.is_dir():
        log(f"{directory} is not a valid directory.", "ERROR")
    folders = [ f.path for f in os.scandir(directory) if f.is_dir() ]
    if folders is None:
        log(f"No folders found in {directory}", "WARNING")
    return folders

def split_string(name: str | Path, breaker: str = '_', module: int = 1, is_int: bool = True) -> int:
    """Get the number from a string e.g. speaker_0 and grab the number"""

    if name is Path:
        name = str(Path(name).name)

    parts = str(str(name).lower()).split(breaker)

    if module < 0 or module >= len(parts):
        raise ValueError(
            f"Cannot extract index {module} from {name!r} "
            f"using separator {breaker!r}. "
            f"Available parts: {parts}"
        )

    if is_int:
        parts = int(parts[module]) 
        if not isinstance(parts, int):
            log(f"No number found in string: {name}", "ERROR") 
        return parts
        
    return parts[module]
    
def ensure_type(value, type, name: str, config: str) -> None:
    """Throw an error if passed in variable is not a certain type"""
    
    if not isinstance(value, type):
        log(
            f"{name}, from {config} config, is not a {type}. Please change the value to the correct type.", 
            "ERROR", no_origin=True)