import shutil
from character_chat.shared.paths import CHUNKS_DIR, SPEAKER_CHUNKS_DIR, TRANSCRIPT_DIR, TEMP_DIR, SEPARATED_DIR, RUN_DIR, DATASET_DIR, DATASET_WAV_DIR
from character_chat.shared.file_utils import ensure_dir, load_json, write_file, list_folders_in_directory, move_files, name_to_int, path_exists, load_config
from character_chat.shared.logger import log
import re
from num2words import num2words

def main(cache):
    log("===================================", "INFO")
    log("Starting dataset building...", "INFO")
    log("===================================", "INFO")

    speaker_folders = list_folders_in_directory(CHUNKS_DIR)

    for speaker in speaker_folders:
        speaker_id = name_to_int(name=speaker, breaker='_')
        log(f"Building dataset entries for speaker_{speaker_id}...", "INFO")
        
        transcript_path = TRANSCRIPT_DIR / f"speaker_{speaker_id}" / f"word_data.json"
        path_exists(transcript_path)
        transcripts = load_json(transcript_path)

        build_dataset_for_speaker(transcripts, speaker_id)
    if cache:
        log(f"Caching run...", "INFO")
        cache_run(RUN_DIR)

def build_dataset_for_speaker(transcripts, speaker_id):
    shutil.copytree(CHUNKS_DIR / f"speaker_{speaker_id}", DATASET_WAV_DIR / f"speaker_{speaker_id}", dirs_exist_ok=True)
    metadata = DATASET_DIR / f"speaker_{speaker_id}.csv"
    lines = []
    for i, chunk in enumerate(transcripts):
        wav_filename = f"chunk_{i:04d}"
        relative_path = f"speaker_{speaker_id}\{wav_filename}"
        normalized = convert_numbers_in_string(("".join(char for char in chunk['sentence'] if char.isalnum() or char.isspace())).lower())
        clean_sentence = convert_numbers_in_string(chunk['sentence'].encode('utf-8', 'ignore').decode('utf-8'))
        lines.append(f"{relative_path}|{clean_sentence}|{normalized}")
    write_file(metadata, "\n".join(lines))
             
def cache_run(path):
    # To an array of all folders in the directory
    folders = list_folders_in_directory(path)
    folder_num = 0

    for folder in folders:
        num = name_to_int(name=folder, breaker='_')
        if num > folder_num:
            folder_num = num
        folder_num += 1

    cache_dir = RUN_DIR / f"run_{folder_num}"
    ensure_dir(cache_dir)
    try: 
        shutil.move(CHUNKS_DIR, cache_dir)
        shutil.move(SEPARATED_DIR, cache_dir)
        shutil.move(TRANSCRIPT_DIR, cache_dir)
        shutil.move(SPEAKER_CHUNKS_DIR, cache_dir)
    except:
        pass
    move_files(TEMP_DIR, cache_dir)
    ensure_dir(CHUNKS_DIR)
    ensure_dir(SEPARATED_DIR)
    ensure_dir(TRANSCRIPT_DIR)
    ensure_dir(SPEAKER_CHUNKS_DIR)

def convert_numbers_in_string(text: str) -> str:
    # Match any sequence of digits (\d+) and replace them with their word form
    return re.sub(r'\d+', lambda match: num2words(int(match.group(0))), text)

if __name__ == "__main__":
    config = load_config("dataset_builder")
    main(True)