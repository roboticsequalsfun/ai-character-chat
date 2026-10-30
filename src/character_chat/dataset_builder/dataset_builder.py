from character_chat.shared.paths import CHUNKS_DIR, SPEAKER_CHUNKS_DIR, TRANSCRIPT_DIR, TEMP_DIR, SEPARATED_DIR, RUN_DIR, DATASET_DIR, PROCESSED_DIR
import character_chat.shared.file_utils as file_utils
from character_chat.shared.logger import log
import re
from pathlib import Path
from num2words import num2words

def main(cache: bool = True, dataset_name: str = "character") -> None:
    """Build a dataset in LJSpeech form for each speaker, and then cache all the non-dataset files into temp/runss"""
    
    log("===================================", "INFO")
    log("Dataset Builder Starting...", "INFO")
    log("===================================", "INFO")

    speaker_folders = file_utils.list_folders_in_directory(CHUNKS_DIR)
    if speaker_folders is None:
        log(f"No speakers found in: {CHUNKS_DIR}. Ensure the input audio file is not empty.", "WARNING")

    for speaker in speaker_folders:
        speaker_id = file_utils.split_string(name=speaker, breaker='_')
        log(f"Building dataset entries for speaker_{speaker_id}...", "INFO")

        transcripts = file_utils.load_json(TRANSCRIPT_DIR / f"speaker_{speaker_id}" / f"word_data.json")
        build_dataset_for_speaker(transcripts, speaker_id=speaker_id, dataset_name=dataset_name)
    if cache:
        log(f"Caching run...", "INFO")
        cache_run(RUN_DIR)


def build_dataset_for_speaker(transcripts: dict, speaker_id: int, dataset_name: str) -> None:
    """Move the speaker's chunks into the dataset and create a metadat in the LJSpeech format"""

    file_utils.copy_files(CHUNKS_DIR / f"speaker_{speaker_id}", DATASET_DIR / dataset_name / "wavs" / f"speaker_{speaker_id}")
    metadata = DATASET_DIR / dataset_name / f"speaker_{speaker_id}.csv"
    lines = []

    for i, chunk in enumerate(transcripts):
        relative_path = f"speaker_{speaker_id}/chunk_{i:04d}" # Use a relative path, not a Path obejct, since its for the TTS training

        # This normalizes the text by going through each char in the sentence 
        # and keeping it if it is a letter, number, or space. Else it is discarded. Then lowercase everything
        normalized = convert_numbers_in_string(
            ("".join(char for char in chunk['sentence'] if str(char).isalnum() or char == " ")).lower())
        
        clean_sentence = convert_numbers_in_string(normalized) 

        lines.append(f"{relative_path}|{clean_sentence}|{clean_sentence}") # LJSpeech requires you to duplicate the sentence. Its stupid
    file_utils.write_file(metadata, "\n".join(lines))

             
def cache_run(path: str | Path) -> None:
    """Cache the dataset builder run into a new run and delete everything in temp"""

    folders = file_utils.list_folders_in_directory(path)
    folder_num = 0

    for folder in folders:
        num = file_utils.split_string(name=folder, breaker='_')

        if num > folder_num:
            folder_num = num
        folder_num += 1

    cache_dir = RUN_DIR / f"run_{folder_num}"
    try: 
        file_utils.move_directory(PROCESSED_DIR, cache_dir)
        file_utils.delete_files(TEMP_DIR)
        file_utils.delete_directory(SPEAKER_CHUNKS_DIR)
    except:
        pass

    # Make sure to replace all those folders we got rid of
    file_utils.ensure_base_dirs()


def convert_numbers_in_string(text: str) -> str:
    # Find all numbers blocks using re.sub function ('\d+' matches the blocks), then passes that to the lambda. Then convert it to a number
    return re.sub(r'\d+', lambda match: num2words(int(match.group(0))), text)


if __name__ == "__main__":
    config = file_utils.load_config("dataset_builder")
    main(cache=config.get("cache_run", True), dataset_name=config.get("name", "character"))