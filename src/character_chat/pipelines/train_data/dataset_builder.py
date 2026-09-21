import shutil
from pathlib import Path
from character_chat.shared.paths import CHUNKS_DIR, DATASET_WAV_DIR, TRANSCRIPT_DIR, TEMP_DIR, SEPARATED_DIR, RUN_DIR, DATASET_DIR, CONFIG_PATH
from character_chat.shared.file_utils import ensure_dir, load_json, write_file, load_file, list_files_in_directory, load_config
from character_chat.shared.logger import log

class DatasetBuilder():
    def __init__(self, cache):
        log("===================================", "INFO")
        log("Starting dataset building...", "INFO")
        log("===================================", "INFO")

        speaker_folders = [f for f in CHUNKS_DIR.iterdir() if f.is_dir()]

        for speaker_folder in speaker_folders:
            self.speaker_name = speaker_folder.name
            speaker_id = int(self.speaker_name.split('_')[1])
            
            # Load per-speaker word timings (already remapped to local audio timeline)
            word_timings_path = TEMP_DIR / f"word_timings_speaker_{speaker_id}.json"
            metadata_path = TEMP_DIR / f"chunks_metadata_{speaker_id}.json"

            self.word_timings = load_json(word_timings_path)
            self.chunks_metadata = load_json(metadata_path)

            log(f"[INFO] Building dataset entries for {speaker_folder}...", "INFO")

            build_dataset_for_speaker(chunks_metadata, word_timings, speaker_name)
            create_metadata_for_speaker(speaker_name)
        if cache:
            cache_run(RUN_DIR)
        log("Dataset building complete. Final dataset is ready for training!", "INFO")

def build_dataset_for_speaker(self,):
        for i, chunk in enumerate(self.speaker_metadata):
            chunk_start = chunk['start']
            chunk_end = chunk['end']
            chunk_duration = chunk['duration']
            
            # Find words that fall within this chunk
            chunk_words = [
                word for word in self.word_timings
                if word['start'] >= chunk_start and word['end'] <= chunk_end
            ]

            words = " ".join([w['word'] for w in chunk_words])

            write_file(TRANSCRIPT_DIR / self.speaker_name / f"chunk_transcript_{i:04d}.txt", words)
            src = CHUNKS_DIR / self.speaker_name / f"chunk_{i:04d}.wav"
            dst = DATASET_WAV_DIR / self.speaker_name / f"chunk_{i:04d}.wav"
            shutil.copy2(src, dst)

            log(f"Created dataset entry for {i:04d} with {len(chunk_words)} words.", "INFO")

def cache_run(path, prefix="run"):
    # To an array of all folders in the directory
    arr = list(path.iterdir())

    folder_nums = []
    new_folder_num = 0

    for folder_name in arr:
        if not folder_name.is_dir():
            continue

        num = int(folder_name.name.split('_')[1])
        folder_nums.append(num)
    
    if folder_nums:
        new_folder_num = max(folder_nums) + 1
    else:
        new_folder_num = 0

    cache_dir = RUN_DIR / f"{prefix}_{new_folder_num}"
    ensure_dir(cache_dir)
    shutil.move(CHUNKS_DIR, cache_dir)
    shutil.move(SEPARATED_DIR, cache_dir)
    shutil.move(TRANSCRIPT_DIR, cache_dir)
    move_files(TEMP_DIR, cache_dir)
    ensure_dir(CHUNKS_DIR)
    ensure_dir(SEPARATED_DIR)
    ensure_dir(TRANSCRIPT_DIR)

def move_files(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        if item.is_file():
                shutil.move(str(item), str(dst_dir / item.name))

def create_metadata_for_speaker(speaker_id):
    log(f"[INFO] Creating metadata file for {speaker_id}", "INFO")
    files = list_files_in_directory(TRANSCRIPT_DIR / speaker_id)
    metadata = DATASET_DIR / f"metadata_{speaker_id}.csv"
    lines = []

    for i, file in enumerate(files):
        txt_contents = load_file(file)
        wav_filename = f"chunk_{i:04d}"
        relative_path = Path(f"{speaker_id}/{wav_filename}")
        normalized = txt_contents.lower().strip()
        lines.append(f"{relative_path}|{txt_contents}|{normalized}")
    write_file(metadata, "\n".join(lines))