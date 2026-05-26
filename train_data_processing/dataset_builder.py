import shutil
from pathlib import Path
from shared.paths import CHUNKS_DIR, DATASET_WAV_DIR, TRANSCRIPT_DIR, TEMP_DIR, SEPARATED_DIR, RUN_DIR, DATASET_DIR, CONFIG_PATH
from shared.file_utils import ensure_dir, load_json, write_file, load_file, list_files_in_directory, load_config

def build_dataset_for_speaker(speaker_metadata, word_timings, speaker_name):
        for i, chunk in enumerate(speaker_metadata):
            chunk_start = chunk['start']
            chunk_end = chunk['end']
            chunk_duration = chunk['duration']
            
            # Find words that fall within this chunk
            chunk_words = [
                word for word in word_timings
                if word['start'] >= chunk_start and word['end'] <= chunk_end
            ]

            words = " ".join([w['word'] for w in chunk_words])

            write_file(TRANSCRIPT_DIR / speaker_name / f"chunk_transcript_{i:04d}.txt", words)
            src = CHUNKS_DIR / speaker_name / f"chunk_{i:04d}.wav"
            dst = DATASET_WAV_DIR / speaker_name / f"chunk_{i:04d}.wav"
            shutil.copy2(src, dst)

            print(f"Created dataset entry for {i:04d} with {len(chunk_words)} words.")

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
    print(f"[INFO] Creating metadata file for {speaker_id}")
    files = list_files_in_directory(TRANSCRIPT_DIR / speaker_id)
    metadata = DATASET_DIR / f"metadata_speaker{speaker_id}.txt"
    wav_dir = DATASET_WAV_DIR / speaker_id
    lines = []

    for i, file in enumerate(files):
        txt_contents = load_file(file)
        wav_path = wav_dir / f"chunk_{i:04d}.wav"
        relative_path = wav_path.relative_to(DATASET_DIR)
        lines.append(f"{relative_path.as_posix()} | {txt_contents}")
    write_file(metadata, "\n".join(lines))



def main():
    print("===================================")
    print("Starting dataset building...")
    print("===================================")
    
    speaker_folders = [f for f in CHUNKS_DIR.iterdir() if f.is_dir()]

    config = load_config()
    cache = config.get("dataset", {}).get("cache_run")

    for speaker_folder in speaker_folders:
        speaker_name = speaker_folder.name
        speaker_id = int(speaker_name.split('_')[1])
        
        # Load per-speaker word timings (already remapped to local audio timeline)
        word_timings_path = TEMP_DIR / f"word_timings_speaker_{speaker_id}.json"
        metadata_path = TEMP_DIR / f"chunks_metadata_{speaker_id}.json"

        word_timings = load_json(word_timings_path)
        chunks_metadata = load_json(metadata_path)

        print(f"[INFO] Building dataset entries for {speaker_folder}...")

        build_dataset_for_speaker(chunks_metadata, word_timings, speaker_name)
        create_metadata_for_speaker(speaker_name)
    if cache:
        cache_run(RUN_DIR)
    print("Dataset building complete. Final dataset is ready for training!")

if __name__ == "__main__":
    main()