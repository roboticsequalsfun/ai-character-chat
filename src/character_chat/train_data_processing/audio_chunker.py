from pathlib import Path
import time
from character_chat.shared.paths import CHUNKS_DIR, SEPARATED_DIR, TEMP_DIR
from character_chat.shared.audio_io import load_audio_librosa, write_audio
from character_chat.shared.file_utils import ensure_dir, write_json, load_json, list_files_in_directory, load_config
import numpy as np
import soundfile as sf

def create_intelligent_chunks(word_timings, max_chunk_duration=10.0, min_chunk_duration=2.0):
    """
    Create chunks at word boundaries, respecting max duration.
    Returns list of chunks with start/end times.
    """
    if not word_timings:
        return []
    
    chunks = []
    current_chunk_start = 0
    current_chunk_words = []
    
    for word_data in word_timings:
        word_end = word_data['end']
        chunk_duration = word_end - current_chunk_start
        
        # If adding this word would exceed max duration, save current chunk
        if chunk_duration > max_chunk_duration and current_chunk_words:
            chunk_end = current_chunk_words[-1]['end']
            chunks.append({
                'start': current_chunk_start,
                'end': chunk_end,
                'duration': chunk_end - current_chunk_start
            })
            current_chunk_start = word_data['start']
            current_chunk_words = [word_data]
        else:
            current_chunk_words.append(word_data)
    
    # Add final chunk
    if current_chunk_words:
        chunk_end = current_chunk_words[-1]['end']
        duration = chunk_end - current_chunk_start

        if duration >= min_chunk_duration:
            chunks.append({
                'start': current_chunk_start,
                'end': chunk_end,
                'duration': chunk_end - current_chunk_start
            })
    
    return chunks


def save_chunks_for_speaker(audio_file, word_timings, speaker_id, max_chunk_duration=10.0, min_chunk_duration=2.0, start_padding=0.5, end_padding=0.5):
    """
    Load audio, create intelligent chunks, and save them.
    Word timings should already be remapped to this speaker's local audio timeline.
    """
    # Load audio
    y, sr = load_audio_librosa(audio_file)
    
    # Create output directory for this speaker
    chunks_dir = CHUNKS_DIR / f"speaker_{speaker_id}"
    ensure_dir(chunks_dir)
    
    # Word timings are already per-speaker with times remapped to local timeline
    if not word_timings:
        print(f"  No words found for speaker {speaker_id}")
        return 0

    # Create intelligent chunks
    chunks = create_intelligent_chunks(
        word_timings,
        max_chunk_duration=max_chunk_duration,
        min_chunk_duration=min_chunk_duration
    )
    valid_chunks = []

    # Save each chunk
    for i, chunk in enumerate(chunks):
        # Convert to samples
        start_sample = max(0, int((chunk['start'] - start_padding) * sr))
        end_sample = min(len(y), int((chunk['end'] + end_padding) * sr))

        if end_sample > start_sample:
            # Extract chunk audio
            chunk_audio = y[start_sample:end_sample]

            # Save chunk
            time.sleep(0.01)
            chunk_file = chunks_dir / f"chunk_{i:04d}.wav"
            chunk_file.parent.mkdir(parents=True, exist_ok=True)
            write_audio(chunk_file, chunk_audio, sr)
            
            duration_ms = int(chunk['duration'] * 1000)
            print(f"  Saved chunk {i:04d}: {duration_ms}ms")
            valid_chunks.append(chunk)
    
    write_json(TEMP_DIR / f"chunks_metadata_{speaker_id}.json", valid_chunks)
    return len(valid_chunks)


def main():
    print("===================================")
    print(" Audio Chunker Starting...")
    print("===================================")

    config = load_config()
    max_chunk_duration = config.get("audio", {}).get("chunk_duration", 10.0)
    min_chunk_duration = config.get("audio", {}).get("min_chunk_duration", 2.0)
    enable_padding = config.get("audio", {}).get("padding", {}).get("enabled", True)
    start_padding = config.get("audio", {}).get("padding", {}).get("start", 0.5)
    end_padding = config.get("audio", {}).get("padding", {}).get("end", 0.5)
    if not enable_padding:
        start_padding = 0.0
        end_padding = 0.0

    print(f"[INFO] Starting intelligent chunking process...")
    print(f"[INFO] Max chunk duration: {max_chunk_duration}s")

    # Get unique speakers from separated directory
    speaker_files = list_files_in_directory(SEPARATED_DIR, extension=".wav")

    total_chunks = 0
    for speaker_file in speaker_files:
        speaker_name = Path(speaker_file).stem
        speaker_id = int(speaker_name.split('_')[1]) if '_' in speaker_name else 0
        word_timings = load_json(TEMP_DIR / f"word_timings_speaker_{speaker_id}.json")
        
        print(f"[INFO] Processing {speaker_file}...")

        chunks_count = save_chunks_for_speaker(
            SEPARATED_DIR / speaker_file,
            word_timings,
            speaker_id,
            max_chunk_duration,
            min_chunk_duration,
            start_padding,
            end_padding
        )
        total_chunks += chunks_count
        print(f"  Created {chunks_count} chunks for {speaker_name}")

    print("===================================")
    print(f"[DONE] All chunks saved ({total_chunks} total chunks)")
    print("===================================")


if __name__ == "__main__":
    main()