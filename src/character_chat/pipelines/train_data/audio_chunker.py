from pathlib import Path
from pydub import AudioSegment
from character_chat.shared.paths import CHUNKS_DIR, SEPARATED_DIR, TEMP_DIR, CHUNK_DATA_DIR, INPUT_AUDIO_DIR
from character_chat.shared.audio_io import load_audio_librosa, write_audio
from character_chat.shared.file_utils import ensure_dir, write_json, load_json, list_files_in_directory
from character_chat.shared.logger import log

def main(max_chunk_duration=10.0, min_chunk_duration=5.0, ideal_chunk_duration=7.0, min_silence_duration=0.001):
    audio_file = None
    speaker_id = 0
    word_timings = TEMP_DIR / f"word_timings_speaker_{speaker_id}.json"

    log("===================================" , "INFO")
    log(" Audio Chunker Starting...", "INFO")
    log("===================================\n", "INFO")

    # Get unique speakers from separated directory
    speaker_files = list_files_in_directory(SEPARATED_DIR, extension=".wav")


    for speaker_file in speaker_files:
        speaker_name = Path(speaker_file).stem
        speaker_id = int(speaker_name.split('_')[1]) if '_' in speaker_name else 0
        audio_file = SEPARATED_DIR / f"speaker_{speaker_id}.wav"
        word_timings = load_json(TEMP_DIR / f"word_timings_speaker_{speaker_id}.json")

        log(f"============ Speaker {speaker_id} ============", "INFO")

        log(f"Finding cuts (This might take a while)...", "INFO")
        raw_chunk_data = define_chunks(word_timings, 
            max_chunk_duration=max_chunk_duration, 
            min_chunk_duration=min_chunk_duration, 
            ideal_chunk_duration=ideal_chunk_duration, 
            min_silence_duration=min_silence_duration
        )

        cuts, chunks = format_exports(raw_chunk_data)

        write_json(TEMP_DIR / f"raw_chunk_data_speaker_{speaker_id}.json", raw_chunk_data)
        write_json(CHUNK_DATA_DIR / f"speaker_{speaker_id}" / f"word_data.json", chunks)
        write_json(CHUNK_DATA_DIR / f"speaker_{speaker_id}" / f"cuts.json", cuts)

        log(f"Chunking Audio...", "INFO")
        amount = cut_audio(audio_file, cuts, CHUNKS_DIR / f"speaker_{speaker_id}")
        log(f"[DONE] {amount} chunks saved for speaker {speaker_id}\n ", "INFO")

    log("===================================", "INFO")
    log(f"[DONE] All chunks saved ", "INFO")
    log("===================================", "INFO")

def define_chunks(word_timings, max_chunk_duration=None, min_chunk_duration=None, ideal_chunk_duration=None, min_silence_duration=None):
    """
    Create chunks at word boundaries, respecting max duration.
    Returns list of chunks with start/end times.
    """

    # Null catch for empty word_timings
    if not word_timings:
        return []

    cuts = [] 
    max_chunk = []
    last_marker = 0
    i = 0
    chunk_duration = 0
    last_word_end = 0
    markers = [min_chunk_duration + last_marker, max_chunk_duration + last_marker, ideal_chunk_duration + last_marker]

    # Loop through word timings and create chunks based on duration and silence
    while i < len(word_timings):
        print("next iteration")
        # Get the current word and markers for chunking
        word = word_timings[i]

        if not max_chunk:
            chunk_start_time = word['start']

        # Update chunk duration and check if it exceeds max_chunk_duration
        chunk_duration = word['end'] - chunk_start_time

        if chunk_duration >= max_chunk_duration:
            possible_cuts = []

            last_word_end = last_marker

            # If so, evaluate possible cuts based on silence duration and score them
            for item in max_chunk:
                # Calculate the duration of silence between the last word and the current word, and check if it meets the minimum silence duration requirement
                silence_duration = item['start'] - last_word_end

                if silence_duration >= min_silence_duration:
                    # If so then calculate the score for this cut and add it to the list of possible cuts
                    score, mid, min, max, ideal = score(item['start'], item['end'], markers[0], markers[1], markers[2], silence_duration)
                    if mid > markers[0] and mid < markers[1]:
                        possible_cuts.append(cut_template(mid, item['index'], "normal", score, min, max, ideal))
                        print("passed")
                    print("Mid: ", mid)
                    print("Markers: ", markers)
                # Update the last word end time for the next iteration
                last_word_end = item['end']

            # Null catch for no possible cuts found, force a cut at the last word
            if not possible_cuts:
                mid = max_chunk[-1]['end']
                fallback_index = max_chunk[-1]['index']
                possible_cuts.append(
                    cut_template(mid, fallback_index, "default", 1, 0, 0, 0)
                )

            # Select the cut with the highest score from the possible cuts
            highest_value = -1
            chosen_cut = possible_cuts[0]
            for cut in possible_cuts:
                if cut['score']['total'] > highest_value:
                    highest_value = cut['score']['total']
                    chosen_cut = cut

            words = []
            for word in max_chunk:
                if word['index'] >= chosen_cut['index']:
                    break
                words.append(word)

            cuts.append({'cut': chosen_cut, 'words': words})

            # Reset everything and go back to the iteration of the cut
            chunk_duration = 0
            last_marker = chosen_cut['location']
            max_chunk = []
            markers = [
                min_chunk_duration + last_marker, 
                max_chunk_duration + last_marker, 
                ideal_chunk_duration + last_marker
            ]

            i = chosen_cut['index'] + 1
            continue

        # If not then append the word to the max chunk
        max_chunk.append({
            'word': word['word'], 
            'duration': word['end'] - word['start'], 
            'start': word['start'], 
            'end': word['end'],
            'index': i
        })  
        i += 1

    # Handle any remaining words after the loop
    if max_chunk:
        cuts.append({
            'cut': cut_template(max_chunk[-1]['end'], len(word_timings) - 1, "cutoff", 1, 0, 0, 0),
            'words': max_chunk
        })

    # Make a cut at the end to signify the end of the audio
    cuts.append({
        'cut': cut_template(max_chunk[-1]['end'], len(word_timings) - 1, "end", 1, 0, 0, 0), 
        'words': []
    })
    return cuts

def cut_audio(audio_file, cuts, path, padding=0): 
    path.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_file(audio_file)

    last_cut_ms = 0
    for cut, i in range(cuts):
        cut_ms = int(cut['location'] * 1000)

        chunk = audio[last_cut_ms:cut_ms]
        chunk.export(path / f"chunk_{i:04}.wav", format="wav")

        last_cut_ms = cut_ms

    return i

def cut_template(location, index, tag, score, min_d, max_d, ideal_d):
    return {
        'location': location, 
        'index': index, 
        'score': {
            'total': score,
            'min_dist': min_d,
            'max_dist': max_d,
            'ideal_dist': ideal_d,
        },
        'tags': tag
    }

def format_exports(data):
    cuts = []
    chunks = []
    i = 0
    for item in data:
        cuts.append(item['cut'])

        if not item['words']:
            continue

        chunks.append({'chunk': i, 'words': item['words']})
        i += 1
    return cuts, chunks

# -------------- Operations for scoring cuts --------------

def score(start, end, min, max, ideal, silence):
    mid = mid(start, end)
    min_d = farther(mid, min) 
    max_d = farther(mid, max) 
    ideal_d = closer(mid, ideal) + normalize(silence, 0, 1)
    score = normalize(min_d + max_d + ideal_d, 0, 5)

    return score, mid, min_d, max_d, ideal_d

closer = lambda x, y: normalize(1 / (1 + abs(x - y)), 0, 1)
farther = lambda x, y: normalize(abs(x - y) / (1 + abs(x - y)), 0, 1)
mid = lambda a, b: (a + b) / 2
normalize = lambda x, min_val, max_val: 0 if max_val - min_val == 0 else (x - min_val) / (max_val - min_val)

if __name__ == "__main__":
    main()