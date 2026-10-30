from pathlib import Path
from pydub import AudioSegment
from character_chat.shared.paths import CHUNKS_DIR, SEPARATED_DIR, TEMP_DIR, TRANSCRIPT_DIR
import character_chat.shared.file_utils as file_utils
from character_chat.shared.logger import log
from typing import TypedDict

# -------------- Dict templates --------------

class Wieghts(TypedDict):
    min: int | float
    max: int | float
    ideal: int | float
    silence: int | float

class Score(TypedDict):
    total: int | float
    min_dist: int | float
    max_dist: int | float
    ideal_dist: int | float

class CutTemplate(TypedDict):
    location: int | float
    index: int
    score: Score
    tags: str

class Words(TypedDict):
    word: float
    duration: float
    start: float
    end: float
    index: int

class WordsTemplate(TypedDict):
    sentence: str
    words: Words


def main(
        max_chunk_duration: int | float = 10.0, 
        min_chunk_duration: int | float = 5.0, 
        ideal_chunk_duration: int | float = 7.0, 
        min_silence_duration: int | float = 0.01, 
        weights=None) -> None:
    log("===================================" , "INFO")
    log(" Audio Chunker Starting...", "INFO") 
    log("===================================\n", "INFO")

    speaker_files = file_utils.list_files_in_directory(SEPARATED_DIR)

    if speaker_files is None:
        log(f"No speakers found in: {SEPARATED_DIR}. Ensure the input audio file is not empty.", "WARNING")

    for speaker_file in speaker_files:
        speaker_id = file_utils.split_string(Path(speaker_file).stem)
        log(f"============ Speaker {speaker_id} ============", "INFO")

        audio_file = SEPARATED_DIR / f"speaker_{speaker_id}.wav"
        file_utils.path_exists(audio_file)

        log(f"Chunking Audio...", "INFO")
        word_timings = file_utils.load_json(TRANSCRIPT_DIR / f"speaker_{speaker_id}" / f"word_timings.json")

        cuts, words = define_chunks(word_timings, 
            max_chunk_duration=max_chunk_duration, 
            min_chunk_duration=min_chunk_duration, 
            ideal_chunk_duration=ideal_chunk_duration, 
            min_silence_duration=min_silence_duration,
            weights=weights
        )

        file_utils.write_json(TRANSCRIPT_DIR / f"speaker_{speaker_id}" / f"word_data.json", words)
        file_utils.write_json(TEMP_DIR / f"cuts_speaker_{speaker_id}.json", cuts)

        amount = cut_audio(audio_file, cuts=cuts, path=CHUNKS_DIR / f"speaker_{speaker_id}")
        log(f"[DONE] {amount} chunks saved for speaker {speaker_id}\n ", "INFO")

    log("===================================", "INFO")
    log(f"[DONE] All chunks saved ", "INFO")
    log("===================================", "INFO")


def define_chunks(
        word_timings: dict, 
        max_chunk_duration: float | int, 
        min_chunk_duration: float | int, 
        ideal_chunk_duration: float | int, 
        min_silence_duration: float | int, 
        weights: Wieghts) -> tuple[CutTemplate, WordsTemplate]:
    """
    Create intelligent chunks. Takes in word timings, loops through each word, and defines a list of 
    possible cuts at silence points, based on the min and max chunk duration. Each one is scored based on how close it is 
    to the ideal time, how far from the min and max, and the silence duration. It then returns a list of final cuts, and words within them

    Key arguments:
        word_timings -- The list of words in the audio. Critical for the function to run
        max_chunk_duration -- The maximum time in seconds the chunk can be
        min_chunk_duration -- The minimum time in seconds the chunk can be
        ideal_chunk_duration -- The ideal time the chunk should be
        min_silence_duration -- The minimum duration of silence can be accepted for a cut there
        weights -- A dictonary of weights that the algorithm multiplies the score values, to define importance. E.g. silence should be more, 
            since it is much more inportant it doesn't cut off words, then being close to the ideal time
    """

    # -------------- Setup variables --------------

    # Null catch for empty word_timings
    if not word_timings:
        return []
    
    cuts = [] 
    words = []
    max_chunk = []
    last_marker = 0
    i = 0
    chunk_duration = 0
    last_word_end = 0

    # We use window markers because its MUCH easier to calculate
    markers = [min_chunk_duration + last_marker, max_chunk_duration + last_marker, ideal_chunk_duration + last_marker] 

    # -------------- Loop through all the words --------------

    while i < len(word_timings):
        # Usually python does this for us, but I'm using a while loop (since I need to be able to change the index)
        word = word_timings[i]

        # Only at the begginning 
        if not max_chunk:
            chunk_start_time = word['start']

        chunk_duration = word['end'] - chunk_start_time

        # -------------- Define the chunks --------------
        if chunk_duration >= max_chunk_duration:
            possible_cuts = []
            last_word_end = last_marker

            for item in max_chunk:
                silence_duration = item['start'] - last_word_end # Subtract from last word end. Not the word's end

                if silence_duration >= min_silence_duration:
                    # -------------- Calculate scores --------------
                    total_score, mid, min, max, ideal = score(item['start'], item['end'], markers[0], markers[1], markers[2], silence_duration, weights)
                    if mid > markers[0] and mid < markers[1]:
                        possible_cuts.append(cut_template(mid, item['index'], "normal", total_score, min, max, ideal))

                # Update the last word end time for the next iteration
                last_word_end = item['end']

            # Null catch for no possible cuts found, force a cut at the last word
            if not possible_cuts:
                mid = max_chunk[-1]['end']
                fallback_index = max_chunk[-1]['index']
                possible_cuts.append(
                    # We just pass in "dummy values", and make sure we specify its a fallback
                    cut_template(mid, fallback_index, "default", 1, 0, 0, 0)
                )

            # -------------- Select the cut with the highest score --------------
            highest_value = -1  # Now anything can be the highest value
            chosen_cut = None
            
            for cut in possible_cuts:
                if cut['score']['total'] > highest_value:
                    highest_value = cut['score']['total']
                    chosen_cut = cut

            # -------------- Define that chunks word's --------------
            words_in_chunk = []
            for word in max_chunk:
                if word['index'] >= chosen_cut['index']:
                    break
                words_in_chunk.append(word)

            cuts.append(chosen_cut)
            words.append({
                "sentence": ' '.join(word['word'] for word in words_in_chunk),
                "words": words_in_chunk
            })

            # -------------- Reset and, go back to the iteration of the cut --------------
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

        # Append the word to the "max_chunk", which is the physical max the chunk is allowed to be
        # This is mostly because its easier to make the algorithm this way
        max_chunk.append({
            'word': word['word'], 
            'duration': word['end'] - word['start'], 
            'start': word['start'], 
            'end': word['end'],
            'index': i
        })  
        i += 1

    # -------------- Add last end cuts --------------
    # Handle any remaining words after the loop
    if max_chunk:
        words.append({
            "sentence": ' '.join(word['word'] for word in max_chunk),
            "words": max_chunk
        })

    # Make a cut at the end to signify the end of the audio
    cuts.append(cut_template(
        max_chunk[-1]['end'], 
        max_chunk[-1]['index'], 
        "end", 1, 0, 0, 0)) # We just pass in dummy values
    
    return cuts, words


def cut_audio(audio_file: str | Path, cuts: CutTemplate, path: str | Path) -> int: 
    """Cut the speaker's audio into chunks based on the cuts, and return how many chunks there are"""

    file_utils.ensure_dir(path)
    file_utils.path_exists(audio_file)
    audio = AudioSegment.from_file(audio_file)

    last_cut_ms = 0
    for i, cut in enumerate(cuts):
        cut_ms = int(cut['location'] * 1000)

        chunk = audio[last_cut_ms:cut_ms]
        chunk.export(path / f"chunk_{i:04}.wav", format="wav")

        last_cut_ms = cut_ms

    return i


def cut_template(
        location: int | float, index: int, 
        tag: str, score: int | float, 
        min_d: int | float, max_d: int | float, 
        ideal_d: int | float) -> CutTemplate:
    """I made this because it was so annoying to use this everywhere. Its just a template for the dict"""

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


# -------------- Operations for scoring cuts --------------

def score(
        start: int | float, end: int | float, 
        min_chunk: int | float, max_chunk: int | float, 
        ideal: int | float, silence: int | float, 
        weights: Wieghts = None) -> tuple[float, float, float, float, float]:
    """Score the cuts based on the input values"""
    
    mid_point = mid(start, end)
    min_d = farther(mid_point, min_chunk)*weights.get("min", 1) if weights else 1
    max_d = farther(mid_point, max_chunk)*weights.get("max", 1) if weights else 1
    ideal_d = closer(mid_point, ideal)*weights.get("ideal", 1) if weights else 1
    score = min_d + max_d + ideal_d + (silence*weights.get("silence", 1) if weights else 1)

    return score, mid_point, min_d, max_d, ideal_d


def closer(x: int | float, y: int | float) -> float: return normalize(1 / (1 + abs(x - y)), 0, 1)
def farther (x: int | float, y: int | float) -> float: return normalize(abs(x - y) / (1 + abs(x - y)), 0, 1)
def mid (a: int | float, b: int | float) -> float: return (a + b) / 2
def normalize (
        x: int | float, 
        min_val: int | float, 
        max_val: int | float) -> float: 
    return 0 if max_val - min_val == 0 else (x - min_val) / (max_val - min_val)


if __name__ == "__main__":
    config = file_utils.load_config("dataset_builder")

    main(
        min_chunk_duration=config["chunk_audio"]["min_chunk_duration"],
        max_chunk_duration=config["chunk_audio"]["max_chunk_duration"],
        ideal_chunk_duration=config["chunk_audio"]["ideal_chunk_duration"],
        min_silence_duration=config["chunk_audio"]["min_silence_duration"],
        weights=config["chunk_audio"]["weights"]
    )