import time
import os
import json
from deepgram import DeepgramClient
import whisper
from pydub import AudioSegment
from shared.paths import TEMP_DIR, INPUT_AUDIO_DIR, SEPARATED_DIR
from shared.file_utils import load_config, write_json
from shared.audio_io import load_audio_pydub, write_audio

def configure():
    # Initialize Deepgram client
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set, please create it if you haven't already and set it to your Deepgram API key")
    dg_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    # Load configuration
    config = load_config()
    input_file = INPUT_AUDIO_DIR / "Trump_WEF_2018.mp3"
    punctuate = config.get("diarization", {}).get("punctuate", True)
    diarize = config.get("diarization", {}).get("diarize", True)
    MAX_SILENCE_MS = config.get("diarization", {}).get("max_silence_ms", 1000)

    return {
        'dg_client': dg_client,
        'input_file': input_file,
        'punctuate': punctuate,
        'diarize': diarize,
        'max_silence_ms': MAX_SILENCE_MS
    }


# Function to perform diarization and return response
def diarize_audio(file_path, dg_client, punctuate, diarize):
    with open(file_path, "rb") as audio_file:
        response = dg_client.listen.v1.media.transcribe_file(
            request=audio_file,
            punctuate=punctuate,
            diarize=diarize,
        )
    return response


def reconstruct_utterances_from_words(words):
    """Group words by speaker to create utterances."""
    if not words:
        return []
    
    utterances = []
    current_utterance = None
    
    for word in words:
        speaker = word.speaker
        
        # Start new utterance if speaker changes
        if current_utterance is None or current_utterance['speaker'] != speaker:
            if current_utterance is not None:
                utterances.append(current_utterance)
            
            current_utterance = {
                'speaker': speaker,
                'start': word.start,
                'end': word.end,
                'confidence': word.speaker_confidence,
            }
        else:
            # Extend current utterance
            current_utterance['end'] = word.end
    
    # Add the last utterance
    if current_utterance is not None:
        utterances.append(current_utterance)
    
    return utterances

# Process each utterance segment and append to corresponding speaker's audio
def split_audio_by_speaker(utterances, audio, max_silence_ms):    
    speaker_audio_map = {}
    prev_end_map = {}

    for segment in utterances:
        speaker = segment['speaker']
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)

        if speaker not in speaker_audio_map:
            speaker_audio_map[speaker] = AudioSegment.empty()
            prev_end_map[speaker] = None

        # Calculate gap from previous utterance
        if prev_end_map[speaker] is not None:
            gap = start_ms - prev_end_map[speaker]

            # Only add silence if gap exists
            if gap > 0:
                # Clamp long pauses
                if gap > max_silence_ms:
                    gap = max_silence_ms

                speaker_audio_map[speaker] += AudioSegment.silent(duration=gap)

        # Add actual utterance audio
        speaker_audio_map[speaker] += audio[start_ms:end_ms]
        # Update last end
        prev_end_map[speaker] = end_ms

    return speaker_audio_map

# Save per-speaker word timings with remapped times for intelligent chunking
def save_word_timings_per_speaker(words, utterances, output_dir):
    # Build a map of utterance ranges for each speaker
    speaker_utterance_ranges = {}
    for utterance in utterances:
        speaker = utterance['speaker']
        if speaker not in speaker_utterance_ranges:
            speaker_utterance_ranges[speaker] = []
        speaker_utterance_ranges[speaker].append({
            'original_start': utterance['start'],
            'original_end': utterance['end']
        })
    
    # Calculate concatenated audio timeline for each speaker
    # Account for silence gaps between utterances
    speaker_local_timelines = {}
    for speaker, ranges in speaker_utterance_ranges.items():
        timeline = []
        current_local_time = 0
        max_silence_ms = 1000  # Default value, matches split_audio_by_speaker
        
        for i, utterance_range in enumerate(ranges):
            utterance_duration = utterance_range['original_end'] - utterance_range['original_start']
            local_start = current_local_time
            local_end = current_local_time + utterance_duration
            
            timeline.append({
                'original_start': utterance_range['original_start'],
                'original_end': utterance_range['original_end'],
                'local_start': local_start,
                'local_end': local_end
            })
            
            # Add silence gap for next utterance (if there is one)
            if i < len(ranges) - 1:
                gap = ranges[i + 1]['original_start'] - utterance_range['original_end']
                if gap > max_silence_ms:
                    gap = max_silence_ms
                current_local_time = local_end + gap
            else:
                current_local_time = local_end
        
        speaker_local_timelines[speaker] = timeline
    
    # Remap word timings to local speaker timelines
    for speaker, timeline in speaker_local_timelines.items():
        speaker_words = []
        
        for word in words:
            if word.speaker != speaker:
                continue
            
            # Find which utterance this word belongs to
            word_time = word.start
            local_offset = None
            
            for segment in timeline:
                if segment['original_start'] <= word_time < segment['original_end']:
                    # This word is in this segment
                    offset_in_segment = word_time - segment['original_start']
                    local_offset = segment['local_start'] + offset_in_segment
                    break
            
            if local_offset is not None:
                # Calculate remapped end time
                word_duration = word.end - word.start
                speaker_words.append({
                    'word': word.word,
                    'start': local_offset,
                    'end': local_offset + word_duration,
                    'speaker': speaker
                })
        
        # Save per-speaker word timings
        output_file = output_dir / f"word_timings_speaker_{speaker}.json"

        write_json(output_file, speaker_words)

def main():
    print("===================================")
    print(" Voice Track Separator Starting...")
    print("===================================")

    print("Configuring environment and loading settings...")
    config = configure()
    start_time = time.time()
    
    # Unpack config for easier access
    dg_client = config['dg_client']
    input_file = config['input_file']
    punctuate = config['punctuate']
    diarize = config['diarize']
    max_silence_ms = config['max_silence_ms']

    # Run diarization
    print("Processing audio (this may take a while)...")
    response = diarize_audio(input_file, dg_client, punctuate, diarize)
    print("Received response from Deepgram.")

    # Load original audio
    print("Separating audio by speaker(this may take a while)...")
    audio = load_audio_pydub(input_file)
    
    # Extract words and reconstruct utterances from speaker data in words
    words = response.results.channels[0].alternatives[0].words
    utterances = reconstruct_utterances_from_words(words)
    
    if not utterances:
        print("ERROR: No speaker data found in audio.")
        raise ValueError("Diarization failed - no speaker data detected")

    speaker_tracks = split_audio_by_speaker(
        utterances,
        audio,
        max_silence_ms
    )
    save_word_timings_per_speaker(words, utterances, TEMP_DIR)

    print(f"Detected speakers: {sorted(speaker_tracks.keys())}")
    print(f"Total utterances: {len(utterances)}")

    # Export audio files for each speaker
    print("Exporting speaker audio files...")

    # Export each speaker's audio
    for speaker, speaker_audio in speaker_tracks.items():
        print(f"Speaker {speaker}: {len(speaker_audio)} ms of audio")
        audio_path = SEPARATED_DIR / f"speaker_{speaker}.wav"
        write_audio(audio_path, speaker_audio, sample_rate=16000)
        print(f"Exported speaker {speaker} to {audio_path}")

    # Final summary
    elapsed = time.time() - start_time
    print("Processing complete.")
    print(f"Total speakers exported: {len(speaker_tracks)}")
    print(f"Total processing time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()