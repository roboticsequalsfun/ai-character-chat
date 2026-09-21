import torch
import subprocess
import warnings
from pathlib import Path
import sys
from pathlib import Path
from pydub import AudioSegment
from pyannote.audio.utils.reproducibility import ReproducibilityWarning
from pyannote.audio import Pipeline
from character_chat.shared.paths import INPUT_AUDIO_DIR, TEMP_DIR
from character_chat.shared.file_utils import load_config, write_json
from character_chat.shared.audio_io import load_audio_pydub, write_audio
from character_chat.shared.logger import log

def main(input_file_name=None, punctuate=True, diarize=True, max_silence_ms=1000):
    warnings.filterwarnings("ignore", category=ReproducibilityWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    path = Path(INPUT_AUDIO_DIR / input_file_name)

    log("===================================" , "INFO")
    log(" Voice Track Separator Starting...", "INFO")
    log("===================================", "INFO")

    if not path.is_file():
        log(f"File {input_file_name} does not exist", "ERROR")
        sys.exit()

    if input_file_name.split(".")[1] != "wav":
        log("Converting audio file to WAV...", "INFO")
        input = convert_audio(input_file_name, path)
    else:
        input = str(path)

    log("Running diarization... (This may take a while)", "INFO")

    if diarize:
        diarize_audio(input)
    log("Processing complete.", "INFO")

def convert_audio(file_name, path):
    command = [
        "ffmpeg", 
        "-i", 
        str(path), 
        "-ar", 
        "16000", 
        "-ac", 
        "1", 
        str(INPUT_AUDIO_DIR / file_name.split(".")[0]) + ".wav"
    ]
    
    try:
        subprocess.run(command, check=True)
        log(f"Successfully converted input file: {file_name}", "INFO")
    except subprocess.CalledProcessError as e:
        log(f"FFmpeg failed to process the file: {e}", "ERROR")
    
    return str(INPUT_AUDIO_DIR / file_name.split(".")[0]) + ".wav"

def diarize_audio(input):
    speakers = get_speakers(input)
    write_json(TEMP_DIR / f"speakers.json", speakers)
    export_audio(speakers)

def get_speakers(input):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token="PyannoteToken")

    # send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda"))
    diarization = pipeline(input)

    speakers = []
    last_speaker = -1
    last_end = -1
    group = 0

    for turn, _, speaker_name in diarization.speaker_diarization.itertracks(yield_label=True):
        speaker = int(speaker_name.split("_")[1])
        if speaker != last_speaker:
            if not group in speakers:
                speakers.append({})
            if not group == 0:
                speakers[group - 1].append({
                    'stop': last_end, 
                    'speaker': last_speaker
                })
            speakers[group].append({
                'start': round(turn.start, 1), 
                'speaker': speaker
            })
            group += 1
        last_end = round(turn.end, 1)
        last_speaker = speaker
    return speakers

def export_audio(speakers, audio_file):
    audio = AudioSegment.from_file(audio_file)
    chunks = []

    last
    last_cut_ms = 0
    for speakers in speakers:
        cut_ms = int(speakers['location'] * 1000)

        chunks.append(audio[last_cut_ms:cut_ms])

        last_cut_ms = cut_ms


if __name__ == "__main__":
    input = input("Enter the name of the input audio file (or press Enter to use default): ")
    if not input:
        input = "audio.wav"
    main(input_file_name=input)