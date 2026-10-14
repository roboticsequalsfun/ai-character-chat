import warnings
from pathlib import Path
import torch
torch.backends.cudnn.enabled = False
import gc
import whisperx
import os
import torchaudio
from pydub import AudioSegment
from pyannote.audio import Pipeline
from df.enhance import enhance, init_df, load_audio, save_audio
from character_chat.shared.paths import INPUT_AUDIO_DIR, SPEAKER_CHUNKS_DIR, TEMP_DIR, SEPARATED_DIR, TRANSCRIPT_DIR
from character_chat.shared.file_utils import name_to_int, write_json, list_files_in_directory, list_folders_in_directory, path_exists, ensure_dir, load_config
from character_chat.shared.logger import log

def main(input=None, device="cuda", batch_size=16, compute_type="float16"):
    path = Path(INPUT_AUDIO_DIR / input)
    path_exists(path)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    log("===================================" , "INFO")
    log(" Voice Track Separator Starting...", "INFO")
    log("===================================", "INFO")

    # if input.split('.')[1] != "wav" and convert_file:
    #     log("Converting audio file to WAV...", "INFO")
    #     input = convert_audio(input, path)
    # else:
    #     input = str(path)
    # path_exists(input)

    # Load your WAV file
    audio = AudioSegment.from_wav(path)
    audio_48k = audio.set_frame_rate(48000)
    audio_48k.export(path, format="wav")
    input = path

    log("Running diarization... (This may take a while)", "INFO")

    speakers = diarize(input, device=device)
    write_json(TEMP_DIR / f"speaker_diarization.json", speakers)
    cut_audio(input, speakers=speakers)

    speakers = list_files_in_directory(SEPARATED_DIR)

    whisperx_model = whisperx.load_model(
        "large-v2",
        device,
        compute_type=compute_type,
    )

    try:
        for speaker in speakers:
            speaker_id = name_to_int(Path(speaker).stem)
            log(f"Creating word timings for speaker_{speaker_id}", "INFO")

            word_timings = create_word_timings(
                SEPARATED_DIR / f"speaker_{speaker_id}.wav", 
                device=device, 
                batch_size=batch_size, 
                model=whisperx_model
            )

            clean_audio(SEPARATED_DIR / f"speaker_{speaker_id}.wav")
            write_json(TRANSCRIPT_DIR / f"speaker_{speaker_id}" / f"word_timings.json", word_timings)
    finally:
        del whisperx_model
        gc.collect()
        torch.cuda.empty_cache()

    log("Processing complete.", "INFO")

def clean_audio(input_file):
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"GPU reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

    model, df_state, _ = init_df()
    audio, _ = load_audio(input_file, sr=df_state.sr())

    enhanced = enhance(model, df_state, audio)
    del model

    save_audio(input_file, enhanced, df_state.sr())


# def convert_audio(file_name, path):
#     command = [
#         "ffmpeg", 
#         "-i", 
#         str(path), 
#         "-ar", 
#         "16000", 
#         "-ac", 
#         "1", 
#         str(INPUT_AUDIO_DIR / file_name.split(".")[0]) + ".wav"
#     ]
    
#     try:
#         subprocess.run(command, check=True)
#         log(f"Successfully converted input file: {file_name}", "INFO")
#     except subprocess.CalledProcessError as e:
#         log(f"FFmpeg failed to process the file: {e}", "ERROR")
    
#     return str(INPUT_AUDIO_DIR / file_name.split(".")[0]) + ".wav"

def diarize(input, device):
    # Load the diarization pipeline directly
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", 
        token=os.environ.get('PYANNOTE_KEY')
    )
    diarize_model.to(torch.device(device))

    # Manually load audio into memory to avoid torchcodec file-reading issues
    waveform, sample_rate = torchaudio.load(input)
    audio_dict = {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

    # Pass the dictionary instead of the file path string
    diarization = diarize_model(audio_dict)
    
    # Extract start, end, and speaker labels
    speaker_turns = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": name_to_int(speaker)
        })
    del diarize_model

    return speaker_turns
 
def cut_audio(audio_file, speakers): 
    path_exists(audio_file)
    audio = AudioSegment.from_file(audio_file)

    for i, segment in enumerate(speakers):
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        speaker = int(segment['speaker'])

        speaker_chunk = audio[start_ms:end_ms]

        ensure_dir(SPEAKER_CHUNKS_DIR / f"speaker_{speaker}")

        speaker_chunk.export(SPEAKER_CHUNKS_DIR / f"speaker_{speaker}" / f"segment_{i}.wav", format="wav")

    speaker_folders = list_folders_in_directory(SPEAKER_CHUNKS_DIR)
    chunks = []
    for speaker in speaker_folders:
        speaker_id = name_to_int(Path(speaker).stem)
        chunks = list_files_in_directory(SPEAKER_CHUNKS_DIR / f"speaker_{speaker_id}")
        last_chunk = AudioSegment.empty()
        for i, chunk in enumerate(chunks):
            try:
                audio_chunk = AudioSegment.from_file(SPEAKER_CHUNKS_DIR / f"speaker_{speaker_id}" / chunk, format="wav")
            except:
                continue
            last_chunk += audio_chunk
        last_chunk.export(SEPARATED_DIR / f"speaker_{speaker_id}.wav", format="wav")

def create_word_timings(audio_file, device, batch_size, model):
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    return result['word_segments']


if __name__ == "__main__":
    config = load_config("dataset_builder")
    input = input("Enter the name of the input audio file (or press Enter to use default): ")
    if not input:
        input = "audio.wav"
    main(input=input, device=config["diarization"]["device"], batch_size=config["diarization"]["batch_size"], compute_type=config["diarization"]["compute_type"])