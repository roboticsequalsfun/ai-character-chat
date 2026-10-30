from pathlib import Path
import torch
torch.backends.cudnn.enabled = False
import gc
import whisperx
import os
import torchaudio
from pydub import AudioSegment
os.add_dll_directory(r"C:\ffmpeg\bin") 
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from df.enhance import enhance, init_df, load_audio, save_audio
from character_chat.shared.paths import INPUT_AUDIO_DIR, SPEAKER_CHUNKS_DIR, TEMP_DIR, SEPARATED_DIR, TRANSCRIPT_DIR, PROCESSED_DIR
import character_chat.shared.file_utils as file_utils
from character_chat.shared.logger import log
from typing import TypedDict

class Turns(TypedDict):
    start: float
    end: float
    speaker: int

class SpeakerSeparator:
    def __init__(
            self, input_file: str | Path = None, device: str = "cuda", 
            batch_size: int = 16, compute_type: str = "float16",
            target_level: int | float = -20.0, clean: bool = True) -> None:
        """Load the models and run the cleaning, diarization, and transcribing"""

        file_utils.path_exists(input_file)
        self.input_file = self.convert_audio(input_file)
        file_utils.path_exists(self.input_file)

        if device == "cuda":
            if not torch.cuda.is_available():
                log("CUDA not available. Either enable CUDA, or use cpu", "WARNING")
        self.device = device
        self.batch_size = batch_size
        self.target_level = target_level

        log("===================================" , "INFO")
        log(" Voice Separator Starting...", "INFO")
        log("===================================", "INFO")

        # ------------------- Clean Audio -------------------
        
        if clean:
            # DeepFilterNet model for cleaning audio
            self.df_model, self.df_state, _ = init_df()

            log("Cleaning audio...")
            self.input_file = self.clean_audio()
            del self.df_model, self.df_state, _
            torch.cuda.empty_cache()

        # Always normalize audios so TTS doesn't talk quite sometimes and loud others
        self.input_file = self.normalize_audio()
        file_utils.path_exists(self.input_file)

        # ------------------- Load Models -------------------

        log("Loading models (This may take a while)...", "INFO")

        try:
            # Whisperx model for transcriptions
            with torch.no_grad():
                self.whisperx_model = whisperx.load_model(
                    "large-v2",  # I use large since there is no time sensitivity here
                    self.device,
                    compute_type=compute_type,
                )
            torch.cuda.empty_cache()

            # Use os.getenv so we can catch if the key doesn't exist
            token = os.getenv('PYANNOTE_KEY')
            if token is None:
                log("Token: PYANNOTE_KEY was not found. Ensure it exists in your environment variables, and has a value", "ERROR")

            # Pyannote.audio model for Diarization
            with torch.no_grad():
                self.diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.0",  # This is the free model
                    token=token
                )
            torch.cuda.empty_cache()
        except Exception as e:
            log(e, "ERROR")

        if self.whisperx_model is None:
            log("Failed to load whisperx model", "ERROR")
        if self.diarization_model is None:
            log("Failed to load pyannote model", "ERROR")

        # ------------------- Diarize Audio -------------------

        log("Running diarization (This may take a while)...", "INFO")

        speakers_timestamps = self.diarize()
        file_utils.write_json(TEMP_DIR / f"speaker_diarization.json", speakers_timestamps)

        self.cut_audio(speakers=speakers_timestamps)

        try:
            speakers = file_utils.list_files_in_directory(SEPARATED_DIR)
            if speakers is None:
                log(f"No speakers found in: {SEPARATED_DIR}. Ensure the input audio file is not empty.", "WARNING")
            for speaker in speakers:
                speaker_id = file_utils.split_string(
                    str(
                        file_utils.split_string(
                            Path(speaker), 
                            breaker=".",
                            module=0,
                            is_int=False
                        )
                    ), 
                    breaker="_")
                log(f"Creating word timings for speaker_{speaker_id} (This may take a while)...", "INFO")

                word_timings = self.create_word_timings(speaker)
                file_utils.write_json(TRANSCRIPT_DIR / f"speaker_{speaker_id}" / f"word_timings.json", word_timings)
        finally:
            # Delete the huge models manually
            del self.whisperx_model
            del self.diarization_model

            # And emty the cache
            gc.collect()
            torch.cuda.empty_cache()

        log("[DONE] Audio diarized and transcribed.", "INFO")
         

    def convert_audio(self, input: Path | str) -> Path:
        """Converts the inputed path to a wav with 48k frame rate, and returns the new path"""

        path = Path(input)
        file_utils.path_exists(path)

        audio = AudioSegment.from_file(path)
        audio_48k = audio.set_frame_rate(48000)  # DeepFilterNet needs 48k
        audio_48k.export(input, format="wav")

        # I get the original file path. Get just the file, then I throw out the file type, and add the .wav
        path = INPUT_AUDIO_DIR / (str(str(input.name).split(".")[0]) + ".wav")
        file_utils.path_exists(path)

        return path
    
    
    def clean_audio(self) -> Path:
        """Remove background noises and preserve only human speech, and returns the new path"""

        # Anything with running models should be wrapped in try-except blocks
        try:
            audio, _ = load_audio(self.input_file, sr=self.df_state.sr())
            enhanced = enhance(self.df_model, self.df_state, audio)
        except Exception as e:
            log(e, "ERROR")

        save_audio(PROCESSED_DIR / "cleaned.wav", enhanced, self.df_state.sr())
        file_utils.path_exists(PROCESSED_DIR / "cleaned.wav")

        return PROCESSED_DIR / "cleaned.wav"
    

    def normalize_audio(self) -> Path:
        """Normalize the audio to the same rough noise level"""

        audio = AudioSegment.from_wav(self.input_file)

        # We get the change in db and then apply increase or decrease it to average it out
        change_in_dBFS = self.target_level - audio.dBFS
        normalized = audio.apply_gain(change_in_dBFS)

        normalized.export(PROCESSED_DIR / "cleaned.wav", format="wav")
        file_utils.path_exists(PROCESSED_DIR / "cleaned.wav")

        return PROCESSED_DIR / "cleaned.wav"
    

    def diarize(self) -> Turns:
        """Diarize audio and return the speaker turns"""

        try:
            self.diarization_model.to(torch.device(self.device))

            waveform, sample_rate = torchaudio.load(self.input_file)
            audio_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            diarization = self.diarization_model(audio_dict)
        except Exception as e:
            log(e, "ERROR")
        
        speaker_turns = []
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            speaker_turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": file_utils.split_string(speaker, breaker="_")
            })

        return speaker_turns
    
    
    def cut_audio(self, speakers: Turns) -> None: 
        """Cut the audio file based on the speaker_turns, then recombine all the chunks back into their respective speakers"""

        audio = AudioSegment.from_file(self.input_file)

        # We go through each turn, cut that out of the audio and export it into temp
        for i, segment in enumerate(speakers):

            # Turns are in seconds so times it by 1000
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            speaker = int(segment['speaker'])

            speaker_chunk = audio[start_ms:end_ms]

            file_utils.ensure_dir(SPEAKER_CHUNKS_DIR / f"speaker_{speaker}")

            speaker_chunk.export(SPEAKER_CHUNKS_DIR / f"speaker_{speaker}" / f"segment_{i}.wav", format="wav")

        # Now we combine all those chunks back together into their respective speaker
        speaker_folders = file_utils.list_folders_in_directory(SPEAKER_CHUNKS_DIR)
        chunks = []

        for speaker in speaker_folders:
            speaker_id = file_utils.split_string(Path(speaker).stem)
            chunks = file_utils.list_files_in_directory(SPEAKER_CHUNKS_DIR / f"speaker_{speaker_id}")

            speaker_audio = AudioSegment.empty()
            for i, chunk in enumerate(chunks):
                # We use a try except block since they are labled then put into their respective speaker folder
                # Therefore some numbers are not there
                try:
                    audio_chunk = AudioSegment.from_file(SPEAKER_CHUNKS_DIR / f"speaker_{speaker_id}" / chunk, format="wav")
                except:
                    continue

                speaker_audio += audio_chunk
            speaker_audio.export(SEPARATED_DIR / f"speaker_{speaker_id}.wav", format="wav")


    def create_word_timings(self, speaker: str | Path) -> dict:
        """Create the word timings for the current speaker"""
        try:
            audio = whisperx.load_audio(SEPARATED_DIR / speaker)
            result = self.whisperx_model.transcribe(audio, batch_size=self.batch_size) # If the batch_size is to large it may run out of memory

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        except Exception as e:
            log(e, "ERROR")

        return result['word_segments'] # Only return the words not the sentences (Since I do my own chunking)


if __name__ == "__main__":
    config = file_utils.load_config("dataset_builder")
    input = str(input("Enter the name of the input audio file: "))

    separator = SpeakerSeparator(input_file=INPUT_AUDIO_DIR / input, 
        device=config["diarization"]["device"], 
        batch_size=config["diarization"]["batch_size"], 
        compute_type=config["diarization"]["compute_type"], 
        clean=config["clean_audio"],
        target_level=config["diarization"]['normalize_audio']["target_level"]
    )