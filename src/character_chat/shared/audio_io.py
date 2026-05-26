import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

def load_audio_librosa(path):
    return librosa.load(path, sr=None)

def load_audio_pydub(path):
    return AudioSegment.from_file(path)

def write_audio(path, audio_data, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio_data, AudioSegment):
        audio_data.export(path, format="wav")

    elif isinstance(audio_data, np.ndarray):
        sf.write(path, audio_data, sample_rate)

    else:
        raise TypeError(f"Unsupported audio type: {type(audio_data)}")