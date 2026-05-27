import torchaudio as ta
import simpleaudio as sa
from character_chat.shared.paths import OUTPUT_AUDIO_DIR
from pathlib import Path
import torch

class AudioPlayer:
    def __init__(self, audio_path=None):
        self.audio_path = OUTPUT_AUDIO_DIR / "output.wav" if audio_path is None else Path(audio_path)

    def play(self):
        # Load waveform and sample rate
        waveform, sr = ta.load(self.audio_path)  # waveform: [channels, samples], float32 [-1.0,1.0]

        # Convert to PCM16 for simpleaudio
        pcm16 = (waveform * 32767).to(torch.int16).T  # transpose to [samples, channels]

        # Interleave and convert to bytes
        audio_bytes = pcm16.contiguous().numpy().tobytes()

        # Play audio
        wave_obj = sa.WaveObject(audio_bytes, num_channels=pcm16.shape[1], bytes_per_sample=2, sample_rate=sr)
        play_obj = wave_obj.play()
        play_obj.wait_done()