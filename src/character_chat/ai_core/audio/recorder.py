import os
import queue
import sounddevice as sd
from character_chat.shared.file_utils import load_config
import numpy as np
import wavio

class AudioRecorder:
    def __init__(self, sample_rate=16000, block_size=1024, device_index=None, silence_threshold=0.5, silence_duration=0.01, min_record_time=2.0, channel=1):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device_index = device_index
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_record_time = min_record_time
        self.channel = channel
        self.audio_queue = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device_index,
            channels=self.channel,
            blocksize=self.block_size,
            callback=self.callback
        )
        
    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def close(self):
        self.stream.close()

    def get_chunk(self):
        return self.audio_queue.get()
    
    def record(self):
        self.start()

        recording = []
        silence_time = 0
        recording_started = False
        block_duration = self.block_size / self.sample_rate

        while True:
            chunk = self.get_chunk()
            chunk_energy = np.sqrt(np.mean(chunk**2))
            if not recording_started:
                if chunk_energy > self.silence_threshold:
                    recording_started = True
                    recording.append(chunk)
                continue
            
            recording.append(chunk)

            if chunk_energy < self.silence_threshold:
                silence_time += block_duration
            else:
                silence_time = 0

            if silence_time >= self.silence_duration:
                break

        self.stop()
        
        audio = np.concatenate(recording, axis=0)
        wavio.write("recorded_audio.wav", audio, self.sample_rate, sampwidth=3)

def main():
    config = load_config()
    recorder = AudioRecorder(
        sample_rate=config.get("audio", {}).get("sample_rate", 16000),
        block_size=config.get("audio", {}).get("block_size", 1024),
        device_index=config.get("audio", {}).get("device_index"),
        silence_threshold=config.get("audio", {}).get("silence_threshold", 0.5),
        silence_duration=config.get("audio", {}).get("silence_duration", 0.01),
        min_record_time=config.get("audio", {}).get("min_record_time", 2.0),
        channel=config.get("audio", {}).get("channels", 1)
    )
    recorder.record()

if __name__ == "__main__":
    main()