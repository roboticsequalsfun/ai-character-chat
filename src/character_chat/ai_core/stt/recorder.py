import os
from shared.paths import OUTPUT_TEXT_DIR, OUTPUT_AUDIO_DIR, TEMP_DIR, OUTPUT_TEXT_DIR, TRANSCRIPT_DIR
import numpy as np
import queue
import sounddevice as sd

def record(sample_rate, block_size, device_index, start_threshold, silence_threshold, silence_duration, min_record_time):
    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        blocksize=block_size,
        callback=callback
    )

    print("Streaming audio...")

    with stream:
        while True:
            audio_chunk = audio_queue.get()
            print("Chunk received:", audio_chunk.shape)
    
def main(sample_rate=16000, block_size=1024, device_index=None, start_threshold=0.01, silence_threshold=0.005, silence_duration=1.0, min_record_time=2.0):
    record(
        sample_rate=sample_rate,
        block_size=block_size,
        device_index=device_index,
        start_threshold=start_threshold,
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        min_record_time=min_record_time
    )

if __name__ == "__main__":
    main()
