import pyaudio
from character_chat.shared.file_utils import load_config
from character_chat.shared.logger import log
from character_chat.shared.paths import INPUT_AUDIO_DIR
import wave
import math
import time
import struct

class AudioRecorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / 2
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * (1.0/32768.0)
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self, chunk_size=1024, channels=1, 
                rate=48000, input=True, 
                frames_per_buffer=1024, format=pyaudio.paInt16, 
                timeout_length=5, threshold=10,
                output_path=INPUT_AUDIO_DIR / "recorded.wav"):
        self.chunk_size = chunk_size
        self.output_path = str(output_path)
        self.channels = channels
        self.rate = rate
        self.input = input
        self.frames_per_buffer = frames_per_buffer
        self.format = format
        self.timeout_length = timeout_length
        self.threshold = threshold
        
        self.p = pyaudio.PyAudio()

        # Open stream
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=self.input,
            frames_per_buffer=self.frames_per_buffer,
        )

    def record(self):
        log('Noise detected, recording beginning', "INFO")
        rec = []
        current = time.time()
        end = time.time() + self.timeout_length

        while current <= end:
            data = self.stream.read(self.chunk_size)
            if self.rms(data) >= self.threshold: end = time.time() + self.timeout_length

            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        wf = wave.open(self.output_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(recording)
        wf.close()
        log('Written to file: {}'.format(self.output_path), "INFO")



    def listen(self):
        log('Listening beginning', "INFO")
        while True:
            input = self.stream.read(self.chunk_size)
            rms_val = self.rms(input)
            if rms_val > self.threshold:
                self.record()
                break


def main():
    config = load_config("app")
    recorder = AudioRecorder(
        chunk_size=config.get("audio", {}).get("chunk_size", 1024), 
        channels=config.get("audio", {}).get("channels", 1), 
        rate=config.get("audio", {}).get("rate", 48000), 
        input=config.get("audio", {}).get("input", True), 
        frames_per_buffer=config.get("audio", {}).get("frames_per_buffer", 1024),
        format=pyaudio.paInt16,
        timeout_length=config.get("audio", {}).get("timeout_length", 5), 
        threshold=config.get("audio", {}).get("threshold", 10),
        output_path=INPUT_AUDIO_DIR / "recorded.wav"
    )

    recorder.listen()

if __name__ == "__main__":
    main()