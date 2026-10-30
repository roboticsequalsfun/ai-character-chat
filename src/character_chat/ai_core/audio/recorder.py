import pyaudio
from character_chat.shared.file_utils import load_config
from character_chat.shared.logger import log
from character_chat.shared.paths import INPUT_AUDIO_DIR
import soundfile as sf
import math
import time
import struct
import numpy as np

class AudioRecorder:
    @staticmethod
    def rms(frame):
        """Some complex math I "borrowed" from online"""

        count = len(frame) / 2
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * (1.0/32768.0)
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(
            self, chunk_size: int = 1024, channels: int = 1, 
            sample_rate: int = 48000, input: bool = True, 
            frames_per_buffer: int = 1024, format: int = pyaudio.paInt16, 
            timeout_length: int | float = 5, threshold: int | float = 10,
            output_path=INPUT_AUDIO_DIR / "recorded.wav") -> None:
        """Wait until input reaches a certain threshold then record, until silence is detected, and save recording to wav"""
        
        self.chunk_size = chunk_size
        self.output_path = str(output_path)
        self.channels = channels
        self.sample_rate = sample_rate
        self.input = input
        self.frames_per_buffer = frames_per_buffer
        self.format = format
        self.timeout_length = timeout_length
        self.threshold = threshold

        # Inintialize the stream
        self.p = pyaudio.PyAudio()

        # Open the stream
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=self.input,
            frames_per_buffer=self.frames_per_buffer
        )

    def record(self) -> None:
        """Record the audio and save it to a file"""

        log('Noise detected, recording...', "INFO")
        rec = []
        current = time.time()

        # Self.timeout_lenght is how long until it will exit if there is silence
        end = time.time() + self.timeout_length

        while current <= end:
            # Get the stream chunk
            data = self.stream.read(self.chunk_size)

            # While noise is still above threshold, push back the timeout to keep recording
            if self.rms(data) >= self.threshold: 
                end = time.time() + self.timeout_length

            current = time.time()

            # Append the stream chunk
            rec.append(data)
        log("Silence detected. Saving to file.", "INFO")

        # Write the recording to a file
        self.write(b"".join(rec))

    def write(self, recording: bytes) -> None:
        """Save recording to file"""

        # Convert raw bytes to 
        recording = np.frombuffer(recording, dtype=np.int16)

        # Write the file
        sf.write(self.output_path, recording, self.sample_rate)

        log(f"Written to file: {self.output_path}", "INFO")

    def listen(self) -> None:
        """Listen until input reaches the noise threshold then record"""
        
        log('Listening for noise (Note: now is the time to speak)...', "INFO")
        while True:
            # Get the stream chunk
            input = self.stream.read(self.chunk_size)

            # Get the intensity of the noise
            rms_val = self.rms(input)

            # If the noise is above the threshold then start recording
            # Note: this does not neccesarily mean speech
            if rms_val > self.threshold:
                self.record()
                break


def main() -> None:
    """Since we are entering by calling the script we are in testing mode"""

    # Load the config
    config = load_config("app")

    # Ininitialize the class
    recorder = AudioRecorder(
        chunk_size=config.get("audio", {}).get("chunk_size", 1024), 
        channels=config.get("audio", {}).get("channels", 1), 
        sample_rate=config.get("audio", {}).get("sample_rate", 48000), 
        input=config.get("audio", {}).get("input", True), 
        frames_per_buffer=config.get("audio", {}).get("frames_per_buffer", 1024),
        format=pyaudio.paInt16,
        timeout_length=config.get("audio", {}).get("timeout_length", 5), 
        threshold=config.get("audio", {}).get("threshold", 10),
        output_path=INPUT_AUDIO_DIR / "recorded.wav"
    )

    # Start the listening
    recorder.listen()

if __name__ == "__main__":
    main()