import whisper

class STT:
    def __init__(self, model_size="large", audio_path=None):
        # Load larger model for accuracy (small is fast, large is most accurate)
        self.model_size = model_size
        self.model = whisper.load_model(model_size)  # options: tiny, base, small, medium, large
        self.audio_path = audio_path

    def transcribe(self):
        result = self.model.transcribe(self.audio_path)
        return result['text']

def main():
    stt = STT()
    while True:
        input = input("Enter path to audio file for transcription: ")
        if input.lower() == "exit":
            break
        stt.audio_path = input
        result = stt.transcribe()
        print(result)

if __name__ == "__main__":
    main()
