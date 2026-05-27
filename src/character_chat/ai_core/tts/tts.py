import os
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from character_chat.shared.paths import OUTPUT_AUDIO_DIR, DATASET_WAV_DIR

class TTS:
    def __init__(self):
        self.cb = ChatterboxTTS.from_pretrained(device="cpu")  # CPU mode

    def run(self, text_to_speak):
        audio_prompt_path = None
        for file_name in sorted(os.listdir(DATASET_WAV_DIR)):
            file_path = os.path.join(DATASET_WAV_DIR, file_name)
            if file_name.lower().endswith((".wav", ".mp3")) and os.path.getsize(file_path) > 0:
                try:
                    # Test loading to catch corrupted files
                    ta.load(file_path)
                    audio_prompt_path = file_path
                    print(f"[INFO] Using {file_name} as audio prompt")
                    break
                except Exception as e:
                    print(f"[WARN] Skipping corrupted audio file: {file_name} ({e})")

        if not audio_prompt_path:
            raise RuntimeError("No valid audio prompt found in folder!")
        # Generate speech
        print("[INFO] Generating speech...")
        wav = self.cb.generate(text_to_speak, audio_prompt_path=audio_prompt_path)

        # Convert directly to tensor for torchaudio
        wav_tensor = wav.cpu()  # wav is already a torch.Tensor
        ta.save(OUTPUT_AUDIO_DIR, wav_tensor, self.cb.sr)


