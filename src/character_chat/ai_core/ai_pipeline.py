from datetime import datetime
from character_chat.ai_core.llm.llm import LLM
from character_chat.ai_core.stt.stt import STT
from character_chat.ai_core.tts.tts import TTS
from character_chat.ai_core.audio.player import AudioPlayer
from character_chat.ai_core.audio.recorder import AudioRecorder
from character_chat.shared.file_utils import load_config
from character_chat.shared.paths import PROMPT_DIR, OUTPUT_AUDIO_DIR, INPUT_AUDIO_DIR

class AIPipeline:
    def __init__(self):
        self.config = self.build_runtime_config()
        self.llm = None
        self.stt = None
        self.tts = None
        self.audio_player = None
        self.audio_recorder = None
        self.audio_path = INPUT_AUDIO_DIR / "recorded.wav"
        self.output_audio_path = OUTPUT_AUDIO_DIR / f"response_{datetime.now().timestamp()}.wav"

    def build_runtime_config(self):
        config =load_config()  # Ensure config is loaded before accessing it
        character_name = config.get("llm", {}).get("character_name", "Character")

        user_prompt = (PROMPT_DIR / "user_prompt.txt").read_text(encoding="utf-8")
        system_prompt = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
        system_prompt = system_prompt.format(character_name=character_name)
        prompt = system_prompt + "\n" + user_prompt

        return {
            "llm": {
                "model_name": config.get("llm", {}).get("model"),
                "character_name": character_name,
                "n_gpu_layers": config.get("llm", {}).get("n_gpu_layers", -1),
                "low_vram": config.get("llm", {}).get("low_vram", False),
                "max_tokens": config.get("llm", {}).get("max_tokens", 60),
                "temperature": config.get("llm", {}).get("temperature", 0.7),
                "top_p": config.get("llm", {}).get("top_p", 0.9),
                "stop": config.get("llm", {}).get("stop"),
            },

            "audio": {
                "sample_rate": config.get("audio", {}).get("sample_rate", 16000),
                "block_size": config.get("audio", {}).get("block_size", 1024),
                "device_index": config.get("audio", {}).get("device_index"),
                "channels": config.get("audio", {}).get("channels", 1),
                "silence_threshold": config.get("audio", {}).get("silence_threshold", 0.02),
                "silence_duration": config.get("audio", {}).get("silence_duration", 1.0),
                "min_record_time": config.get("audio", {}).get("min_record_time", 2.0),
            },

            "tts": {
                "model_path": config.get("tts", {}).get("model_path")
            },

            "stt": {
                "model_size": config.get("stt", {}).get("model", "large")
            },

            "runtime": {
                "prompt": prompt
            }
        }
    
    def setup(self):
        self.audio_recorder = AudioRecorder(
            sample_rate=self.config["audio"]["sample_rate"],
            block_size=self.config["audio"]["block_size"],
            device_index=self.config["audio"]["device_index"],
            silence_threshold=self.config["audio"]["silence_threshold"],
            silence_duration=self.config["audio"]["silence_duration"],
            min_record_time=self.config["audio"]["min_record_time"],
            channels=self.config["audio"]["channels"],
            output_path=self.audio_path
        )
        self.llm = LLM(
            model=self.config["llm"]["model_name"],
            n_gpu_layers=self.config["llm"]["n_gpu_layers"],
            low_vram=self.config["llm"]["low_vram"]
        )
        self.stt = STT(model_size=self.config["stt"]["model_size"])
        self.tts = TTS(
            model_path=self.config["tts"]["model_path"],
        )
        self.audio_player = AudioPlayer()

    def run_llm(self, user_input):
        response = self.llm.respond(
            character=self.config["llm"]["character_name"],
            user_input=user_input,
            base_prompt=self.config["runtime"]["prompt"],
            max_tokens=self.config["llm"]["max_tokens"],
            temperature=self.config["llm"]["temperature"],
            top_p=self.config["llm"]["top_p"],
            stop=self.config["llm"]["stop"]
        )
        return response
    
    def run_stt(self, audio_path):
        return self.stt.transcribe(audio_path=audio_path)
    
    def run_tts(self, text, output_path):
        self.tts.synthesize(text, output_path)

    def play_audio(self, audio_path):
        self.audio_player.play(audio_path=audio_path)

    def record_audio(self):
        self.audio_recorder.record()

if __name__ == "__main__":
    pipeline = AIPipeline()
    pipeline.setup()

    while True:
        command = input()
        pipeline.record_audio()
        transcript = pipeline.run_stt(pipeline.audio_path)
        output = pipeline.run_llm(transcript)
        pipeline.run_tts(output, pipeline.output_audio_path)
        pipeline.play_audio(pipeline.output_audio_path)
        if command.lower() == "exit":
            break
