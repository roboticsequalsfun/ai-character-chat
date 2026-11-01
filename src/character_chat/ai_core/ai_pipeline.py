from datetime import datetime
from pathlib import Path
from character_chat.ai_core.llm.llm import LLM
from character_chat.ai_core.stt.stt import STT
from character_chat.ai_core.tts.tts import TTS
from character_chat.ai_core.audio.player import play as play_audio
from character_chat.ai_core.audio.recorder import AudioRecorder
from character_chat.shared.logger import log
from character_chat.shared.file_utils import load_config, load_json
from character_chat.shared.paths import CHARACTER_DIR, PROMPT_DIR, OUTPUT_AUDIO_DIR, INPUT_AUDIO_DIR, CONFIG_DIR, TTS_MODEL_DIR, DATASET_DIR

class AIPipeline:
    def __init__(self) -> None:
        """Interactive AI character chat pipeline"""

        self.config = self.build_runtime_config()
        self.llm = None
        self.stt = None
        self.tts = None
        self.audio_recorder = None
        self.input_audio_path = str(INPUT_AUDIO_DIR / f"recorded_{datetime.now().timestamp()}.wav")
        self.output_audio_path = str(OUTPUT_AUDIO_DIR / f"response_{datetime.now().timestamp()}.wav")

    def run(self) -> None:
        """Run the loop"""

        text = input("Press Enter to continue start the interactive chat...")

        log("\n===================================" , "INFO")
        log("Starting Interactive Character Chat...", "INFO") 
        log("===================================\n", "INFO")

        while True:
            self.record_audio()
            transcript = self.run_stt(self.input_audio_path)
            log(f"Creating response (with input \"{transcript}\")...", "INFO")
            output = self.run_llm(transcript)
            log(f"Converting \"{output}\" to speech...", "INFO")
            self.run_tts(output, self.output_audio_path)
            play_audio(audio_path=self.output_audio_path)

    def build_runtime_config(self) -> dict:
        config = load_config("app")  # Ensure config is loaded before accessing it
        character_name = config.get("llm", {}).get("character_name", "Character")

        character_config = config.get("active_character", "character")
        character = load_json(CONFIG_DIR / f"characters" / f"{character_config}.json")
        character_name = character.get("name", "character")
    
        character_prompt = (CHARACTER_DIR / character.get("character_prompt", "character.txt")).read_text(encoding="utf-8")
        system_prompt = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")

        prompt = character_prompt + "\n" + system_prompt
        prompt.format(character_name=character_name)

        return {
            "active_character": character_config,

            "llm": {
                "model_name": config.get("llm", {}).get("model"),
                "n_gpu_layers": config.get("llm", {}).get("n_gpu_layers", -1),
                "low_vram": config.get("llm", {}).get("low_vram", False),
                "max_tokens": config.get("llm", {}).get("max_tokens", 100),
                "prompt": prompt,
                "n_ctx": config.get("llm", {}).get("n_ctx", 2048),
                "repeat_penalty": config.get("llm", {}).get("repeat_penalty", 1.15),
                "temperature": config.get("llm", {}).get("temperature", 0.7),
                "top_p": config.get("llm", {}).get("top_p", 0.9),
                "stop": config.get("llm", {}).get("stop", None),
            },

            "tts": {
                "model": character.get("voice_model", "xtts_speaker_0/best_model.pth"),
                "dataset": character.get("training_dataset", "character"),
                "temperature": config.get("tts", {}).get("temperature", 0.65),
                "length_penalty": config.get("tts", {}).get("length_penalty", 1.0),
                "repetition_penalty": config.get("tts", {}).get("repetition_penalty", 10.0),
                "top_k": config.get("tts", {}).get("top_k", 50),
                "top_p": config.get("tts", {}).get("top_p", 0.8)
            },

            "audio": {
                "chunk_size": config.get("audio", {}).get("chunk_size", 960),
                "channels": config.get("audio", {}).get("channels", 1), 
                "sample_rate": config.get("audio", {}).get("sample_rate", 44100),
                "frames_per_buffer": config.get("audio", {}).get("frames_per_buffer", 960),
                "threshold": config.get("audio", {}).get("threshold", 20),
                "timeout_length": config.get("audio", {}).get("timeout_length", 2.5),
                "input": config.get("audio", {}).get("input", True),
            },

            "stt": {
                "model_size": config.get("stt", {}).get("model", "large")
            },
        }
    
    def setup(self) -> None:
        """Setup all the models"""

        log("===================================" , "INFO")
        log("Setting Up Models...", "INFO") 
        log("===================================\n", "INFO")

        self.audio_recorder = AudioRecorder(
            chunk_size=self.config["audio"]["chunk_size"], 
            channels=self.config["audio"]["channels"], 
            sample_rate=self.config["audio"]["sample_rate"],
            input=self.config["audio"]["input"], 
            frames_per_buffer=self.config["audio"]["frames_per_buffer"],  
            timeout_length=self.config["audio"]["timeout_length"], 
            threshold=self.config["audio"]["threshold"],
            output_path=self.input_audio_path
        )
        self.llm = LLM(
            model_name=self.config["llm"]["model_name"],
            n_gpu_layers=self.config["llm"]["n_gpu_layers"],
            low_vram=self.config["llm"]["low_vram"],
            n_ctx=self.config["llm"]["n_ctx"],
            max_tokens=self.config["llm"]["max_tokens"], 
            temperature=self.config["llm"]["temperature"], 
            top_p=self.config["llm"]["top_p"], 
            repeat_penalty=self.config["llm"]["repeat_penalty"], 
            stop=self.config["llm"]["stop"]
        )
        self.stt = STT(model_size=self.config["stt"]["model_size"])
        self.tts = TTS(
            model_path=TTS_MODEL_DIR / self.config["tts"]["model"],
            dataset=DATASET_DIR / self.config["tts"]["dataset"],
            temperature=self.config["tts"]["temperature"],
            length_penalty=self.config["tts"]["length_penalty"],
            repetition_penalty=self.config["tts"]["repetition_penalty"],
            top_k=self.config["tts"]["top_k"],
            top_p=self.config["tts"]["top_p"],
            sample_rate=self.config["audio"]["sample_rate"]
        )

    def run_llm(self, user_input: str) -> str:
        response = self.llm.respond(
            user_input=user_input,
            prompt=self.config["llm"]["prompt"]
        )
        return response
    
    def run_stt(self, audio_path: str | Path) -> str:
        return self.stt.transcribe(audio_path=audio_path)
    
    def run_tts(self, text: str, output_path: str | Path):
        self.tts.synthesize(text, output_path)

    def record_audio(self) -> None:
        self.audio_recorder.listen()

if __name__ == "__main__":
    log("Initializing models (this may take a while)...")
    pipeline = AIPipeline()
    pipeline.setup()

    while not input == "start":
        input = input("All model's set up! Type \"start\" to start chatting: ")

    pipeline.run()
