from character_chat.ai_core.llm.llm import LLM
from character_chat.shared.file_utils import load_config
from character_chat.shared.paths import PROMPT_DIR, TRANSCRIPT_DIR

class AIPipeline:
    def __init__(self):
        self.config = self.build_runtime_config()
        self.llm = None

    def build_runtime_config(self):
        config =load_config()  # Ensure config is loaded before accessing it
        character_name = config.get("llm", {}).get("character_name", "Character")

        user_prompt = (PROMPT_DIR / "user_prompt.txt").read_text(encoding="utf-8")
        system_prompt = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
        system_prompt = system_prompt.format(character_name=character_name)
        prompt = system_prompt + "\n" + user_prompt

        file_path = TRANSCRIPT_DIR / "transcription.txt"

        return {
            "model_name": config.get("llm", {}).get("model"),
            "character_name": character_name,
            "n_gpu_layers": config.get("llm", {}).get("n_gpu_layers", -1),
            "low_vram": config.get("llm", {}).get("low_vram", False),
            "prompt": prompt,
            "file_path": file_path,
            "max_tokens": config.get("llm", {}).get("max_tokens", 60),
            "temperature": config.get("llm", {}).get("temperature", 0.7),
            "top_p": config.get("llm", {}).get("top_p", 0.9),
            "stop": config.get("llm", {}).get("stop", None)
        }    
    
    def setup(self):
        self.llm = LLM(
            model=self.config["model_name"],
            n_gpu_layers=self.config["n_gpu_layers"],
            low_vram=self.config["low_vram"]
        )

    def run_llm(self, user_input):
        response = self.llm.respond(
            character=self.config["character_name"],
            user_input=user_input,
            base_prompt=self.config["prompt"],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            stop=self.config["stop"]
        )
        return response

def main():
    pipeline = AIPipeline()
    pipeline.setup()

if __name__ == "__main__":
    main()
