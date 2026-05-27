from llama_cpp import Llama
from pathlib import Path
from character_chat.shared.paths import LLM_MODELS_DIR, TRANSCRIPT_DIR, PROMPT_DIR
from character_chat.shared.file_utils import load_config

class LLM:
    def __init__(self, model, n_gpu_layers=-1, low_vram=False):
        self.model = Llama(
            model_path=str(LLM_MODELS_DIR / model),
            n_gpu_layers=n_gpu_layers,  # put all layers on GPU
            low_vram=low_vram,
            verbose=False
        )

    def respond(self, character, user_input, base_prompt, max_tokens=60, temperature=0.7, top_p=0.9, stop=None):
        # Always use fresh prompt with only last user input
        prompt = base_prompt + f"\nUser: {user_input}\n{character}: "

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )

        reply = response["choices"][0]["text"].strip()
        return reply

def configure_testing():
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

if __name__ == "__main__":
    config = configure_testing()

    llm = LLM(
        model=config["model_name"],
        n_gpu_layers=config["n_gpu_layers"],
        low_vram=config["low_vram"]
    )

    while True:
        text = input("You: ")
        if text.lower() == "exit":
            break
        print(llm.respond(config["character_name"], text, base_prompt=config["prompt"], max_tokens=config["max_tokens"], temperature=config["temperature"], top_p=config["top_p"], stop=config["stop"]))