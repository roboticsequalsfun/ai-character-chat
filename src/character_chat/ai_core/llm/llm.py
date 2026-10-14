from llama_cpp import Llama
import re
from character_chat.shared.paths import LLM_MODELS_DIR, PROMPT_DIR, CONFIG_DIR
from character_chat.shared.file_utils import load_config, load_json

class LLM:
    def __init__(self, model, n_gpu_layers=-1, n_ctx=2048, low_vram=False, max_tokens=60, temperature=0.7, top_p=0.9, repeat_penalty=1.15, stop=None):
        self.conversation_history = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.stop = stop
        self.model = Llama(
            model_path=str(LLM_MODELS_DIR / model),
            n_gpu_layers=n_gpu_layers,  # put all layers on GPU
            low_vram=low_vram,
            n_ctx=n_ctx,
            verbose=False
        )

    def respond(self, user_input, base_prompt):
        # Always use fresh prompt with only last user input
        base_prompt

        messages = [
            {"role": "system", "content": base_prompt},
        ]

        messages.extend(self.conversation_history[-4:])

        messages.append({"role": "user", "content": user_input})

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stop=self.stop
        )

        reply = response["choices"][0]["message"]["content"]

        # Strip out anything wrapped in asterisks (actions, stage directions, etc.)
        cleaned_reply = re.sub(r'\*.*?\*', '', reply).strip()
        cleaned_reply = re.sub(r'\s+', ' ', cleaned_reply)

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": cleaned_reply})
        return cleaned_reply

def configure_testing():
    config = load_config("app")

    character_config = config.get("active_character", "character")
    character = load_json(CONFIG_DIR / f"characters" / f"{character_config}.json")
    character_name = character.get("name", "character")

    character_prompt = (PROMPT_DIR / character.get("character_prompt", "prompts/character/character_prompt.txt")).read_text(encoding="utf-8")
    system_prompt = (PROMPT_DIR / character.get("system_prompt", "prompts/character/system_prompt.txt")).read_text(encoding="utf-8")
    system_prompt = system_prompt.format(character_name=character_name)
    prompt = system_prompt + "\n" + character_prompt

    return {
        "model_name": config.get("llm", {}).get("model"),
        "n_gpu_layers": config.get("llm", {}).get("n_gpu_layers", -1),
        "n_ctx": config.get("llm", {}).get("n_ctx", 2048),
        "low_vram": config.get("llm", {}).get("low_vram", False),
        "prompt": prompt,
        "max_tokens": config.get("llm", {}).get("max_tokens", 60),
        "temperature": config.get("llm", {}).get("temperature", 0.7),
        "top_p": config.get("llm", {}).get("top_p", 0.9),
        "repeat_penalty": config.get("llm", {}).get("repeat_penalty", 1.15),
        "stop": config.get("llm", {}).get("stop", None)
    }    

if __name__ == "__main__":
    config = configure_testing()

    llm = LLM(
        model=config["model_name"],
        n_gpu_layers=config["n_gpu_layers"],
        low_vram=config["low_vram"],
        n_ctx=config["n_ctx"],
        max_tokens=config["max_tokens"], 
        temperature=config["temperature"], 
        top_p=config["top_p"], 
        repeat_penalty=config["repeat_penalty"], 
        stop=config["stop"]
    )

    while True:
        text = input("You: ")
        if text.lower() == "exit":
            break
        print(llm.respond(text, base_prompt=config["prompt"]))