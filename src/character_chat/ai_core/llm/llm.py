from llama_cpp import Llama
import re
from character_chat.shared.paths import LLM_MODELS_DIR, PROMPT_DIR, CONFIG_DIR, CHARACTER_DIR
from character_chat.shared.file_utils import load_config, load_json, load_file
from character_chat.shared.logger import log
from typing import TypedDict

class Config(TypedDict):
    model_name = str
    n_gpu_layers = int
    n_ctx = int
    low_vram = bool
    prompt = str
    max_tokens = int
    temperature = float | int
    top_p = float | int
    repeat_penalty = float | int
    stop = bool
    memory = int

# We use a class so it can store the model more efficiently
class LLM:
    def __init__(self, model_name: str, n_gpu_layers: int = -1, n_ctx: int = 4096, 
            low_vram: bool = False, max_tokens: str = 60, temperature: int| float = 0.7, 
            top_p: int | float = 0.9, repeat_penalty: int | float = 1.15, 
            stop: bool = None, memory: int = -4) -> None:
        """Generate a response from inputed text, using a model stored in a .gguf file.
        Key arguments
            model_name -- The model.gguf
            n_gpu_layers -- How many layers are put onto the GPU. -1 for all, 0 for none (CPU only)
            n_ctx -- How many tokens the model can hold in its memory (includes both prompt, user input, and response)
            repeat_penalty -- How sensitive it is to the LLM repeating. 
                Higher the number, the more it will be a unique answer. 
                Not two high or else the model will not talk normally
            memory -- How may responses from user and llm the model remebers (in negative form)
            max_tokens -- How many "words" the model can say in one response.
                A general rule of thumb is that every token is equal to 4 characters.
        """

        self.conversation_history = []
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.stop = stop
        self.memory = memory

        # Load the model
        log("Loading Llama model (This may take a while)...")
        self.model = Llama(
            model_path=str(LLM_MODELS_DIR / model_name),
            n_gpu_layers=n_gpu_layers,
            low_vram=low_vram,
            n_ctx=n_ctx,
            verbose=False
        )

    def respond(self, user_input: str, prompt: str) -> str:
        """Create the response based on the users input
        We use the create_chat_completion system to prevent the LLM, Fabulous Assisant:,
        Sucky User:, label headache when responding.
        """
        
        # Create the prompt for the LLM, in the role, content system
        messages = [
            {"role": "system", "content": str(prompt)},
        ]
        messages.append({"role": "user", "content": str(user_input)})

        # Add the last responses (from both the user and the LLM) to the message so it has a bit of memory
        messages.extend(self.conversation_history[self.memory:])

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            stop=self.stop
        )

        # Response returns a dict, that looks like this: 
        # "choices": [
        #     {
        #         "message": {
        #             "role": "assistant",
        #             "content": "Hello! How can I help you?"
        #         }
        #     }
        # ]

        # We want the actual response ("content"). (Probably in the future going to make it smarter)
        reply = response["choices"][0]["message"]["content"]

        # Strip out anything wrapped in asterisks like stage directions, emotions,
        # or [breathes heavily making the entire world shake under his pure godlike power] 
        # garbage because it breaks the TTS
        cleaned_reply = re.sub(r"[\(\[\*].*?[\)\]\*]", "", reply)

        self.conversation_history.append({"role": "user", "content": str(user_input)})
        self.conversation_history.append({"role": "assistant", "content": cleaned_reply})

        return cleaned_reply  # Use the cleaned reply

def configure_testing() -> Config:
    """Since the LLM script requires so many parameters, 
    we must load the config as we would do normally
    """
    
    config = load_config("app")

    # Get the active character
    character = load_json(CONFIG_DIR / f"characters" / f"{config.get("active_character", "character")}.json")

    # Find the prompts specified in the character json
    character_prompt = (CHARACTER_DIR / character.get("character_prompt", "character.txt")).read_text(encoding="utf-8")
    system_prompt = (PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")
    prompt = character_prompt + "\n" + system_prompt  # Combine them all as one
    
    prompt.format(character_name=character.get("name", "character")) # Replace all occurances of {character_name} with the characters name

    # Return all of the parameters for easy access
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
        "stop": config.get("llm", {}).get("stop", None),
        "memory": config.get("llm", {}).get("memory", -4)
    }    

if __name__ == "__main__":
    # Since we are entering by calling the script we are in testing mode
    # We load the config
    config = configure_testing()

    # Load the LLM model
    llm = LLM(
        model_name=config["model_name"],
        n_gpu_layers=config["n_gpu_layers"],
        low_vram=config["low_vram"],
        n_ctx=config["n_ctx"],
        max_tokens=config["max_tokens"], 
        temperature=config["temperature"], 
        top_p=config["top_p"], 
        repeat_penalty=config["repeat_penalty"], 
        stop=config["stop"],
        memory=config["memory"]
    )

    # Now we can talk with it!
    while True:
        # Takes the input
        text = input("You (type 'exit' to quit): ")

        # If it says exit we stop the script
        if text.lower() == "exit":
            break

        # Else we pass it to the LLM and print its response
        print(llm.respond(text, prompt=config["prompt"]))