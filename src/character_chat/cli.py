from pathlib import Path
import subprocess
from character_chat.shared.file_utils import ensure_base_dirs
from character_chat.ai_core.ai_pipeline import AIPipeline
from character_chat.dataset_builder.data_pipeline import main as data
from character_chat.ai_core.tts.train import main as train
from character_chat.shared.logger import log
import sys
import warnings
from pick import pick

def main():
    title = "What would you like to do?: "
    options = ["Run the interactive character chat", "Prepare the dataset for TTS training", "Train the TTS model", "Exit"]

    # pick returns a tuple: (chosen_option_string, index)
    option, index = pick(options, title)

    match index:
        case 0:
            ai = AIPipeline()
            ai.setup()
            ai.run()
        case 1:
            data()
        case 2:
            train()
        case 3:
            sys.exit()

if __name__ == "__main__": 
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="torchcodec")
    warnings.filterwarnings("ignore", module="libtorchcodec")
    main()