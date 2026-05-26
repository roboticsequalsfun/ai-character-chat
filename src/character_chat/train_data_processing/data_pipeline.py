from character_chat.train_data_processing.speaker_separator import main as run_separator
from character_chat.train_data_processing.audio_chunker import main as run_splitter
from character_chat.train_data_processing.dataset_builder import main as run_dataset_builder
from character_chat.shared.file_utils import load_config

config = load_config()

run_separator()
run_splitter()
run_dataset_builder()

print("All processing complete. Dataset is ready for training!")