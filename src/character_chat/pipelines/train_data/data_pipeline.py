from character_chat.pipelines.train_data.speaker_separator import main as run_separator
from character_chat.pipelines.train_data.audio_chunker import main as run_splitter
from character_chat.pipelines.train_data.dataset_builder import main as run_dataset_builder
from character_chat.shared.file_utils import load_config
from character_chat.shared.logger import log

config = load_config()

run_separator()
run_splitter()
run_dataset_builder()

log("All processing complete. Dataset is ready for training!", "INFO")