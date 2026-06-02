from character_chat.shared.paths import DATASET_DIR, MODELS_DIR
from character_chat.shared.file_utils import load_config
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# LOAD CONFIG
tts_config = load_config()
batch_size = tts_config.get("tts_training", {}).get("batch_size", 32)
eval_batch_size = tts_config.get("tts_training", {}).get("eval_batch_size", 16)
num_loader_workers = tts_config.get("tts_training", {}).get("num_loader_workers", 4)
num_eval_loader_workers = tts_config.get("tts_training", {}).get("num_eval_loader_workers", 4)
run_eval = tts_config.get("tts_training", {}).get("run_eval", True)
test_delay_epochs = tts_config.get("tts_training", {}).get("test_delay_epochs", -1)
epochs = tts_config.get("tts_training", {}).get("epochs", 1000)
text_cleaner = tts_config.get("tts_training", {}).get("text_cleaner", "phoneme_cleaners")
use_phonemes = tts_config.get("tts_training", {}).get("use_phonemes", True)
phoneme_language = tts_config.get("tts_training", {}).get("phoneme_language", "en-us")
print_step = tts_config.get("tts_training", {}).get("print_step", 25)
print_eval = tts_config.get("tts_training", {}).get("print_eval", False)
mixed_precision = tts_config.get("tts_training", {}).get("mixed_precision", True)
model_name = tts_config.get("tts_training", {}).get("model_name", "glow_tts")

# DEFINE MODEL OUTPUT PATH
speaker = tts_config.get("tts_training", {}).get("speaker_id", 0)
model_dir = MODELS_DIR / "tts" / f"{model_name}_speaker_{speaker}"
if not model_dir.exists():
    model_dir.mkdir(parents=True, exist_ok=True)

# DEFINE DATASET CONFIG
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", meta_file_train=DATASET_DIR / f"metadata_speaker_{speaker}.csv", path=DATASET_DIR
)

# INITIALIZE THE TRAINING CONFIGURATION
config = GlowTTSConfig(
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    num_loader_workers=num_loader_workers,
    num_eval_loader_workers=num_eval_loader_workers,
    run_eval=run_eval,
    test_delay_epochs=test_delay_epochs,
    epochs=epochs,
    text_cleaner=text_cleaner,
    use_phonemes=use_phonemes,
    phoneme_language=phoneme_language,
    phoneme_cache_path=MODELS_DIR / "tts" / "phoneme_cache",
    print_step=print_step,
    print_eval=print_eval,
    mixed_precision=mixed_precision,
    output_path=model_dir,
    datasets=[dataset_config],
)

# INITIALIZE THE AUDIO PROCESSOR
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# INITIALIZE THE MODEL
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE TRAINER
trainer = Trainer(
    TrainerArgs(), config, model_dir, model=model, train_samples=train_samples, eval_samples=eval_samples
)

def main():
    trainer.fit()

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()