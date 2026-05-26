# PeppaVoice_train.py
# Run: .venv\Scripts\python.exe PeppaVoice_train.py

import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

if __name__ == "__main__":
    # -------------------- USER SETTINGS --------------------
    dataset_csv = os.path.abspath("metadata_fixed.csv")  # LJSpeech-style CSV
    output_path = os.path.abspath("output_peppa")       # Folder to save checkpoints
    base_model_name = "tts_models/en/vctk/vits"        # Pretrained base model
    epochs = 25                                        # Number of training epochs
    batch_size = 4                                    # Adjust for your GPU memory
    use_cuda = True                                    # True/False
    speaker_name = "peppa"                             # Single-speaker
    # -------------------------------------------------------

    os.makedirs(output_path, exist_ok=True)

    # ----- Dataset Config -----
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train=dataset_csv,
    )

    # ----- Training Config -----
    config = VitsConfig(
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        epochs=epochs,
        output_path=output_path,
        datasets=[dataset_config],
        # Force single-speaker to simplify inference
        speakers=["peppa"],  
        multi_speaker=False
    )

    # ----- Initialize Audio Processor -----
    ap = AudioProcessor.init_from_config(config)

    # ----- Initialize Tokenizer -----
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # ----- Load Dataset Samples -----
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=0.015,  # 1.5% of dataset for eval
    )

    # ----- Initialize Model -----
    # Provide a speaker manager to avoid garbage audio during inference
    model = Vits(config, ap, tokenizer)

    # ----- Initialize Trainer -----
    trainer_args = TrainerArgs()
    trainer = Trainer(
        trainer_args,
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )

    # ----- Start Training -----
    trainer.fit()