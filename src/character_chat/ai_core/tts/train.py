import os
from pathlib import Path
from trainer import Trainer, TrainerArgs
from character_chat.shared.file_utils import load_config, ensure_dir, path_exists
from character_chat.shared.paths import TTS_MODEL_DIR, DATASET_DIR, TTS_MODEL_DIR
from character_chat.shared.logger import log
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.utils.manage import ModelManager

def download_files() -> None:
    """Download neccesary files for fine-tuning model"""

    checkpoints_path = str(TTS_MODEL_DIR / "xtts_v2.0.1_checkpoints")
    ensure_dir(checkpoints_path)

    # -------------- Download DVAE files --------------
    dvae_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    mel_norm_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    dvae_checkpoint = Path(checkpoints_path) / os.path.basename(dvae_link)
    mel_norm_file = Path(checkpoints_path) / os.path.basename(mel_norm_link)

    if not dvae_checkpoint.is_file() or not mel_norm_file.is_file():
        log("Downloading DVAE files...", "INFO")
        ModelManager._download_model_files([mel_norm_link, dvae_link], checkpoints_path, progress_bar=True)

    # -------------- Download model files --------------
    tokenizer_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    checkpoint_link = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    tokenizer_file = Path(checkpoints_path) / os.path.basename(tokenizer_link)  # vocab.json file
    xtts_checkpoint = Path(checkpoints_path) / os.path.basename(checkpoint_link)  # model.pth file

    # download XTTS v2.0 files if needed
    if not tokenizer_file.is_file() or not xtts_checkpoint.is_file():
        log("Downloading XTTS v2.0 files...", "INFO")
        ModelManager._download_model_files(
            [tokenizer_link, checkpoint_link], checkpoints_path, progress_bar=True
        )

    # -------------- Safe check --------------
    path_exists(dvae_checkpoint)
    path_exists(mel_norm_file)
    path_exists(tokenizer_file)
    path_exists(xtts_checkpoint)

    return str(dvae_checkpoint), str(mel_norm_file), str(tokenizer_file), str(xtts_checkpoint)


def main() -> None:
    """Fine tune the model"""

    log("===================================" , "INFO")
    log("Training TTS Model...", "INFO") 
    log("===================================\n", "INFO")
    
    # Download the neccesary files
    dvae_checkpoint, mel_norm_file, tokenizer_file, xtts_checkpoint = download_files()

    # -------------- Load the custom config --------------
    tts_config = load_config("tts_training")

    formatter = tts_config.get("dataset", {}).get("formatter", "ljspeech")
    dataset_name = tts_config.get("dataset", {}).get("dataset_name", "trump")
    speaker = tts_config.get("dataset", {}).get("speaker", 0)
    meta_file_train = tts_config.get("dataset", {}).get("meta_file_train", "speaker_0.csv")

    betas = tts_config.get("optimizer_params", {}).get("betas", [0.9, 0.96])
    eps = tts_config.get("optimizer_params", {}).get("eps", 1e-8)
    weight_decay = tts_config.get("optimizer_params", {}).get("weight_decay", 1e-2)
    milestones = tts_config.get("lr_scheduler_params", {}).get("milestones", [900000, 2700000, 5400000])
    gamma = tts_config.get("lr_scheduler_params", {}).get("gamma", 0.5)
    last_epoch = tts_config.get("lr_scheduler_params", {}).get("last_epoch", -1)

    run_name = tts_config.get("run_name", "trump_voice")
    project_name = tts_config.get("project_name", "trump_training")
    run_description = tts_config.get("run_description", "GPT XTTS training")
    epochs = tts_config.get("epochs", 200)
    dashboard_logger = tts_config.get("dashboard_logger", "tensorboard")
    logger_uri = tts_config.get("logger_uri", None)
    batch_size = tts_config.get("batch_size", 3)
    batch_group_size = tts_config.get("batch_group_size", 3)
    eval_batch_size = tts_config.get("eval_batch_size", 3)
    eval_split_max_size = tts_config.get("eval_split_max_size", 256)
    num_loader_workers = tts_config.get("num_loader_workers", 8)
    print_step = tts_config.get("print_step", 50)
    plot_step = tts_config.get("plot_step", 100)
    log_model_step = tts_config.get("log_model_step", 1000)
    save_step = tts_config.get("save_step", 10000)
    save_n_checkpoints = tts_config.get("save_n_checkpoints", 1)
    save_checkpoints = tts_config.get("save_checkpoints", True)
    print_eval = tts_config.get("print_eval", False)
    optimizer = tts_config.get("optimizer", "AdamW")
    optimizer_wd_only_on_weights = tts_config.get("optimizer_wd_only_on_weights", True)
    lr = tts_config.get("lr", 5e-06)
    lr_scheduler = tts_config.get("lr_scheduler", "MultiStepLR")
    start_with_eval = tts_config.get("start_with_eval", True)
    grad_accum_steps = tts_config.get("grad_accum_steps", 84)
    sentences = tts_config.get("test_sentences", ["Hello world", "This is a test sentence.", "How are you doing today?"])

    max_conditioning_length = tts_config.get("model_args", {}).get("max_conditioning_length", 132300)
    min_conditioning_length = tts_config.get("model_args", {}).get("min_conditioning_length", 66150)
    debug_loading_failures = tts_config.get("model_args", {}).get("debug_loading_failures", False)
    max_wav_length = tts_config.get("model_args", {}).get("max_wav_length", 255995)
    max_text_length = tts_config.get("model_args", {}).get("max_text_length", 200)
    gpt_num_audio_tokens = tts_config.get("model_args", {}).get("gpt_num_audio_tokens", 1026)
    gpt_start_audio_token = tts_config.get("model_args", {}).get("gpt_start_audio_token", 1024)
    gpt_stop_audio_token = tts_config.get("model_args", {}).get("gpt_stop_audio_token", 1025)
    gpt_use_masking_gt_prompt_approach = tts_config.get("model_args", {}).get("gpt_use_masking_gt_prompt_approach", True)
    gpt_use_perceiver_resampler = tts_config.get("model_args", {}).get("gpt_use_perceiver_resampler", True)

    sample_rate = tts_config.get("audio", {}).get("sample_rate", 22050)
    dvae_sample_rate = tts_config.get("audio", {}).get("dvae_sample_rate", 22050)
    output_sample_rate = tts_config.get("audio", {}).get("output_sample_rate", 24000)

    output_path = str(TTS_MODEL_DIR / run_name)

    # -------------- Setup the Dataset config --------------
    dataset_config = BaseDatasetConfig(
        formatter=formatter,
        dataset_name=dataset_name,
        path=DATASET_DIR / dataset_name,
        meta_file_train=meta_file_train,
        language="en",
    )

    dataset_config = [dataset_config]

    # The more the merrier! 
    speaker_references = [
        str(DATASET_DIR / dataset_name / "wavs" / f"speaker_{speaker}" / "chunk_0000.wav"),
        str(DATASET_DIR / dataset_name / "wavs" / f"speaker_{speaker}" / "chunk_0001.wav"),
        str(DATASET_DIR / dataset_name / "wavs" / f"speaker_{speaker}" / "chunk_0002.wav"),
        str(DATASET_DIR / dataset_name / "wavs" / f"speaker_{speaker}" / "chunk_0003.wav"),
    ]

    # -------------- Setup the built in configs --------------
    # Init args and config
    model_args = GPTArgs(
        max_conditioning_length=max_conditioning_length,
        min_conditioning_length=min_conditioning_length,
        debug_loading_failures=debug_loading_failures,
        max_wav_length=max_wav_length,
        max_text_length=max_text_length,
        mel_norm_file=mel_norm_file,
        dvae_checkpoint=dvae_checkpoint,
        xtts_checkpoint=xtts_checkpoint,
        tokenizer_file=tokenizer_file,
        gpt_num_audio_tokens=gpt_num_audio_tokens,
        gpt_start_audio_token=gpt_start_audio_token,
        gpt_stop_audio_token=gpt_stop_audio_token,
        gpt_use_masking_gt_prompt_approach=gpt_use_masking_gt_prompt_approach,
        gpt_use_perceiver_resampler=gpt_use_perceiver_resampler,
    )

    # Define audio config
    audio_config = XttsAudioConfig(sample_rate=sample_rate, dvae_sample_rate=dvae_sample_rate, output_sample_rate=output_sample_rate)
    test_sentences = []
    for sentence in sentences:
        test_sentences.append({
            "text": sentence,
            "speaker_wav": speaker_references,
            "language": "en", # I'm only doing english for v0.1.0. I will change this probably
        })

    # Training parameters config
    config = GPTTrainerConfig(
        output_path=output_path,
        model_args=model_args,
        run_name=run_name,
        project_name=project_name,
        run_description=run_description,
        dashboard_logger=dashboard_logger,
        logger_uri=logger_uri,
        audio=audio_config,
        batch_size=batch_size,
        batch_group_size=batch_group_size,
        epochs=epochs,
        eval_batch_size=eval_batch_size,
        num_loader_workers=num_loader_workers,
        eval_split_max_size=eval_split_max_size,
        print_step=print_step,
        plot_step=plot_step,
        log_model_step=log_model_step,
        save_step=save_step,
        save_n_checkpoints=save_n_checkpoints,
        save_checkpoints=save_checkpoints,
        print_eval=print_eval,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer=optimizer,
        optimizer_wd_only_on_weights=optimizer_wd_only_on_weights,
        optimizer_params={"betas": betas, "eps": eps, "weight_decay": weight_decay},
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params={"milestones": milestones, "gamma": gamma, "last_epoch": last_epoch},
        test_sentences=test_sentences,
    )

    # -------------- Run the model --------------

    # Init the model from config
    model = GPTTrainer.init_from_config(config)

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            # XTTS checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=start_with_eval,
            grad_accum_steps=grad_accum_steps,
        ),
        config,
        output_path=output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()