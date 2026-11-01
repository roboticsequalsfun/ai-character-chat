
# AI Character Chat

Now, the moment no one's been waiting for 🥁...

# THE SUPER FREE INTERACTIVE CHARACTER CHAT!!!!!! 🎉🎉🎉

Okay, seriously. What is this... thing?

This is a completely free and open-source alternative to [Character.AI](https://character.ai). It also comes with a built-in Coqui TTS LJSpeech dataset builder and trainer, if you're really nerdy.

Currently, it's in the **"Works on my computer"** phase. Expect bugs. Lots of them.

## Table of Contents

- [About The Project](#about-the-project)
- [Notice](#notice)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Setup](#setup)
- [Usage](#usage)
  - [Creating a Character](#creating-a-character)
  - [Running the Project](#running-the-project)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## About The Project

AI Character Chat is a free and open-source alternative to Character.AI.

The project also includes a built-in Coqui TTS LJSpeech dataset builder and trainer, allowing you to create and train custom voice models for your characters.

The goal is to provide an interactive character-chat experience that you can run locally, customize, and expand without relying on a paid third-party service.

## Notice

This project includes tools for voice cloning and custom TTS model training.

**Please use these features responsibly.**

Voice cloning can be fun, useful, and interesting to experiment with. However, there are a few rules to keep in mind:

- **Keep cloned voices on your PC.** As a general rule of thumb, any voice you clone should stay on your own computer. Don't distribute, publish, or share cloned voice models or output of the models without appropriate authorization.
- **Don't impersonate people.** Never use a cloned voice to trick someone into believing they're talking to the actual person.
- **Don't deceive or defraud others.** Cloned voices should never be used for scams, harassment, or other malicious purposes.
- **Respect people's privacy.** Just because a voice is available online doesn't mean it should be used to mislead or harm the person behind it.

The developers of this project are not responsible for misuse of the software. By using its voice cloning features, you agree to use them responsibly and in accordance with applicable laws.

## Built With

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Coqui AI TTS](https://github.com/idiap/coqui-ai-TTS)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Hugging Face](https://huggingface.co/)

## Getting Started

Follow these instructions to get the project running locally on your machine.

### Prerequisites

Before getting started, you'll need:

- **Windows 10/11** (Linux support is planned for the future.)
- **NVIDIA GPU** (CPU support is planned for the future.)
- **[uv](https://docs.astral.sh/uv/)** (version `0.13.x` or newer)
- **[Hugging Face account](https://huggingface.co/join)**
- **[FFmpeg](https://ffmpeg.org/)** (Required for audio processing through PyDub.)

To install `uv`, run the following command in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/roboticsequalsfun/ai-character-chat
   ```

2. Navigate to the project directory:

   ```bash
   cd ai-character-chat
   ```

3. Install the project dependencies:

   ```bash
   uv sync
   ```

4. Install PyTorch with CUDA support:

   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Setup

Once the dependencies are installed, there are a few more things to configure.

1. Activate the virtual environment:

   ```powershell
   .venv\Scripts\activate
   ```

2. Log in to Hugging Face:

   ```bash
   hf auth login
   ```

   Follow the instructions to authenticate your account.

3. Create a [Hugging Face access token](https://huggingface.co/settings/tokens).

4. Add a new Windows environment variable:

   - **Variable name:** `PYANNOTE_KEY`
   - **Variable value:** Your Hugging Face access token.

5. Download a GGUF-compatible Llama model from [Hugging Face](https://huggingface.co/) and place it inside:

   ```text
   models/llm/
   ```

6. Open `configs/app.json` and set the `"model"` value under `"llm"` to the name of your downloaded model.

### Installing FFmpeg

AI Character Chat uses PyDub for audio processing, which requires FFmpeg.

1. Download FFmpeg from the [official FFmpeg website](https://ffmpeg.org/download.html).
2. Install FFmpeg for Windows.
3. Add the FFmpeg `bin` directory to your Windows `PATH` environment variable.
4. Open a new terminal and verify the installation:

   ```powershell
   ffmpeg -version

See [Getting started](https://github.com/roboticsequalsfun/ai-character-chat/wiki/Getting-started) for more details on setup.
You're now ready to go!

## Usage

### Creating a Character

Now that everything is set up, it's time to create your first character!

Each character consists of three main components:

- **Character JSON:** Contains the character's configuration and information. This is arguably the most important part, even though it looks deceptively simple.
- **Character Prompt:** Defines the character's personality, behavior, and how it interacts with users.
- **Voice Model:** Gives your character a voice.

#### Setting Up Your Character

1. Navigate to `configs/characters/`.
2. Find `template.json` and make a copy of it.
3. Rename the copied file to your character's name (e.g., `my_character.json`).
4. Open the newly created JSON file and configure the following parameters:

   - **`name`:** The name of your character.
   - **`character_prompt`:** The filename of your character's prompt `.txt` file.
   - **`voice_model`:** The path to your trained `best_model.pth` file.
   - **`training_dataset`:** The name of the dataset used to train your character's voice model.

Once you've configured everything, save the JSON file. Your character should now be ready to use!

For a more detailed guide, check out the [Creating a Character Wiki](https://github.com/roboticsequalsfun/ai-character-chat/wiki/Creating-a-character).

### Running the Project

Before running the project, make sure:

- Your microphone and speakers are selected correctly in Windows Sound Settings.
- You have a GGUF model inside `models/llm/`.
- The `"model"` value under `"llm"` in `configs/app.json` matches the name of your downloaded GGUF model.
- Your virtual environment is activated.

Once everything is ready, run the following command from the project directory:

```bash
python -m character_chat
```

You should see a menu with the following options:

1. Run the interactive character chat
2. Prepare the dataset for TTS training
3. Train the TTS model
4. Exit

Choose whichever option you want to use.

#### 1. Interactive Character Chat

To start chatting with your character:

1. Select **Run the interactive character chat**.
2. Wait for the models to load.
3. Once everything has loaded, press Enter when you're ready.
4. Follow the instructions displayed in your terminal.

**A quick tip:** When the program displays `Listening`, that's your cue to start speaking!

For information about the available configuration options, see the [Runtime Configuration Wiki](https://github.com/roboticsequalsfun/ai-character-chat/wiki/Runtime-config).

#### 2. TTS Model Training

To train a custom TTS voice model:

1. Open `tts_training.json`.
2. Make sure the `"dataset"` setting points to the correct speaker and dataset.
3. Select **Train the TTS model** from the main menu.
4. Wait for the training process to initialize.
5. I'd recommend keeping an eye on the training process for the first couple of minutes to make sure everything is running smoothly.

For a more detailed guide, see the [Training Wiki](https://github.com/roboticsequalsfun/ai-character-chat/wiki/Training).

#### 3. Dataset Builder

To prepare a dataset for TTS training:

1. Make sure you have an audio recording of the speaker whose voice you want to clone.
2. Don't worry if the recording contains multiple speakers. That's exactly what this pipeline is designed to handle!
3. Select **Prepare the dataset for TTS training** from the main menu.
4. Wait for the pipeline to finish processing the audio.

For a more detailed guide, see the [Creating a Dataset Wiki](https://github.com/roboticsequalsfun/ai-character-chat/wiki/Creating-a-dataset).

## Roadmap

Here's what I'd like to implement in the future:

- [ ] Add CPU support
- [ ] Add logging
- [ ] Add Linux support
- [ ] Support other languages
- [ ] Make TTS training available as a standalone feature

More features and improvements will be added as development continues!

## Contributing

Contributions are welcome! Whether it's fixing bugs, adding features, or improving documentation, all help is appreciated.

To contribute:

1. Fork the project.
2. Create a feature branch:

   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add some AmazingFeature"
   ```

4. Push your branch:

   ```bash
   git push origin feature/AmazingFeature
   ```

5. Open a Pull Request.

## License

Distributed under the GPL-3.0 License. See [LICENSE](LICENSE) for more information.
