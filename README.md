# AI character chat
Now. The moment no ones been waiting for🥁... THE SUPER FREE INTERACTIVE CHARACTER CHAT!!!!!!🎉🎉🎉🎉🎉🎉

Okay, seriously. What is this... thing?
This is a fully open source and free alternative to character.ai, and comes with a built in Coqui-TTS LJSpeech dataset builder if your really nerdy.

Currently it is in the "Works on my computer" phase. Expect many bugs.

## Table of Contents

- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Support & Contact](#support--contact)

## About The Project

This is a free and opensource alternative to character.ai. It also comes with a built in Coqui-TTS LJSpeech dataset builder, and trainer.

## Built With

* [Python](https://www.python.org/)

## Getting Started

Follow these instructions to set up the project locally on your machine.

### Prerequisites

What you need:
* `uv >= 0.13.X`
  
  ```ps1
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
* [HuggingFace Account](https://huggingface.co/login)
* An NVIDIA GPU (Plan to support CPU to)
* Windows 10/11

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/roboticsequalsfun/ai-character-chat
   ```
2. Navigate into the project directory:
   ```bash
   cd ai-character-chat
   ```
3. Set up the environment:
   ```bash
   uv sync
   ```
4. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Set up
1. Run `.venv\Scripts\activate`.
2. Run `hf auth login` and follow the instructions to set up Hugging Face.
3. Create an [access token](https://huggingface.co/settings/tokens).
4. Add a new Environment Variable (on your PC) called `PYANNOTE_KEY` and add the name of your Token.
5. Download a GGUF Llama model from [Hugging Face](https://huggingface.co).

## Usage

Provide code snippets or commands showing how to run or use the project:

```bash
python -m character_chat
```

Explain expected output or basic configuration parameters.

---

## Roadmap

- [ ] Add support for CPU
- [ ] Add logging
- [ ] Add support for Linux
- [ ] Make TTS training standalone as well

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`gitosh origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the GNU License. See [LICENSE](LICENSE) for more information.

---

## Support & Contact

* **Issues:** [GitHub Issues](https://github.com)

