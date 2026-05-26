import whisper
from tqdm import tqdm

# Load larger model for accuracy (small is fast, large is most accurate)
model = whisper.load_model("large")  # options: tiny, base, small, medium, large

def main():
    var = "loremispum to prevent errors until this is implemented"

if __name__ == "__main__":
    main()