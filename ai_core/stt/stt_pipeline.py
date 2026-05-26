from ai_core.stt.recorder import main as recorder
from ai_core.stt.transcriber import main as transcriber
from ai_core.stt.session_manager import main as session_manager
from shared.file_utils import write_file, load_config, load_file

def main():
    config = load_config()

    rate = config["stt"]["rate"]
    channels = config["stt"]["channels"]
    blocksize = config["stt"]["blocksize"]
    device_index = config["stt"]["device_index"]
    start_threshold = config["stt"]["start_threshold"]
    silence_threshold = config["stt"]["silence_threshold"]
    silence_duration = config["stt"]["silence_duration"]
    min_record_time = config["stt"]["min_record_time"]
    recording_started = False
    
    recorder()
    transcriber()
    session_manager()


if __name__ == "__main__":
    main()
