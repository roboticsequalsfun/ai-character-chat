import inspect
import os

def log(msg, level="INFO"):
    # Get the caller's frame (2 levels up: 1 for this method, 1 for the caller)
    frame = inspect.currentframe().f_back
    filename = os.path.basename(frame.f_code.co_filename)
    line_no = frame.f_lineno

    # Format a clean origin string
    origin = f"[{filename}:{line_no}]"

    match level:
        case "INFO":
            print(msg)
        case "WARNING":
            print(f"{level}: {origin}: {msg}")
        case "ERROR":
            raise Exception(f"{level}: {origin} {msg}")
        case _:
            raise Exception("Unkown log level")
