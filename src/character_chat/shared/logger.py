import inspect
import os

def log(msg: str, level: str = "INFO", no_origin: bool = False) -> None:
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
            if not no_origin:
                raise Exception(f"{level}: {origin} {msg}")
            else:
                raise Exception(f"{level}: {msg}")
        case _:
            raise Exception("Unkown log level")
