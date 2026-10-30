from .cli import main
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="torchcodec")
    warnings.filterwarnings("ignore", module="libtorchcodec")
    main()