import torch
import platform

def get_device() -> torch.device:
    """Select the best available device: MPS (Mac), CUDA (GPU), or CPU."""
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
