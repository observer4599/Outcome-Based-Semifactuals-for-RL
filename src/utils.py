import numpy as np
import random
import torch
from typing import Optional


def seed_everything(seed: int, deterministic: Optional[bool] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic is not None:
        torch.backends.cudnn.deterministic = deterministic


def get_device(cuda: bool) -> torch.device:
    if cuda:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"
    return torch.device(device)
