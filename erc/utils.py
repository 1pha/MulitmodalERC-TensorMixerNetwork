import os
from pathlib import Path

from torch import nn


def check_exists(path: str | Path) -> str | Path:
    assert os.path.exists(path), f"{path} does not exist."
    return path


def count_parameters(model: nn.Module):
    # Reference
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    return sum(p.numel() for p in model.parameters() if p.requires_grad)