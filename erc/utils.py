import os
import random
from pathlib import Path
import logging
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_logger(name: str = None, filehandler: bool = False):
    name = name or __name__
    logging.basicConfig()
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    if filehandler:
        fname = f"{time.strftime('%Y%m%d-%H%M', time.localtime())}.log"
        logger.addHandler(logging.FileHandler(filename=fname))
    return logger


def check_exists(path: str | Path) -> str | Path:
    assert os.path.exists(path), f"{path} does not exist."
    return path


def count_parameters(model: nn.Module):
    # Reference
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # If the above line is uncommented, we get the following RuntimeError:
    #  max_pool3d_with_indices_backward_cuda does not have a deterministic implementation
    torch.backends.cudnn.benchmark = False

def reverse_soft(
        y_soft : torch.Tensor,
        r: int = 0.3
    ) -> torch.Tensor:
    """
        We build our own hard labeling function 
        idea source1: https://arxiv.org/pdf/1512.00567.pdf
        idea source2: https://proceedings.mlr.press/v162/wei22b/wei22b.pdf#page=11&zoom=100,384,889
    """
    if y_soft.ndim == 1:
        n_label = len(y_soft[y_soft > 0])
        return F.relu((y_soft - (r/n_label)) / (1-r))
    elif y_soft.ndim == 2:
        n_label = (y_soft > 0).sum(dim=1)
        return F.relu((y_soft - (r / n_label).unsqueeze(1)) / (1-r))
    else:
        assert "plz check your tensor dim ... "