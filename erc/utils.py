import os
import random
from pathlib import Path
import logging
import time

import numpy as np
import torch
from torch import nn


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
