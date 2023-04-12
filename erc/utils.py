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


def normalize_1(logits: torch.Tensor) -> torch.Tensor:
    dim = logits.ndim - 1
    _min, _ = logits.min(dim=dim)
    _min = _min.unsqueeze(dim)
    logits = (logits - _min) / (logits - _min).sum(dim).unsqueeze(dim)
    return logits


def get_gamma(p: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(float).eps
    entropy = -p * torch.log(p + eps)
    entropy = entropy.sum(dim=entropy.ndim -1)
    gamma = torch.tanh(entropy)
    return gamma
    

def apply_peakl(logits : torch.Tensor, r: float = None) -> torch.Tensor:
    """
        We build our own hard labeling function 
        reference
         - https://arxiv.org/pdf/1512.00567.pdf
         - https://proceedings.mlr.press/v162/wei22b/wei22b.pdf
    """ 
    dim = logits.ndim - 1
    n_label = logits.size()[-1]

    if r is None:
        r = get_gamma(logits).unsqueeze(dim)
    else:
        r = torch.tensor(r).unsqueeze(dim)
    logit_peakl = (logits - (r / n_label)) / (1-r)
    logit_peakl = normalize_1(F.relu(logit_peakl))
    return logit_peakl
