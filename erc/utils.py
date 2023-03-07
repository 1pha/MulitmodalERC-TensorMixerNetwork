import os
from pathlib import Path
import logging
import time

from torch import nn


def get_folds(num_session: int = 20, num_folds = 5) -> dict:
    """ Return a sequential fold split information
    For KEMDy19: 20 sessions
    For KEMDy20_v_1_1: 40 sessions """
    ns, div = num_session, num_folds
    num_sessions: list = [ns // div + (1 if x < ns % div else 0)  for x in range (div)]
    fold_dict = dict()
    for f in range(num_folds):
        s = sum(num_sessions[:f])
        e = s + num_sessions[f]
        fold_dict[f] = range(s + 1, e + 1) # Because sessions starts from 1
    return fold_dict


def get_logger(filehandler: bool = False):
    logging.basicConfig()
    logger = logging.getLogger(__name__)
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