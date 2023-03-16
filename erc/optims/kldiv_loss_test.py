import torch
import pytest

from erc.optims.kldiv_loss import KLDiv
import erc


logger = erc.utils.get_logger(__name__)


def test_kldiv():
    batch_size, num_classes = 2, 4
    pred = torch.tensor([
        [-0.98, 0.2, 0.2, 0.6],
        [0.1, -0.7, 0.6, 0.2],
    ], dtype=torch.float)
    criterion = KLDiv(num_classes=num_classes)
    
    # Target (B, num_classes), normalized (softmax)
    target = torch.tensor([
        [0.9, 0.1, 0., 0.],
        [0., 0.1, 0.8, 0.1],
    ], dtype=torch.float)
    loss = criterion(pred=pred, target=target)
    # assert torch.isclose(loss, torch.tensor(1.2693))
    

    # Target (B, num_classes), not normalized
    target = torch.tensor([
        [9, 1, 0, 0],
        [0, 1, 8, 1],
    ], dtype=torch.float)
    loss = criterion(pred=pred, target=target)
    # assert torch.isclose(loss, torch.tensor(1.2693))

    # Target (B, ) -> should raise error
    target = torch.tensor([0, 3], dtype=torch.int)
    with pytest.raises(Exception):
        loss = criterion(pred=pred, target=target)
