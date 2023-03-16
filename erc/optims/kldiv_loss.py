import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import erc


logger = erc.utils.get_logger(__name__)


class KLDiv(nn.Module):
    """ This loss function uses torch KLDivergence loss
    Since torch' KLDivLoss requires log-probability
    and our model feeds raw logits,
    this class was implemented to process all forwards at once.

    For hydra, in cli add the following to your command
    "model.criterion.cls._target_=erc.optims.kldiv_loss
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.num_classes = num_classes
        warnings.warn(f"Please note that KL-Divergence loss always do log-softmax on target.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :params:
            pred: (B, num_labels)
            target: (B, num_labels) or 
        """
        if target.ndim != 2:
            logger.warn("Given target is not (B, num_labels): %s", target)
            raise
        log_pred = F.log_softmax(pred, dim=1)
        log_target = F.log_softmax(target, dim=1)
        loss = self.kldiv(log_pred, log_target) # Scalar Tensor
        return loss
