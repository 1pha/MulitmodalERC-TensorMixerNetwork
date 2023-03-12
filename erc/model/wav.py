""" Training Sequence with wav data only
"""
import torch
from transformers import Wav2Vec2ForSequenceClassification

from .model_base import ModelBase
from erc.constants import Task


class WavCls(ModelBase):
    TASK = Task.CLS
    def __init__(self, config: str, criterion: torch.nn.Module, config_kwargs: dict = None):
        super().__init__()
        if config_kwargs is None:
            config_kwargs = dict()
            config_kwargs["num_labels"] = 7 # Default emotion classification label
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(config, **config_kwargs)
        self.criterion = criterion

    def forward(self, wav: torch.Tensor, wav_mask: torch.Tensor, labels: torch.Tensor = None):
        # We retrieve logits directly in order to avoid last_hidden_state memory allocation issue
        output = self.model(input_values=wav, attention_mask=wav_mask, labels=labels)
        return {
            'loss': output['loss'],
            'emotion': labels.detach().cpu(),
            'cls_pred': output['logits'].detach().cpu(),
        }


class WavOnly(ModelBase):
    TASK = Task.ALL
    def __init__(
        self,
        config: str,
        criterions: torch.nn.Module,
        cls_coef: float = 0.5,
        config_kwargs: dict = None
    ):
        super().__init__()
        if config_kwargs is None:
            config_kwargs = dict()
            config_kwargs["num_labels"] = 9 # Default emotion classification label
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(config, **config_kwargs)
        self.criterions = criterions
        if not (0 < cls_coef < 1):
            cls_coef = 0.7
        self.cls_coef = cls_coef
        self.reg_coef = 1 - cls_coef

    def forward(self, wav: torch.Tensor, wav_mask: torch.Tensor, labels: torch.Tensor = None):
        # We retrieve logits directly in order to avoid last_hidden_state memory allocation issue
        logits = self.model(input_values=wav, attention_mask=wav_mask).logits

        cls_logits = logits[:, :-2]
        cls_loss = self.criterions["cls"](cls_logits, labels["emotion"].long())

        reg_logits = logits[:, -2:]
        reg_loss = self.criterions["reg"](reg_logits, labels["regress"].float())

        total_loss = cls_loss * self.cls_coef + reg_loss * self.reg_coef
        return {
            "loss": total_loss,
            "cls_loss": cls_loss.cpu(),
            "reg_loss": reg_loss.cpu(),
            "emotion": labels["emotion"].detach(),
            "regress": labels["regress"].detach(),
            "cls_pred": cls_logits.detach(),
            "reg_pred": reg_logits.detach(),
        }
