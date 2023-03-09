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
            'labels': output['labels'],
            'logits': output['logits'].detach().cpu(),
        }


class WavReg(ModelBase):
    TASK = Task.REG
    def __init__(self, config: str, criterion: torch.nn.Module, config_kwargs: dict = None):
        super().__init__()
        if config_kwargs is None:
            config_kwargs = dict()
            config_kwargs["num_labels"] = 2 # Default emotion classification label
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(config, **config_kwargs)
        self.criterion = criterion

    def forward(self, wav: torch.Tensor, wav_mask: torch.Tensor, label: torch.Tensor = None):
        result = {}
        # We retrieve logits directly in order to avoid last_hidden_state memory allocation issue
        logits = self.model(input_values=wav, attention_mask=wav_mask).logits
        if label is not None:
            loss = self.criterion(logits, label.float())
            result["loss"] = loss
        return result
