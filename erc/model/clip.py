""" [For Future Works]
"""

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2ForSequenceClassification, BertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

from erc.constants import Task
import erc


logger = erc.utils.get_logger(__name__)


class CLIP(nn.Module):
    TASK = Task.ALL
    def __init__(
        self,
        config: str,
        criterion: torch.nn.Module,
    ):
        super().__init__()
        wav_model = Wav2Vec2ForSequenceClassification.from_pretrained(config['wav'])
        txt_model = BertForSequenceClassification.from_pretrained(config['txt'])
        if "lora" in config:
            logger.info("Train with Lora")
            pcfg_wav = LoraConfig(task_type=TaskType.SEQ_CLS, **config["lora"]["wav"])
            self.wav_model = get_peft_model(wav_model, pcfg_wav).wav2vec2
            pcfg_txt = LoraConfig(task_type=TaskType.SEQ_CLS, **config["lora"]["txt"])
            self.txt_model = get_peft_model(txt_model, pcfg_txt).bert
        else:
            self.wav_model = wav_model.wav2vec2
            self.txt_model = txt_model.bert

        self.wav_projector = nn.Linear(self.wav_model.config.hidden_size, self.wav_model.config.classifier_proj_size)
        self.txt_projector = nn.Linear(768, self.wav_model.config.classifier_proj_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.criterion = criterion

    def forward(
        self,
        wav: torch.Tensor,
        wav_mask: torch.Tensor,
        txt: torch.Tensor,
        txt_mask: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs
    ) -> dict:
        """ Size
         WAV_hidden_dim: 1024
         WAV_proj_size: 256
         BERT_hidden_dim: 768
         BERT_proj_size: 256
        """
        batch_size = wav.shape[0]
        # WAV 
        wav_outputs = self.wav_model(input_values=wav, attention_mask=wav_mask) # (B, S, WAV_hidden_dim)
        hidden_states = self.wav_projector(wav_outputs[0]) # (B, S, WAV_proj_size) 
        # Pool hidden states. (B, proj_size)
        if wav_mask is None:
            pooled_wav_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.wav_model._get_feature_vector_attention_mask(hidden_states.shape[1], wav_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_wav_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # TXT 
        txt_outputs = self.txt_model(input_ids=txt, attention_mask=txt_mask)[1] # (B, BERT_hidden_dim)
        pooled_txt_output = self.txt_projector(txt_outputs) # (B, BERT_proj_size)

        # (B, 1 , WAV_proj_size, BERT_proj_size)
        wav_embed = pooled_wav_output / pooled_wav_output.norm(dim=1, keepdim=True) # (B, WAV_proj_size)
        txt_embed = pooled_txt_output / pooled_txt_output.norm(dim=1, keepdim=True) # (B, BERT_proj_size)

        logit_scale = self.logit_scale.exp() # (,)
        logits = logit_scale * wav_embed @ txt_embed.t() 

        target = torch.arange(batch_size, device=wav.device)
        wav_loss = self.criterion(logits, target)
        txt_loss = self.criterion(logits.t(), target)

        loss = (wav_loss + txt_loss) / 2
        return {
            "loss": loss
        }
