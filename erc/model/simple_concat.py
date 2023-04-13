import torch
from torch import nn
from transformers import Wav2Vec2ForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification

import erc
from erc.constants import Task


class SimpleConcat(nn.Module):
    TASK = Task.ALL
    def __init__(self,
                 config: str,
                 criterions: torch.nn.Module,
                 cls_coef: float = 0.7,
                 **config_kwargs):
        super().__init__()
        self.wav_model = Wav2Vec2ForSequenceClassification.from_pretrained(config['wav']).wav2vec2
        self.txt_model = BertForSequenceClassification.from_pretrained(config['txt']).bert
        
        proj_size = self.wav_model.config.classifier_proj_size
        self.wav_projector = nn.Linear(self.wav_model.config.hidden_size, proj_size)
        self.txt_projector = nn.Linear(768, proj_size)
        self.simple_concat = nn.Sequential(
                                nn.Linear(proj_size * 2, proj_size),
                                nn.ReLU(),
                                nn.Linear(proj_size, 9)
                            )

        self.criterions = criterions
        if not (0 < cls_coef < 1):
            cls_coef = 0.7
        self.cls_coef = cls_coef
        self.reg_coef = 1 - cls_coef

    def forward(
        self,
        wav: torch.Tensor,
        wav_mask: torch.Tensor,
        txt: torch.Tensor,
        txt_mask: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs) -> dict:
        """ Size
        WAV_hidden_dim: 1024
        WAV_proj_size: 256
        RoBERTa_hidden_dim: 1024 (large)
        RoBERTa_proj_size: 256
        """
        # Get Wave Hidden States
        wav_outputs = self.wav_model(input_values=wav, attention_mask=wav_mask) # (B, S, WAV_hidden_dim)
        hidden_states = self.wav_projector(wav_outputs[0]) # (B, S, WAV_proj_size) 
        
        # Pool WAV hidden states. (B, proj_size)
        if wav_mask is None:
            pooled_wav_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.wav_model._get_feature_vector_attention_mask(hidden_states.shape[1], wav_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_wav_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # Get Text Hidden States 
        txt_last_hidden_state = self.txt_model(input_ids=txt, attention_mask=txt_mask)[0] # (B, RoBERTa_hidden_dim)
        txt_outputs = txt_last_hidden_state[:, 0, :]
        pooled_txt_output = self.txt_projector(txt_outputs) # (B, RoBERTa_proj_size)

        # Tensor Fusion
        # (B, 1 , WAV_proj_size, BERT_proj_size)
        concat_output = torch.cat((pooled_wav_output, pooled_txt_output), dim=1)
        logits = self.simple_concat(concat_output) # (B, num_labels)

        # Calculate Loss
        cls_logits = logits[:, :-2]
        cls_labels = labels["emotion"]
        if cls_labels.ndim == 1: # Single label case
            cls_loss = self.criterions["cls"](cls_logits, cls_labels.long())
        elif cls_labels.ndim == 2: # Multi label case
            if self.use_peakl:
                cls_labels = erc.utils.apply_peakl(logits=cls_labels)
            cls_loss = self.criterions["cls"](cls_logits, cls_labels.float())
        
        reg_logits = logits[:, -2:]
        reg_loss = self.criterions["reg"](reg_logits, labels["regress"].float())

        total_loss = cls_loss * self.cls_coef + reg_loss * self.reg_coef
        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach().cpu(),
            "reg_loss": reg_loss.detach().cpu(),
            "emotion": cls_labels.detach(),
            "regress": labels["regress"].detach().float(),
            "cls_pred": cls_logits.detach(),
            "reg_pred": reg_logits.detach().float(),
        }


class SimpleConcatRoberta(nn.Module):
    TASK = Task.ALL
    def __init__(self,
                 config: str,
                 criterions: torch.nn.Module,
                 cls_coef: float = 0.7,
                 **config_kwargs):
        super().__init__()
        self.wav_model = Wav2Vec2ForSequenceClassification.from_pretrained(config['wav']).wav2vec2
        self.txt_model = RobertaForSequenceClassification.from_pretrained(config['txt']).roberta

        proj_size = self.wav_model.config.classifier_proj_size
        self.wav_projector = nn.Linear(self.wav_model.config.hidden_size, proj_size)
        last_hdn_size = {
            "klue/roberta-base": 768, "klue/roberta-large": 1024
        }[config["txt"]]
        self.txt_projector = nn.Linear(last_hdn_size, proj_size)        
        self.simple_concat = nn.Sequential(
                                nn.Linear(proj_size * 2, proj_size),
                                nn.ReLU(),
                                nn.Linear(proj_size, 9)
                            )

        self.use_peakl = config_kwargs.get("use_peakl", False)

        self.criterions = criterions
        if not (0 < cls_coef < 1):
            cls_coef = 0.7
        self.cls_coef = cls_coef
        self.reg_coef = 1 - cls_coef

    def forward(
        self,
        wav: torch.Tensor,
        wav_mask: torch.Tensor,
        txt: torch.Tensor,
        txt_mask: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs) -> dict:
        """ Size
        WAV_hidden_dim: 1024
        WAV_proj_size: 256
        RoBERTa_hidden_dim: 1024 (large)
        RoBERTa_proj_size: 256
        """
        # Get Wave Hidden States
        wav_outputs = self.wav_model(input_values=wav, attention_mask=wav_mask) # (B, S, WAV_hidden_dim)
        hidden_states = self.wav_projector(wav_outputs[0]) # (B, S, WAV_proj_size) 
        
        # Pool WAV hidden states. (B, proj_size)
        if wav_mask is None:
            pooled_wav_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self.wav_model._get_feature_vector_attention_mask(hidden_states.shape[1], wav_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_wav_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # Get Text Hidden States 
        txt_last_hidden_state = self.txt_model(input_ids=txt, attention_mask=txt_mask)[0] # (B, RoBERTa_hidden_dim)
        txt_outputs = txt_last_hidden_state[:, 0, :]
        pooled_txt_output = self.txt_projector(txt_outputs) # (B, RoBERTa_proj_size)

        # Tensor Fusion
        # (B, 1 , WAV_proj_size, BERT_proj_size)
        concat_output = torch.cat((pooled_wav_output, pooled_txt_output), dim=1)
        logits = self.simple_concat(concat_output) # (B, num_labels)

        # Calculate Loss
        cls_logits = logits[:, :-2]
        cls_labels = labels["emotion"]
        if cls_labels.ndim == 1: # Single label case
            cls_loss = self.criterions["cls"](cls_logits, cls_labels.long())
        elif cls_labels.ndim == 2: # Multi label case
            if self.use_peakl:
                cls_labels = erc.utils.apply_peakl(logits=cls_labels)
            cls_loss = self.criterions["cls"](cls_logits, cls_labels.float())
        
        reg_logits = logits[:, -2:]
        reg_loss = self.criterions["reg"](reg_logits, labels["regress"].float())

        total_loss = cls_loss * self.cls_coef + reg_loss * self.reg_coef
        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach().cpu(),
            "reg_loss": reg_loss.detach().cpu(),
            "emotion": cls_labels.detach(),
            "regress": labels["regress"].detach().float(),
            "cls_pred": cls_logits.detach(),
            "reg_pred": reg_logits.detach().float(),
        }
