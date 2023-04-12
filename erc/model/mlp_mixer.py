from functools import partial

import torch
from transformers import (
    Wav2Vec2ForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification
)
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from peft import get_peft_model, LoraConfig, TaskType

from erc.constants import Task
import erc


logger = erc.utils.get_logger(__name__)


pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class MLP_Mixer(nn.Module):
    TASK = Task.ALL
    def __init__(self,
                 config: str,
                 criterions: torch.nn.Module,
                 cls_coef: float = 0.5,
                 **config_kwargs):
        super().__init__()
        self.wav_model = Wav2Vec2ForSequenceClassification.from_pretrained(config['wav'])
        self.txt_model = BertForSequenceClassification.from_pretrained(config['txt'])

        proj_size = self.wav_model.config.classifier_proj_size
        self.mlp_mixer = MLPMixer(image_size=proj_size,
                                  **config['mlp_mixer'])
        self.wav_projector = nn.Linear(self.wav_model.config.hidden_size, proj_size)
        self.txt_projector = nn.Linear(768, proj_size)
        
        self.use_peakl = config_kwargs.get("use_peakl", False)
        
        # Loading Checkpoints
        if config_kwargs.get("checkpoint"):
            logger.info("Load from checkpoint")
            def parser(k):
                _k = k.split(".")[1:]
                if _k[0] == "wav_model" and _k[1] != "classifier":
                    _k.insert(1, "wav2vec2")
                elif _k[0] == "txt_model" and _k[1] != "classifier":
                    _k.insert(1, "bert")
                return ".".join(_k)
            ckpt = {parser(k): v for k, v in config_kwargs["checkpoint"].items()}
            self.load_state_dict(ckpt, strict=False)
        
        # Setting up LORA
        if "lora" in config:
            logger.info("Train with Lora")
            pcfg_wav = LoraConfig(task_type=TaskType.SEQ_CLS, **config["lora"]["wav"])
            self.wav_model = get_peft_model(self.wav_model, pcfg_wav)
            pcfg_txt = LoraConfig(task_type=TaskType.SEQ_CLS, **config["lora"]["txt"])
            self.txt_model = get_peft_model(self.txt_model, pcfg_txt)
        
        # Retrieving Encoders
        self.wav_model = self.wav_model.wav2vec2
        self.txt_model = self.txt_model.bert

        self.criterions = criterions
        if not (0 < cls_coef < 1):
            cls_coef = 0.7
        self.cls_coef = cls_coef
        self.reg_coef = 1 - cls_coef

    def forward(self,
                wav: torch.Tensor,
                wav_mask: torch.Tensor,
                txt: torch.Tensor,
                txt_mask: torch.Tensor,
                labels: torch.Tensor = None,
                **kwargs) -> dict:
        """ Size
         WAV_hidden_dim: 1024
         WAV_proj_size: 256
         BERT_hidden_dim: 768
         BERT_proj_size: 256
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
        txt_last_hidden_state = self.txt_model(input_ids=txt, attention_mask=txt_mask)[0] # (B, BERT_hidden_dim)
        txt_outputs = txt_last_hidden_state[:, 0, :]
        pooled_txt_output = self.txt_projector(txt_outputs) # (B, BERT_proj_size)

        # Tensor Fusion
        # (B, 1 , WAV_proj_size, BERT_proj_size)
        matmul_output = torch.bmm(pooled_wav_output.unsqueeze(2),
                                  pooled_txt_output.unsqueeze(1)).unsqueeze(1)
        logits = self.mlp_mixer(matmul_output) # (B, num_labels)

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
            "regress": labels["regress"].detach(),
            "cls_pred": cls_logits.detach(),
            "reg_pred": reg_logits.detach(),
        }


class MLP_Mixer_Roberta(nn.Module):
    TASK = Task.ALL
    def __init__(
        self,
        config: str,
        criterions: torch.nn.Module,
        cls_coef: float = 0.7,
        **config_kwargs
    ):
        super().__init__()
        self.wav_model = Wav2Vec2ForSequenceClassification.from_pretrained(config['wav'])
        self.txt_model = RobertaForSequenceClassification.from_pretrained(config['txt'])

        proj_size = self.wav_model.config.classifier_proj_size
        self.mlp_mixer = MLPMixer(image_size=proj_size,
                                  **config['mlp_mixer'])
        self.wav_projector = nn.Linear(self.wav_model.config.hidden_size, proj_size)
        last_hdn_size = {
            "klue/roberta-base": 768, "klue/roberta-large": 1024
        }[config["txt"]]
        self.txt_projector = nn.Linear(last_hdn_size, proj_size)
        
        self.use_peakl = config_kwargs.get("use_peakl", False)
        
        # Loading Checkpoints
        if config_kwargs.get("checkpoint"):
            for name, param in self.state_dict().items():
                if param.requires_grad:
                    print(name)
            logger.info("Load from checkpoint")
            def parser(k):
                _k = k.split(".")[1:]
                if _k[0] == "wav_model" and _k[1] != "classifier":
                    _k.insert(1, "wav2vec2")
                elif _k[0] == "txt_model" and _k[1] != "classifier":
                    _k.insert(1, "roberta")
                return ".".join(_k)
            ckpt = {parser(k): v for k, v in config_kwargs["checkpoint"].items()}
            self.load_state_dict(ckpt, strict=False)
            
        # Setting up LORA
        if "lora" in config:
            logger.info("Train with Lora")
            pcfg_wav = LoraConfig(task_type=TaskType.SEQ_CLS, **config["lora"]["wav"])
            self.wav_model = get_peft_model(self.wav_model, pcfg_wav)
            pcfg_txt = LoraConfig(task_type=TaskType.SEQ_CLS, **config["lora"]["txt"])
            self.txt_model = get_peft_model(self.txt_model, pcfg_txt)
        
        # Retrieving Encoders
        self.wav_model = self.wav_model.wav2vec2
        self.txt_model = self.txt_model.roberta

        self.criterions = criterions
        if not (0 < cls_coef < 1):
            cls_coef = 0.7
        self.cls_coef = cls_coef
        self.reg_coef = 1 - cls_coef

    def forward(self,
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
        matmul_output = torch.bmm(pooled_wav_output.unsqueeze(2), pooled_txt_output.unsqueeze(1)).unsqueeze(1)
        logits = self.mlp_mixer(matmul_output) # (B, num_labels)

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
