import torch
from transformers import (
    Wav2Vec2ForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification
)
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType

from erc.constants import Task
import erc
from .cross_attention_utils import TransformerEncoder


logger = erc.utils.get_logger(__name__)

class CrossAttentionRoberta(nn.Module):
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

        last_hdn_size = {
            "klue/roberta-base": 768, "klue/roberta-large": 1024
        }[config["txt"]]
        
        # Cross Attention
        cross_attn_config = dict(embed_dim=last_hdn_size,
                                 num_heads=8,
                                 layers=1,
                                 attn_dropout=0,
                                 relu_dropout=0,
                                 res_dropout=0,
                                 embed_dropout=0,
                                 attn_mask=None)
        self.wav2txt = TransformerEncoder(**cross_attn_config)
        self.txt2wav = TransformerEncoder(**cross_attn_config)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Last mixer
        dropout_p = 0.1
        output_dim = 512
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(2 * last_hdn_size, output_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(output_dim, 9)
        )
        
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
        wav = self.wav_model(input_values=wav, attention_mask=wav_mask).last_hidden_state # (B, S, WAV_hidden_dim)
        wav = wav.permute(1, 0, 2) # (S, B, WAV_hidden_dim)

        # Get Text Hidden States 
        txt = self.txt_model(input_ids=txt, attention_mask=txt_mask).last_hidden_state # (B, seq_len RoBERTa_hidden_dim)
        txt = txt.permute(1, 0, 2) # (S, B, RoBERTa_hidden_dim)

        # Cross Attention
        cross_w2t = self.wav2txt(wav, txt, txt).permute(1, 2, 0) # (B, proj_size, seq_len)
        cross_w2t = self.avgpool(cross_w2t) # (B, proj_size)
        cross_w2t = cross_w2t.squeeze(dim=-1)
        cross_t2w = self.txt2wav(txt, wav, wav).permute(1, 0, 2) # (B, seq_len, proj_size)
        cross_t2w = cross_t2w[:, 0, :] # (B, proj_size), cls_token only
        
        output = torch.cat([cross_w2t, cross_t2w], dim=1)
        logits = self.classifier(output)
        
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
