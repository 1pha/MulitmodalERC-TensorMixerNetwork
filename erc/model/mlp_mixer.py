from functools import partial

import torch
from transformers import Wav2Vec2ForSequenceClassification, BertForSequenceClassification
from torch import nn
from einops.layers.torch import Rearrange, Reduce

from erc.constants import Task

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
    def __init__(
        self,
        config: str,
        criterions: torch.nn.Module,
        cls_coef: float = 0.5,
        config_kwargs: dict = None
    ):
        super().__init__()
        self.wav_model = Wav2Vec2ForSequenceClassification.from_pretrained(config['wav']).wav2vec2
        self.txt_model = BertForSequenceClassification.from_pretrained(config['txt']).bert
        self.mlp_mixer = MLPMixer(image_size=self.wav_model.config.classifier_proj_size,
                                  **config['mlp_mixer'])
        self.wav_projector = nn.Linear(self.wav_model.config.hidden_size, self.wav_model.config.classifier_proj_size)
        self.txt_projector = nn.Linear(768, self.wav_model.config.classifier_proj_size)

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
         BERT_hidden_dim: 768
         BERT_proj_size: 256
           """
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
        matmul_output = torch.bmm(pooled_wav_output.unsqueeze(2), pooled_txt_output.unsqueeze(1)).unsqueeze(1)
        logits = self.mlp_mixer(matmul_output) # (B, num_labels)

        # calcuate the loss fct
        cls_logits = logits[:, :-2]
        cls_loss = self.criterions["cls"](cls_logits, labels["emotion"].long())

        reg_logits = logits[:, -2:]
        reg_loss = self.criterions["reg"](reg_logits, labels["regress"].float())

        total_loss = cls_loss * self.cls_coef + reg_loss * self.reg_coef
        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach().cpu(),
            "reg_loss": reg_loss.detach().cpu(),
            "emotion": labels["emotion"].detach(),
            "regress": labels["regress"].detach(),
            "cls_pred": cls_logits.detach(),
            "reg_pred": reg_logits.detach(),
        }
        return pred
    
