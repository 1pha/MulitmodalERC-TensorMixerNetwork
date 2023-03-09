import logging
import hydra
import pandas as pd

from collections import defaultdict
from tqdm import tqdm 

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.patches import Ellipse

from erc import drawing_ellipse, split_df_by_gender

import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AdamW
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer # wer metircs 
from transformers import Wav2Vec2Processor
import torch 
from torch.utils.data import DataLoader

# from accelerate import Accelerator

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Pre-training Scheme ... 
# device = torch.device('cuda:1')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with hydra.initialize(version_base=None, config_path="./config"):
    cfg = hydra.compose(config_name="config", overrides={"dataset._target_=erc.datasets.KEMDy19Dataset"})
# select 1-fold 
cfg.dataset.mode = "train"
train_dataset = hydra.utils.instantiate(cfg.dataset)

cfg.dataset.mode = "valid"
cfg.dataset.validation_fold = 0
valid_dataset = hydra.utils.instantiate(cfg.dataset)




pretrain_str = "kresnik/wav2vec2-large-xlsr-korean"
# pretrain_str = "w11wo/wav2vec2-xls-r-300m-korean"

# processor= Wav2Vec2Processor.from_pretrained(pretrain_str)
pretrained_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    # "wav2vec2-xls-r-300m-korean",
    pretrain_str,
    num_labels=7
    )

model = pretrained_model
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr = 1e-5,  eps = 1e-8)

train_loader = DataLoader(train_dataset, batch_size= 2)
# accelerator = Accelerator()

# model, optimizer, train_loader, criterion, processor = accelerator.prepare(
#      model, optimizer, train_loader, criterion,processor )





total_loss = 0
train_acc_sum = 0
train_loss = []
for step, batch in enumerate(train_loader): 
    optimizer.zero_grad()
    labels = batch['emotion']
    inputs = {"input_values":batch['wav'],
              "attention_mask":batch['wav_mask'],
    }
    # inputs = {key: inputs[key] for key in inputs}
    logits = model(**inputs).logits

    
    # outputs = torch.argmax(logits, dim=-1)
    # print(logi)

    loss = criterion(logits, labels.long())
    total_loss += loss.item()
    train_loss.append(total_loss/(step+1))
    # print(loss.item())
    loss.backward()
    # accelerator.backward(loss)
    optimizer.step()

avg_train_loss = total_loss / len(train_loader)
print(f'  Average training loss: {avg_train_loss:.2f}')


# if __name__ == '__main__':
