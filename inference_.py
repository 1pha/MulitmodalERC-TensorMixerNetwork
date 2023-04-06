import logging
import pandas as pd
from tqdm import tqdm 
import torch
import torch.nn as  nn 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_from_disk
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize

import erc
import os
import pickle

def get_label(batch: dict, task: erc.constants.Task = None):
    # labels = batch["emotion"].long()
    # labels = torch.stack([batch["valence"], batch["arousal"]], dim=1).float()
    labels = {
    "emotion": batch["emotion"],
    "regress": torch.stack([batch["valence"], batch["arousal"]], dim=1),
    "vote_emotion": batch.get("vote_emotion", None)
    }
    # TODO: Add Multilabel Fetch
    return labels





def main():

    ##################
    # valid-dataloader
    ##################
    BATCH_SIZE = 8
    valid_dataset = load_from_disk("/home/hoesungryu/etri-erc/kemdy19-kemdy20_valid4_multilabelFalse_rdeuceTrue")
    valid_dataloadaer = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    ################
    # model load 
    ################
    with initialize(version_base=None, config_path="./config/model"):
        cfg = compose(config_name="mlp_mixer_roberta")
    cfg.config['txt'] = "klue/roberta-large"
    cfg['_target_'] = "erc.model.inference_mlp_mixer.MLP_Mixer_Roberta"


    CKPT = '/home/hoesungryu/etri-erc/weights_AI_HUB/RobertaL_valid4_onehot_epoch25.ckpt'
    SAVE_PATH = "./RobertaL_valid_results"
    os.makedirs(SAVE_PATH, exist_ok=True)
    # ckpt = torch.load(CKPT, map_location="cpu")
    ckpt = torch.load(CKPT, map_location = torch.device('cuda:0'))
    model_ckpt = ckpt.pop("state_dict")

    model = hydra.utils.instantiate(cfg, checkpoint=model_ckpt).eval()

    pbar = tqdm(
    total=int(len(valid_dataset)/BATCH_SIZE), 
    iterable =enumerate(valid_dataloadaer))

    for batch_idx, batch in pbar:
        labels = get_label(batch) # concat 
        
        wav_pooled, txt_pooled = model(wav=batch["wav"],
                wav_mask=batch["wav_mask"],
                txt=batch["txt"],
                txt_mask=batch["txt_mask"],
                labels=labels)
        save_dict = {
                    "batch_idx":batch_idx,
                    "emotion": labels['emotion'],
                    "wav_pooled":wav_pooled.detach().numpy(),
                    "txt_pooled":txt_pooled.detach().numpy(),
        }
        save_name = os.path.join(SAVE_PATH, f'wav_txt_{batch_idx:03d}.pickle')
        with open(save_name, 'wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()