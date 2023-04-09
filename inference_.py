import os
import pickle

from datasets import load_from_disk
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
import hydra
from hydra import compose, initialize

import erc


def get_label(batch: dict, task: erc.constants.Task = None, device: str = "cpu"):
    labels = {
        "emotion": batch["emotion"].to(device),
        "regress": torch.stack([batch["valence"], batch["arousal"]], dim=1).to(device),
        "vote_emotion": batch.get("vote_emotion", None)
    }
    # TODO: Add Multilabel Fetch
    return labels


@torch.no_grad()
def main():
    erc.utils._seed_everything(42)
    ##################
    # valid-dataloader
    ##################
    BATCH_SIZE = 8
    valid_dataset = load_from_disk("/home/1pha/codespace/etri-erc/kemdy19-kemdy20_valid4_multilabelFalse_rdeuceTrue")
    valid_dataloadaer = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    ################
    # model load 
    ################
    with initialize(version_base=None, config_path="./config/model"):
        cfg = compose(config_name="mlp_mixer_roberta")
    cfg.config['txt'] = "klue/roberta-large"
    # cfg['_target_'] = "erc.model.inference_mlp_mixer.MLP_Mixer_Roberta"


    CKPT = '/home/1pha/codespace/etri-erc/weights_AI_HUB/epoch=29-step=92640.ckpt'
    # CKPT = '/home/1pha/codespace/etri-erc/weights_AI_HUB/RobertaL_valid4_onehot_epoch25.ckpt'
    SAVE_PATH = "./RobertaL_valid_output_hub"
    os.makedirs(SAVE_PATH, exist_ok=True)
    device = "cuda:0"
    ckpt = torch.load(CKPT, map_location=torch.device(device))
    model_ckpt = ckpt.pop("state_dict")

    model = hydra.utils.instantiate(cfg, checkpoint=model_ckpt).eval()
    model = model.to(device)

    pbar = tqdm(total=int(len(valid_dataset)/BATCH_SIZE),
                iterable=enumerate(valid_dataloadaer))

    for batch_idx, batch in pbar:
        labels = get_label(batch, device=device) # concat 
        result = model(wav=batch["wav"].to(device),
                       wav_mask=batch["wav_mask"].to(device),
                       txt=batch["txt"].to(device),
                       txt_mask=batch["txt_mask"].to(device),
                       labels=labels)
        save_dict = {
            "batch_idx": batch_idx,
            # "emotion": labels['emotion'],
        }
        save_dict.update(result)
        
        save_name = os.path.join(SAVE_PATH, f'linear_emb{batch_idx:03d}.pickle')
        with open(save_name, 'wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()