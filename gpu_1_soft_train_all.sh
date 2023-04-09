###################  Pretrained With AI HUB
#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

# One-hot | HubRobertal
for fold in $(seq 0 4) # k-fold
do
    python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=$fold --remove_deuce=True
    python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=$fold --remove_deuce=True
    python train.py dataset.validation_fold=$fold\
                    dataset.multilabel=False\
                    dataset.remove_deuce=True\
                    dataloader.batch_size=6\
                    module.load_from_checkpoint=weights_AI_HUB/RobertaL_pretrained_aihub_weight.ckpt\
                    logger.name="Hub | vfold=$fold | one-hot"
done

# Ppeakl + HubRobertal
for fold in $(seq 0 4) # k-fold
do
    python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=$fold --remove_deuce=False
    python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=$fold --remove_deuce=False
    python train.py dataset.validation_fold=$fold\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=True\
                    dataloader.batch_size=6\
                    module.load_from_checkpoint=weights_AI_HUB/RobertaL_pretrained_aihub_weight.ckpt\
                    logger.name="Hub | vfold=$fold | peakl"
    python train.py dataset.validation_fold=$fold\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=False\
                    dataloader.batch_size=6\
                    module.load_from_checkpoint=weights_AI_HUB/RobertaL_pretrained_aihub_weight.ckpt\
                    logger.name="Hub | vfold=$fold | soft"
done