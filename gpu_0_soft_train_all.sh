###################  Scratch Training
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# Hard | Scratch
for fold in $(seq 0 4) # k-fold
do
    # python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=$fold --remove_deuce=True
    # python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=$fold --remove_deuce=True
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=False\
                    dataset.remove_deuce=True\
                    +model.use_peakl=False\
                    logger.name="Scratch | one-hot"
done

# Soft + Peakl | Scratch
for fold in $(seq 0 4) # k-fold
do
    # python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=$fold --remove_deuce=False
    # python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=$fold --remove_deuce=False
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=True\
                    logger.name="Scratch | peakl"
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=False\
                    logger.name="Scratch | soft"
done
