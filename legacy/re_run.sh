#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1


python train.py dataset.validation_fold=4\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=False\
                    logger.name="Scratch | soft"


# No Ppeakl
for fold in $(seq 0 4) # k-fold
do
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=False\
                    model=simple_concat\
                    logger.name="Concat | soft"
done

# One-hot 
for fold in $(seq 0 4) # k-fold
do
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=False\
                    dataset.remove_deuce=True\
                    model=simple_concat\
                    logger.name="Concat | one-hot"
done

