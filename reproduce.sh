#!/bin/bash
export HYDRA_FULL_ERROR=1

### Tensor Mixer Network ###
# Hard
for fold in $(seq 0 4)
do
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=False\
                    dataset.remove_deuce=True\
                    +model.use_peakl=False
done

# Soft + Peakl
for fold in $(seq 0 4)
do
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=True
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=False
done


### Simple Concat ###
# Hard 
for fold in $(seq 0 4)
do
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=False\
                    dataset.remove_deuce=True\
                    model=simple_concat
done


# Soft + Peakl
for fold in $(seq 0 4)
do
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=True\
                    model=simple_concat
    python train.py dataset.validation_fold=${fold}\
                    dataset.multilabel=True\
                    dataset.remove_deuce=False\
                    +model.use_peakl=False\
                    model=simple_concat
done

