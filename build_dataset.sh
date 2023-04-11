###################  Scratch Training
#!/bin/bash

# Hard | Scratch
for fold in $(seq 0 4) # k-fold
do
    python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=$fold --remove_deuce=True
    python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=$fold --remove_deuce=True
done

# Soft + Peakl | Scratch
for fold in $(seq 0 4) # k-fold
do
    python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=$fold --remove_deuce=False
    python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=$fold --remove_deuce=False

done
