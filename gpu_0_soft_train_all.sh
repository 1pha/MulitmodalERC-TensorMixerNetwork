# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=0 --remove_deuce=False
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=0 --remove_deuce=False

# CUDA_VISIBLE_DEVICES=0 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=0 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=True 
# # CUDA_VISIBLE_DEVICES=0 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=0 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=False # soft-label 


# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=1 --remove_deuce=False
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=1 --remove_deuce=False

# CUDA_VISIBLE_DEVICES=0 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=1 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=True 
# CUDA_VISIBLE_DEVICES=0 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=1 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=False # soft-label 

# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=0 --remove_deuce=True
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=0 --remove_deuce=True

# CUDA_VISIBLE_DEVICES=0 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=0 dataset.multilabel=False dataset.remove_deuce=True
# CUDA_VISIBLE_DEVICES=0 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=0 dataset.multilabel=False dataset.remove_deuce=True 


# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=1 --remove_deuce=True
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=1 --remove_deuce=True

# CUDA_VISIBLE_DEVICES=0 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=1 dataset.multilabel=False dataset.remove_deuce=True


###################  Pretrained With AI HUB
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for fold in $(seq 0 4) # k-fold
do
    python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=$fold --remove_deuce=False
    python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=$fold --remove_deuce=False
    python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=$fold dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=False dataloader.batch_size=6 module.load_from_checkpoint=weights_AI_HUB/RobertaL_pretrained_aihub_weight.ckpt
done
