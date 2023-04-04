# # validation flod 3 
# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=2 --remove_deuce=False
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=2 --remove_deuce=False

# CUDA_VISIBLE_DEVICES=1 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=2 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=True 
# # CUDA_VISIBLE_DEVICES=1 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=2 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=False # soft-label 


# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=3 --remove_deuce=False
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=3 --remove_deuce=False

# CUDA_VISIBLE_DEVICES=1 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=3 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=True 
# CUDA_VISIBLE_DEVICES=1 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=3 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=False # soft-label 

# validation flod 3 
# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=2 --remove_deuce=True
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=2 --remove_deuce=True

# CUDA_VISIBLE_DEVICES=1 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=2 dataset.multilabel=False dataset.remove_deuce=True
# CUDA_VISIBLE_DEVICES=1 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=2 dataset.multilabel=False dataset.remove_deuce=True 


# python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=3 --remove_deuce=True
# python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=3 --remove_deuce=True

# CUDA_VISIBLE_DEVICES=1 python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=3 dataset.multilabel=False dataset.remove_deuce=True


###################  Pretrained With AI HUB
#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# for fold in $(seq 0 4) # k-fold
# do
#     python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=$fold --remove_deuce=True
#     python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=$fold --remove_deuce=True
#     python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=$fold dataset.multilabel=False dataset.remove_deuce=True dataloader.batch_size=6 module.load_from_checkpoint=weights_AI_HUB/RobertaL_pretrained_aihub_weight.ckpt
# done

# True 안 했음... 
for fold in $(seq 0 4) # k-fold
do
    python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=$fold --remove_deuce=False
    python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=$fold --remove_deuce=False
    python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=$fold dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=True dataloader.batch_size=6 module.load_from_checkpoint=weights_AI_HUB/RobertaL_pretrained_aihub_weight.ckpt
done