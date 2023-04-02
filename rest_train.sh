# k-fold 5 안 햇었음 ... 하면서 weight 저장하자... 
python -m fire erc.datasets HF_KEMD --mode=train --multilabel=False --validation_fold=4 --remove_deuce=True
python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=False --validation_fold=4 --remove_deuce=True
python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=4 dataset.multilabel=False dataset.remove_deuce=True

python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=4 --remove_deuce=False
python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=4 --remove_deuce=False
python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=4 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=True
python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=4 dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=False

# True 안 했음... 
for fold in $(seq 0 4) # k-fold
do
    python -m fire erc.datasets HF_KEMD --mode=train --multilabel=True --validation_fold=$fold --remove_deuce=False
    python -m fire erc.datasets HF_KEMD --mode=valid --multilabel=True --validation_fold=$fold --remove_deuce=False
    python train.py model=mlp_mixer_roberta model.config.txt=klue/roberta-large dataset.validation_fold=$fold dataset.multilabel=True dataset.remove_deuce=False +model.use_peakl=True dataloader.batch_size=6 module.load_from_checkpoint=weights_AI_HUB/RobertaL_pretrained_aihub_weight.ckpt
done