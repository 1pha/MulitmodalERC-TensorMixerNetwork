echo "Note that training with thie shell script CANNOT fix configurations through CLI."
read -p "Enter validation fold [4]: " valfold
valfold=${valfold:-4}

read -p "Enter devices. For multiple devices, separate with comma [0]: " device
device=${device:-0}

read -p "Enter num_proc(num_threads) to preprocess datasets [8]: " num_proc
num_proc=${num_proc:-8}

echo "Create/Check valfold=${valfold} dataset"
python -m fire erc.datasets HF_KEMD --mode=train --num_proc=${num_proc} --multilabel=False --validation_fold=${valfold} --remove_deuce=True
python -m fire erc.datasets HF_KEMD --mode=valid --num_proc=${num_proc} --multilabel=False --validation_fold=${valfold} --remove_deuce=True
# python -m fire erc.datasets HF_KEMD --paths=aihub --mode=train --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --paths=aihub --mode=valid --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --mode=train --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --mode=valid --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --paths=aihub --mode=train --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --paths=aihub --mode=valid --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --paths=kemdy19 --mode=train --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --paths=kemdy19 --mode=valid --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --paths=kemdy20 --mode=train --validation_fold=${valfold}
# python -m fire erc.datasets HF_KEMD --paths=kemdy20 --mode=valid --validation_fold=${valfold}

echo "Start training with device=${device}"
# CUDA_VISIBLE_DEVICES=${device} python train.py model=mlp_mixer
