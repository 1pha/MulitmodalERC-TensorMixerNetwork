echo "Note that training with thie shell script CANNOT fix configurations through CLI."
read -p "Enter validation fold [4]: " valfold
valfold=${valfold:-4}

read -p "Enter devices. For multiple devices, separate with comma [0]: " device
device=${device:-0}

echo "Create/Check valfold=${valfold} dataset"
python -m fire erc.datasets HF_KEMD --mode=train --validation_fold=${valfold} --multilabel=True --num_proc=1
python -m fire erc.datasets HF_KEMD --mode=valid --validation_fold=${valfold} --multilabel=True --num_proc=1
echo "Start training with device=${device}"
# CUDA_VISIBLE_DEVICES=${device} python train.py