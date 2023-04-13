echo "Note that training with this shell script CANNOT fix configurations through CLI."
read -p "Enter validation fold [4]: " valfold
valfold=${valfold:-4}

read -p "Enter devices. For multiple devices, separate with comma [0]: " device
device=${device:-0}

read -p "Enter num_proc(num_threads) to preprocess datasets [8]: " num_proc
num_proc=${num_proc:-1}

echo "Start training with device=${device}"
CUDA_VISIBLE_DEVICES=${device} python train.py dataset.validation_fold=${valfold}\
                                               dataset.num_proc=${num_proc}\
                                               dataset.multilabel=True\
                                               dataset.remove_deuce=False\
                                               +model.use_peakl=True\
                                               model=mlp_mixer_roberta