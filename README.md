# Multi-modal Emotion Recognition in Conversation (ERC) @ ETRI

## Emotion Distribution across
![image](./assets/embed.png)
[Competition Links](https://aifactory.space/competition/detail/2234)

## 1. Data

### About Data
Data contains 3 modalities
- `.wav`: Audio file
- `.txt`: Script of an audio
- `.csv`: Electrocardiogram & Electrodermal activity data
```
./
├── README.md
├── assets
├── config
├── data
├── erc
├── train.py
├── requirements.txt
├── setup.sh
└── train.sh
```

```
./data
├── KEMDy19
│   ├── ECG
│   ├── EDA
│   ├── TEMP
│   ├── annotation
│   └── wav
├── KEMDy20_v1_1
│   ├── EDA
│   ├── IBI
│   ├── TEMP
│   ├── annotation
│   └── wav
```


## 2. Code
### Basic Setups
```zsh
(base) conda create -n erc python=3.10
(base) conda activate erc
(erc) chmod +x ./setup.sh
(erc) ./setup.sh
```

- Put data and source codes on the same hierarchy. Prevent hard copy and use soft-link: `ln -s ACTUAL_DATA_PATH data`
- It is good to

### Training
Since creating a new dataset requires computational burden & `num_proc > 1` for multiprocessing datasets gets deadlocked, one first needs to **explicitly create a dataset with following commands**
```zsh
python -m fire erc.datasets HF_KEMD --mode=train --validation_fold=${valfold}
python -m fire erc.datasets HF_KEMD --mode=valid --validation_fold=${valfold}
```
With default configuration of [./config/train.yaml]
```zsh
python train.py
```

Note that above processes in a single shell script are written in [train.sh](./train.sh). However, modifying configurations through CLI is not possible but default configurations saved in [config](./config) directory is only available.
```zsh
(erc) chmod +x train.sh
(erc) ./train.sh
```

**Fast Dev**:
Cases where cpu is not available, debugging required. Below command reduces number of dataset being forwarded.
```zsh
python train.py dataset.num_data=4 dataloader.batch_size=4 trainer.accelerator=cpu
```
or use `lightning`s' `fast_dev_run` flag. (_Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test to find any bugs (ie: a sort of unit test). Default: False._)
```zsh
python train.py +trainer.fast_dev_run=True
```

### Testing Functions

One may need to test a specific function on CLI. Writing an extra script for such temporal task is very nagging. Use **`fire` library** to boost-up productivity.

#### Merge `.csv`
For example, if one needs to test [`preprocess.make_total_df`](erc/preprocess.py) on CLI, try the following -
```zsh
(erc) python -m fire erc.preprocess make_total_df --base_path="./data/KEMDy19"
```
#### Create huggingface `datasets`
```zsh
(erc) python -m fire erc.preprocess run_generate_datasets --dataset_name="kemdy19"
```

#### Create cache file for datasetes
```zsh
(erc) python -m fire erc.datasets HF_KEMD
```

## 3. Reference
### Pre-trained Models

#### `.wav`

- [wav2vec](https://huggingface.co/models?sort=downloads&search=wav2vec)

#### `.txt`

- [Korean Pre-trained Survey](https://arxiv.org/pdf/2112.03014.pdf)
- [Huggingface Hub](https://huggingface.co/models?language=ko&sort=downloads)