# Multi-modal Emotion Recognition in Conversation (ERC) @ ETRI

[Competition Links](https://aifactory.space/competition/detail/2234)

## Data

### About Data
Data contains 3 modalities
- `.wav`: Audio file
- `.txt`: Script of an audio
- `.csv`: Electrocardiogram & Electrodermal activity data

## Code
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
With default configuration of [./config/train.yaml]
```zsh
python train.py
```
Predicting both emotion and regression
```zsh
python train.py model=combined
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

### Pre-trained Models

#### `.wav`

- [wav2vec](https://huggingface.co/models?sort=downloads&search=wav2vec)

#### `.txt`

- [Korean Pre-trained Survey](https://arxiv.org/pdf/2112.03014.pdf)
- [Huggingface Hub](https://huggingface.co/models?language=ko&sort=downloads)