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
```bash
(base) conda create -n erc python=3.10
(base) conda activate erc
(erc) chmod +x ./setup.sh
(erc) ./setup.sh
```

Put data and source codes on the same hierarchy. Prevent hard copy and use soft-link: `ln -s ACTUAL_DATA_PATH data`

### Testing Functions

One may need to test a specific function on CLI. Writing an extra script for such temporal task is very nagging. Use `fire` library to boost-up productivity.

For example, if one needs to test [`preprocess.make_total_df`](erc/preprocess.py) on CLI, try the following -
```bash
(erc) python -m fire erc.preprocess make_total_df --base_path="./data/KEMDy19"
```

### Pre-trained Models

#### `.wav`

[wav2vec](https://huggingface.co/models?sort=downloads&search=wav2vec)

#### `.txt`

[Korean Pre-trained Survey](https://arxiv.org/pdf/2112.03014.pdf)
[Huggingface Hub](https://huggingface.co/models?language=ko&sort=downloads)