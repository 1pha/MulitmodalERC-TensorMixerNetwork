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
(erc) conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia 
(erc) pip install -r requirements.txt
# (erc) pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Put data and source codes on the same hierarchy. Prevent hard copy and use soft-link: `ln -s ACTUAL_DATA_PATH data`
### Pre-trained Models

#### `.wav`

[wav2vec](https://huggingface.co/models?sort=downloads&search=wav2vec)

#### `.txt`

[Korean Pre-trained Survey](https://arxiv.org/pdf/2112.03014.pdf)
[Huggingface Hub](https://huggingface.co/models?language=ko&sort=downloads)