# Model Descriptions

## [Wav2Vec2](https://arxiv.org/abs/2006.11477)
- [Pre-trained Weights](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean)

## [RoBERTa](https://arxiv.org/abs/1907.11692)
- [Pre-trained Weights](https://huggingface.co/klue/roberta-large)

## [MLP Mixer](https://arxiv.org/abs/2105.01601)

Implemented with the help of [lucidrains/mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch)
Total number of layers: 17
- [0] Rearrange
- [1] Linear 256 → 512
- [2 - 13] MixerBlock
- [14] LayerNorm
- [15] Reduce: b n c → b c, mean
- [16] Linear: 512 → 9 (num_logits)

Take the last output [16] and last block [13]