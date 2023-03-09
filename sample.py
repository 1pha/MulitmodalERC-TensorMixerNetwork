import hydra
import omegaconf
import torch.nn as nn 
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AdamW
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

from erc.utils import get_logger


logger = get_logger()

@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    # TODO: Much more work required
    train_dataset = hydra.utils.instantiate(cfg.dataset, mode="train")
    valid_dataset = hydra.utils.instantiate(cfg.dataset, mode="valid")

    pretrain_str = "kresnik/wav2vec2-large-xlsr-korean"
    processor = Wav2Vec2Processor.from_pretrained(pretrain_str)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(pretrain_str,
                                                              num_labels=7)

    criterion = hydra.utils.instantiate(cfg.criterion)
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
    
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for _, batch in enumerate(train_loader): 
        labels = batch['emotion']
        inputs = {"input_values": batch['wav'],
                "attention_mask": batch['wav_mask']}
        logits = model(**inputs).logits
        loss = criterion(logits, labels.long())

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    return 0

if __name__=="__main__":
    main()
