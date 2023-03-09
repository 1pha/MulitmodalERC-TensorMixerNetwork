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

    model, optimizer, train_loader, criterion, processor = accelerator.prepare(
        model, optimizer, train_loader, criterion,processor)

    total_loss = 0
    train_loss = []
    for batch_idx, batch in enumerate(train_loader): 
        labels = batch['emotion']
        inputs = {"input_values": batch['wav'],
                "attention_mask": batch['wav_mask']}
        logits = model(**inputs).logits

        loss = criterion(logits, labels.long())
        total_loss += loss.item()
        train_loss.append(total_loss / (batch_idx + 1))

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f'  Average training loss: {avg_train_loss:.2f}')
    return avg_train_loss

if __name__=="__main__":
    main()
