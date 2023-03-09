import hydra
import omegaconf
import torch.nn as nn 
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

from erc.utils import get_logger
logger = get_logger()


def loop(batch, model, criterion, optimizer, accelerator):
    
    labels = batch['emotion']
    inputs = {"input_values": batch['wav'],
            "attention_mask": batch['wav_mask']}
    logits = model(**inputs).logits
    loss = criterion(logits, labels.long())

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()


@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    # TODO: Much more work required
    train_dataset = hydra.utils.instantiate(cfg.dataset, mode="train")
    valid_dataset = hydra.utils.instantiate(cfg.dataset, mode="valid")

    pretrain_str = "kresnik/wav2vec2-large-xlsr-korean"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(pretrain_str,
                                                              num_labels=7)

    criterion = hydra.utils.instantiate(cfg.criterion)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    train_loader = hydra.utils.instantiate(cfg.dataloader, dataset=train_dataset)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(device_placement=True, kwargs_handlers=[ddp_kwargs])
    model, optimizer, train_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, criterion)
    
    accelerator.free_memory()
    for batch_idx, batch in enumerate(train_loader): 
        loop(batch=batch, model=model, criterion=criterion, optimizer=optimizer, accelerator=accelerator)
        accelerator.wait_for_everyone()
    return

if __name__=="__main__":
    main()
