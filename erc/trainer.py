import torch
import hydra
import omegaconf
from torch import nn
import pytorch_lightning as pl

import erc

logger = erc.utils.get_logger()


class ERCModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_dataloader(self):
        return self.train_loader

    def valid_dataloader(self):
        return self.valid_loader
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def get_label(self, batch: dict, task: erc.constants.Task = None):
        task = task or self.model.TASK
        if task == erc.constants.Task.CLS:
            # (batch_size,) | Long
            label = batch["emotion"].long()
        elif task == erc.constants.Task.REG:
            # (batch_size, 2) | Float
            label = torch.hstack([batch["valence"], batch["arousal"]]).float()
        return label

    def forward(self, batch):
        label = self.get_label(batch)
        loss = self.model(wav=batch["wav"],
                          wav_mask=batch["wav_mask"],
                          label=label)
        return loss

    def training_step(self, batch):
        loss = self.forward(batch)
        return loss

    def training_epoch_end(self, outputs):
        breakpoint()

    def validation_step(self, batch):
        loss = self.forward(batch)
        return loss
    

def setup_trainer(config: omegaconf.DictConfig) -> pl.LightningModule:
    logger.info("Start Setting up")
    erc.utils._seed_everything(config.misc.seed)

    model = hydra.utils.instantiate(config.model)
    optim = hydra.utils.instantiate(config.optim, params=model.parameters())
    sch = hydra.utils.instantiate(config.sch, optimizer=optim) \
        if config.get("sch", None) else None

    train_dataset = hydra.utils.instantiate(config.dataset, mode="train")
    train_loader = hydra.utils.instantiate(config.dataloader, dataset=train_dataset)
    valid_dataset = hydra.utils.instantiate(config.dataset, mode="valid")
    valid_loader = hydra.utils.instantiate(config.dataloader, dataset=valid_dataset)
    
    module = hydra.utils.instantiate(config.module,
                                      model=model,
                                      optimizer=optim,
                                      scheduler=sch,
                                      train_loader=train_loader,
                                      valid_loader=valid_loader)

    return module


def train(config: omegaconf.DictConfig) -> None:
    module: pl.LightningModule = setup_trainer(config)
    trainer = hydra.utils.instantiate(config.trainer)
    trainer.fit(model=module)