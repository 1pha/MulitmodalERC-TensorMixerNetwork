import importlib
from typing import List, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, AUROC, ConcordanceCorrCoef, F1Score
from torchmetrics.functional.classification import accuracy, auroc, multiclass_f1_score
from torchmetrics.functional import concordance_corrcoef
import wandb

import erc


logger = erc.utils.get_logger(name=__name__)


class ERCModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 optimizer: omegaconf.DictConfig,
                 scheduler: omegaconf.DictConfig = None,
                 load_from_checkpoint: str = None,
                 separate_lr: dict = None):
        super().__init__()
        self.model = model

        # Dataloaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Optimizations
        if separate_lr is not None:
            _opt_groups = []
            for _submodel, _lr in separate_lr.items():
                submodel = getattr(self.model, _submodel, None)
                if submodel is None:
                    logger.warn("separate_lr was given but submodel was not found: %s", _submodel)
                    self.opt_config = self._configure_optimizer(optimizer=optimizer,
                                                                scheduler=scheduler)
                    break
                _opt_groups.append(
                    {"params": submodel.parameters(), "lr": _lr}
                )
            opt = dict(optimizer)
            _o = opt.pop("_target_").split(".")
            _oc = importlib.import_module(".".join(_o[:-1]))
            _oc = getattr(_oc, _o[-1])
            _opt = _oc(params=_opt_groups, **opt)
            _sch = hydra.utils.instantiate(scheduler, scheduler={"optimizer": _opt})
            self.opt_config = {"optimizer": _opt, "lr_scheduler": dict(**_sch)}
        else:
            self.opt_config = self._configure_optimizer(optimizer=optimizer, scheduler=scheduler)

        # Metrics Configuration
        self.acc = Accuracy(task="multiclass", num_classes=7)
        self.auroc = AUROC(task="multiclass", num_classes=7)
        self.f1 = F1Score(task="multiclass", num_classes=7, average="macro")
        self.ccc_val = ConcordanceCorrCoef(num_outputs=1)
        self.ccc_aro = ConcordanceCorrCoef(num_outputs=1)

        self.label_keys = list(erc.constants.emotion2idx.keys())[:-1]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def _configure_optimizer(self, optimizer: omegaconf.DictConfig, scheduler: omegaconf.DictConfig):
        opt = hydra.utils.instantiate(optimizer, params=self.model.parameters())
        sch: dict = hydra.utils.instantiate(scheduler, scheduler={"optimizer": opt})\
                            if scheduler is not None else None
        opt_config = {
            "optimizer": opt, "lr_scheduler": dict(**sch)
        } if sch is not None else opt
        return opt_config
    
    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        return self.opt_config

    def get_label(self, batch: dict, task: erc.constants.Task = None):
        task = task or self.model.TASK
        if task == erc.constants.Task.CLS:
            # (batch_size,) | Long
            labels = batch["emotion"].long()
        elif task == erc.constants.Task.REG:
            # (batch_size, 2) | Float
            labels = torch.stack([batch["valence"], batch["arousal"]], dim=1).float()
        elif task == erc.constants.Task.ALL:
            labels = {
                "emotion": batch["emotion"],
                "regress": torch.stack([batch["valence"], batch["arousal"]], dim=1),
                "vote_emotion": batch.get("vote_emotion", None)
            }
        # TODO: Add Multilabel Fetch
        return labels

    def forward(self, batch):
        try:
            labels = self.get_label(batch)
            result: dict = self.model(wav=batch["wav"],
                                      wav_mask=batch["wav_mask"],
                                      txt=batch["txt"],
                                      txt_mask=batch["txt_mask"],
                                      labels=labels)
            return result
        except RuntimeError:
            # For CUDA Device-side asserted error
            print(f"Label given {labels}")
            logger.warn("Label given %s", labels)
            raise RuntimeError

    def _sort_outputs(self, outputs: List[Dict]):
        try:
            result = dict()
            keys: list = outputs[0].keys()
            for key in keys:
                data = outputs[0][key]
                if data.ndim == 0:
                    # Scalar value result
                    result[key] = torch.stack([o[key] for o in outputs if key in o])
                elif data.ndim in [1, 2]:
                    # Batched 
                    result[key] = torch.concat([o[key] for o in outputs if key in o])
        except AttributeError:
            logger.warn("Error provoking data %s", outputs)
            breakpoint()
        return result

    def remove_deuce(self, outputs: dict) -> dict:
        """ Find deuced emotions and remove from batch """
        result = outputs
        emotion = outputs["emotion"]
        if emotion.ndim == 2:
            # For multi-dimensional emotion cases
            _, num_class = emotion.shape
            v, _ = emotion.max(dim=1)
            v = v.unsqueeze(dim=1).repeat(1, num_class)
            um = (emotion == v).sum(dim=1) == 1 # (bsz, ), unique mask
            if (um.sum() == 0).item():
                # If every batches had deuce data
                # Return scalar metrics only (removing cls/reg pred and logits)
                result = {k: _v for k, _v in outputs.items() if _v.ndim == 0}
            else:
                result.update(
                    {k: _v[um] for k, _v in outputs.items() if _v.ndim > 0}
                )
                result["emotion"] = result["emotion"].argmax(dim=1)
            return result
        else:
            return result

    def log_result(
        self, 
        outputs: List[Dict] | dict, 
        mode: erc.constants.RunMode | str = "train",
        unit: str = "epoch"
    ):
        result: dict = self._sort_outputs(outputs=outputs) if isinstance(outputs, list) else outputs
        on_step: bool = unit == "step"
        on_epoch: bool = unit == "epoch"
        
        if unit == "step":
            # No need to on epochs
            result = self.remove_deuce(outputs=result)
        # Log Losses
        for loss_key in ["loss", "cls_loss", "reg_loss"]:
            if loss_key in result:
                self.log(f"{unit}/{mode}_{loss_key}",
                         torch.mean(result.get(loss_key, 0)),
                         prog_bar=True, on_step=on_step, on_epoch=on_epoch)

        # Log Classification Metrics: Accuracy & AUROC
        if "cls_pred" in result and "emotion" in result:
            _acc = accuracy(preds=result["cls_pred"],
                            target=result["emotion"],
                            task="multiclass",
                            num_classes=7)
            _auroc = auroc(preds=result["cls_pred"],
                           target=result["emotion"],
                           task="multiclass",
                           num_classes=7)
            _macro_f1 = multiclass_f1_score(preds=result["cls_pred"],
                                      target=result["emotion"],
                                      average="macro",
                                      num_classes=7)
            _micro_f1 = multiclass_f1_score(preds=result["cls_pred"],
                                      target=result["emotion"],
                                      average="micro",
                                      num_classes=7)
            self.log(f'{unit}/{mode}_acc', _acc, on_step=on_step, on_epoch=on_epoch)
            self.log(f'{unit}/{mode}_auroc', _auroc, on_step=on_step, on_epoch=on_epoch)
            self.log(f'{unit}/{mode}_macrof1', _macro_f1, on_step=on_step, on_epoch=on_epoch)
            self.log(f'{unit}/{mode}_microf1', _micro_f1, on_step=on_step, on_epoch=on_epoch)

        # Log Regression Metrics: CCC
        if "reg_pred" in result and "regress" in result:
            ccc_val = concordance_corrcoef(preds=result["reg_pred"][:, 0], target=result["regress"][:, 0])
            ccc_aro = concordance_corrcoef(preds=result["reg_pred"][:, 1], target=result["regress"][:, 1])
            self.log(f"{unit}/{mode}_ccc(val)", ccc_val, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{unit}/{mode}_ccc(aro)", ccc_aro, on_step=on_step, on_epoch=on_epoch)
        return result

    def log_confusion_matrix(self, result: dict):
        preds = result["cls_pred"].cpu().detach() if "cls_pred" in result else None
        labels = result["emotion"].cpu().numpy() if "emotion" in result else None
        if preds is not None and labels is not None:
            preds = preds.argmax(dim=1).numpy()
            cf = wandb.plot.confusion_matrix(y_true=labels,
                                            preds=preds,
                                            class_names=self.label_keys)
            self.logger.experiment.log({"confusion_matrix": cf})

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        result = self.forward(batch)
        result = self.log_result(outputs=result, mode="train", unit="step")
        return result

    def training_epoch_end(self, outputs: List[Dict]):
        result = self.log_result(outputs=outputs, mode="train", unit="epoch")

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        result = self.forward(batch)
        result = self.log_result(outputs=result, mode="valid", unit="step")
        return result
    
    def validation_epoch_end(self, outputs: List[Dict], *args, **kwargs):
        result = self.log_result(outputs=outputs, mode="valid", unit="epoch")
        self.log_confusion_matrix(result)


def setup_trainer(config: omegaconf.DictConfig) -> pl.LightningModule:
    logger.info("Start Setting up")
    erc.utils._seed_everything(config.misc.seed)

    ckpt = config.module.load_from_checkpoint
    if ckpt:
        ckpt = torch.load(ckpt)
        model_ckpt = ckpt.pop("state_dict")
    else:
        model_ckpt = None

    logger.info("Start intantiating Models & Optimizers")
    model = hydra.utils.instantiate(config.model, checkpoint=model_ckpt)

    logger.info("Start instantiating dataloaders")
    dataloaders = erc.datasets.get_dataloaders(ds_cfg=config.dataset,
                             dl_cfg=config.dataloader,
                             modes=config.misc.modes)
    
    logger.info("Start instantiating Pytorch-Lightning Trainer")
    module = hydra.utils.instantiate(config.module,
                                    model=model,
                                    optimizer=config.optim,
                                    scheduler=config.scheduler,
                                    train_loader=dataloaders["train"],
                                    valid_loader=dataloaders["valid"])
    return module, dataloaders


def train(config: omegaconf.DictConfig) -> None:
    module, dataloaders = setup_trainer(config)
    
    # Logger Setup
    logger = hydra.utils.instantiate(config.logger)
    logger.watch(module)
    # Hard-code config uploading
    wandb.config.update(
        omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )

    # Callbacks
    callbacks: dict = hydra.utils.instantiate(config.callbacks)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer,
                                                  logger=logger,
                                                  callbacks=list(callbacks.values()))
    trainer.fit(model=module,
                train_dataloaders=dataloaders["train"],
                val_dataloaders=dataloaders["valid"])
    
    
def inference(config: omegaconf.DictConfig) -> None:
    module, dataloaders = setup_trainer(config)
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)
    prediction = trainer.predict(model=module,
                                 dataloaders=dataloaders["valid"],
                                 return_predictions=True)
    prediction = module._sort_outputs(prediction)
    # TODO
    breakpoint()
