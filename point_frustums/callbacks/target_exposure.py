import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class TargetExposureMonitor(Callback):
    """
    Log the share of the training targets per class that the model has seen in each epoch, e.g. to verify the effect of
    weighted sampling (see point_frustums.dataloaders.sampling). Logged as `Sampling/class_share/<class>` plus the mean
    number of targets per frame as `Sampling/targets_per_frame`.
    """

    def __init__(self):
        self.counts = None
        self.n_frames = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.counts = None
        self.n_frames = 0

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int) -> None:
        targets = batch.get("targets") if isinstance(batch, dict) else None
        if not targets:
            return
        n_classes = trainer.datamodule.dataset.annotations.n_classes
        if self.counts is None:
            self.counts = torch.zeros(n_classes, dtype=torch.int64)
        for sample_targets in targets:
            self.counts += torch.bincount(sample_targets["class"].long().cpu(), minlength=n_classes)
        self.n_frames += len(targets)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.counts is None or self.n_frames == 0:
            return
        class_aliases = trainer.datamodule.dataset.annotations.class_aliases
        total = max(int(self.counts.sum()), 1)
        metrics = {f"Sampling/class_share/{alias}": c / total for alias, c in zip(class_aliases, self.counts.tolist())}
        metrics["Sampling/targets_per_frame"] = total / self.n_frames
        pl_module.log_dict(metrics, on_epoch=True, sync_dist=True)
