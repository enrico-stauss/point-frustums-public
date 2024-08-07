from typing import Optional

import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, Logger

from point_frustums.augmentations.augmentations_other import de_normalize
from point_frustums.config_dataclasses.dataset import Labels
from point_frustums.utils.custom_types import Boxes, Targets
from .plotting import plot_pointcloud_bev, plot_pointcloud_wandb


@torch.no_grad()
def log_pointcloud(
    logger: Logger,
    data: torch.Tensor,
    targets: Optional[Targets],
    detections: Optional[Boxes],
    label_enum: Optional[Labels] = None,
    augmentations_log: Optional[dict] = None,
    tag: str = "",
    step: int = 0,
    lower_z_threshold: float = 0.0,
):
    if isinstance(augmentations_log, dict):
        pc_normalization = augmentations_log.get("Normalize", {}).get("lidar")
        if pc_normalization is not None:
            data = data.clone()
            data = de_normalize(data=data, **pc_normalization)

    #  Transfer everything to the CPU
    data = data.cpu()
    detections: Boxes = {k: v.cpu() for k, v in detections.items()}
    targets: Targets = {k: v.cpu() for k, v in targets.items()}

    if isinstance(logger, TensorBoardLogger):
        fig = plot_pointcloud_bev(points=data, targets=targets, detections=detections)
        logger.experiment.add_figure(tag, fig, step)
    elif isinstance(logger, WandbLogger):
        plot_pointcloud_wandb(
            logger,
            points=data,
            targets=targets,
            detections=detections,
            tag=tag,
            step=step,
            label_enum=label_enum,
            lower_z_threshold=lower_z_threshold,
        )
