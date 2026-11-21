from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from point_frustums.models.base_models import Detection3DModel


class Detection3DRuntime(LightningModule, metaclass=ABCMeta):
    def __init__(self, *args, model: Detection3DModel, **kwargs):
        """
        The base class for a pytorch lightning module that will serve as runtime for the pytorch model.

        :param args:
        :param model:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model = model

    def forward(  # pylint: disable=arguments-differ
        self,
        lidar: dict[str, list[Tensor]],
        camera: Optional[MutableMapping[str, Tensor]] = None,
        radar: Optional[MutableMapping[str, Tensor]] = None,
    ):
        return self.model.forward(lidar=lidar, camera=camera, radar=radar)

    def _release_mps_cache(self):
        """
        The MPS caching allocator does not reuse blocks well across batches of varying point cloud sizes and keeps
        growing (~1GB per batch). On unified-memory Macs this leads into swapping, which stalls the GPU indefinitely.
        """
        if self.device.type == "mps":
            torch.mps.empty_cache()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._release_mps_cache()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._release_mps_cache()

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._release_mps_cache()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._release_mps_cache()

    @abstractmethod
    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass
