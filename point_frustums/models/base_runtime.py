from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from typing import Optional

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

    @abstractmethod
    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass
