import inspect
from typing import Sequence

import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import move_data_to_device
from torch import nn

from point_frustums.models import Detection3DRuntime


class ModelWeightsFromCheckpoint(Callback):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def on_fit_start(self, trainer: Trainer, pl_module: Detection3DRuntime) -> None:
        logger.info(f"Initializing the model weights from the checkpoint '{self.checkpoint_path}'")
        pl_module.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu")["state_dict"])

    def on_validation_start(self, trainer: Trainer, pl_module: Detection3DRuntime) -> None:
        logger.info(f"Initializing the model weights from the checkpoint '{self.checkpoint_path}'")
        pl_module.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu")["state_dict"])


class WeightInitLSUV(Callback):
    """
    A Callback that applies Layer-Sequential UnitVariance (LSUV) initialization to the model.
    Reference: arXiv:1511.06422v7
    """

    def __init__(
        self,
        init_types: Sequence[str],
        t_max=15,
        std_tol=0.01,
        verbose=False,
    ):
        """
        Initialize the callback.
        :param init_types: The names of the module types that should be initialized.
        :param t_max:
        :param std_tol:
        :param verbose:
        """
        self.init_types = tuple(self._str_to_module(m) for m in init_types)
        self.t_max = t_max
        self.std_tol = std_tol
        self.verbose = verbose

    @staticmethod
    def _str_to_module(module_str: str) -> type:
        """
        Load the module instances from torch.nn by string.
        :param module_str:
        :return:
        """
        try:
            return getattr(nn, module_str)
        except AttributeError as err:
            raise ValueError(f"Module {module_str} not found in torch.nn") from err

    @staticmethod
    def _preprocess_batch(module: nn.Module, batch: torch.Tensor | Sequence | dict) -> dict:
        """
        Pack all elements of the input batch into a dict to enable generic kwargs unpacking later on.
        :param module:
        :param batch:
        :return:
        """
        # Get the function's signature and extract the names of the input args
        sig = inspect.signature(module.forward)
        valid_params = set(sig.parameters.keys())

        # Now pack the batch into a dict
        if isinstance(batch, torch.Tensor) and len(valid_params) == 1:
            # A simple one to one mapping
            (param,) = valid_params
            batch = {param: batch}
        elif isinstance(batch, Sequence):
            # Assuming we are dealing with *args, so just put them in order
            _batch = {}
            for key, value in zip(valid_params, batch):
                _batch[key] = value
            batch = _batch
        elif isinstance(batch, dict):
            # If the input batch contains fields that are not meant for model.forward(), throw them out now
            invalid_params = set(batch.keys()) - valid_params
            for param in invalid_params:
                batch.pop(param)

        return batch

    def on_fit_start(self, trainer, pl_module):
        """
        Implement the hook that performs the initialization before the training loop is started.
        :param trainer:
        :param pl_module:
        :return:
        """
        # Get a mini-batch from the train dataloader
        batch = next(iter(trainer.train_dataloader))
        batch = move_data_to_device(batch, device=pl_module.device)
        self.apply_lsuv(pl_module, batch=batch)

    def apply_lsuv(self, model: nn.Module, batch: torch.Tensor | Sequence | dict):
        """
        Apply LSUV initialization to all nn.Modules of the specified types that are part of the overall model.
        :param model:
        :param batch:
        :return:
        """
        model.eval()

        def lsuv_hook(module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            """
            This hook shall be registered as forward hook to each of the nn.Modules that shall be initialized.
            :param module:
            :param input:
            :param output:
            :return:
            """
            # Unpack the solitary input arg
            (input,) = input
            with torch.no_grad():
                nn.init.orthogonal_(module.weight)
                for t in range(self.t_max):
                    output = module.forward(input)
                    output_std = output.std()
                    if abs(output_std - 1) < self.std_tol:
                        break
                    module.weight.data /= output_std + 1e-6
                if self.verbose:
                    logger.debug(f"Layer {module.__class__.__name__}: final std = {output_std:.4f}")
            return output

        # Register the LSUV hook for the entire model
        handles = [m.register_forward_hook(lsuv_hook) for m in model.model.modules() if isinstance(m, self.init_types)]
        batch = self._preprocess_batch(model.model, batch)
        model.model(**batch)

        # Remove the hook after initialization
        for handle in handles:
            handle.remove()

        logger.info("LSUV initialization completed")
