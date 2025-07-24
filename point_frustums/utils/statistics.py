"""Implementation of statistical utilities."""

from typing import NoReturn, Optional

import numpy as np
import torch

DTYPES = int | float | torch.Tensor | np.ndarray


class WelfordsAlgorithm:
    """This is an implementation of Welfords Algorithm [1].
    It can be used for the online computation of mean and variance (biased and unbiased) to reduce memory consumption.

    [1]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self):
        """Initialize the state variables to None."""
        self._count: Optional[DTYPES] = None
        self._mean: Optional[DTYPES] = None
        self._m2: Optional[DTYPES] = None

    def _initialize(self, dformat: type, shape: Optional[tuple], dtype: Optional[type]):
        """Properly initialize the state variables based on the provided input data configuration."""
        assert self._count is None
        if issubclass(dformat, float):
            self._count = 0
            self._mean = 0.0
            self._m2 = 0.0
        elif issubclass(dformat, torch.Tensor):
            self._count = torch.zeros(shape).to(torch.int32)
            self._mean = torch.zeros(shape).to(dtype)
            self._m2 = torch.zeros(shape).to(dtype)
        elif issubclass(dformat, np.ndarray):
            self._count = np.zeros(shape).astype(np.int32)
            self._mean = np.zeros(shape).astype(dtype)
            self._m2 = np.zeros(shape).astype(dtype)
        else:
            raise NotImplementedError

    def update(self, x: DTYPES):
        """
        Perform the update step of Welford's Algorithm.

        This updates the running mean and the running 'sum of squares of differences from the current mean'.
        :param x: One sample of the data that Welford's Algorithm shall be applied to.
        :return:
        """
        if self._count is None:
            self._initialize(
                dformat=type(x),
                shape=x.shape if hasattr(x, "shape") else None,
                dtype=x.dtype if hasattr(x, "dtype") else None,
            )

        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def _ensure_enough_samples(self):
        _too_few_samples = self._count < 2
        if isinstance(self._count, np.ndarray | torch.Tensor):
            _too_few_samples = _too_few_samples.any()
        if _too_few_samples:
            raise ArithmeticError("Cannot finalize Welford's Algorithm for less than 2 samples.")

    def finalize(self) -> tuple[DTYPES, DTYPES, DTYPES]:
        """
        Finalize the computation of Welford's Algorithm.

        :return: Mean, Biased Variance and Unbiased Variance
        """
        self._ensure_enough_samples()

        variance_biased = self._m2 / self._count
        variance_unbiased = self._m2 / (self._count - 1)
        return (self._mean, variance_biased, variance_unbiased)
