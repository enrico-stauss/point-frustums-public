from functools import cached_property
from math import isclose
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Conv2dSpherical(nn.Conv2d):
    @cached_property
    def pad_horizontal(self) -> tuple[int, int, int, int]:
        # The kernel size is (height, width), the horizontal padding depends on the width
        assert not isclose(self.kernel_size[1] % 2, 0), (
            f"The horizontal kernel size {self.kernel_size[1]} should be uneven to generate valid padding."
        )
        pad = (self.kernel_size[1] - 1) // 2
        return pad, pad, 0, 0

    @cached_property
    def pad_vertical(self) -> tuple[int, int, int, int]:
        assert not isclose(self.kernel_size[0] % 2, 0), (
            f"The vertical kernel size {self.kernel_size[0]} should be uneven to generate valid padding."
        )
        pad = (self.kernel_size[0] - 1) // 2
        return 0, 0, pad, pad

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        input = F.pad(input, pad=self.pad_horizontal, mode="circular")  # pylint: disable=not-callable
        input = F.pad(input, pad=self.pad_vertical, mode="constant", value=0.0)  # pylint: disable=not-callable
        return F.conv2d(  # pylint: disable=not-callable
            input, weight, bias, stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups
        )


class UpsampleSpherical(nn.Module):
    """
    Upsample a featuremap consistently with the sampling positions of the strided convolutions.

    A strided convolution with an odd kernel (and the padding of Conv2dSpherical) centers its output node k on the input
    node s * k. The node m of the upsampled featuremap is hence located at the position m / f of the input featuremap,
    which is interpolated linearly between the input nodes floor(m / f) and floor(m / f) + 1. The horizontal (azimuthal)
    dimension is circular, so the last node is interpolated with the first; the vertical (polar) dimension repeats the
    last row. In contrast, nn.Upsample assumes that node k is centered on the f input nodes [f * k, f * (k + 1)) and
    shifts the upsampled featuremap by (f - 1) / 2 input nodes w.r.t. the strided convolutions.
    """

    def __init__(self, scale_factor: int | tuple[int, int]):
        super().__init__()
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        self.scale_factor = tuple(int(f) for f in scale_factor)

    @staticmethod
    def _interpolate(x: Tensor, factor: int, dim: int, circular: bool) -> Tensor:
        if factor == 1:
            return x
        size = x.shape[dim]
        position = torch.arange(size * factor, device=x.device)
        index_0 = torch.div(position, factor, rounding_mode="floor")
        index_1 = torch.remainder(index_0 + 1, size) if circular else (index_0 + 1).clamp(max=size - 1)
        weight = (torch.remainder(position, factor).to(x.dtype) / factor).view(
            [-1 if d == dim % x.dim() else 1 for d in range(x.dim())]
        )
        return torch.lerp(x.index_select(dim, index_0), x.index_select(dim, index_1), weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self._interpolate(x, self.scale_factor[0], dim=-2, circular=False)
        return self._interpolate(x, self.scale_factor[1], dim=-1, circular=True)
