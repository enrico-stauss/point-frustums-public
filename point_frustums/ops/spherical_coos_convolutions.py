from functools import cached_property
from math import isclose
from typing import Optional

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
