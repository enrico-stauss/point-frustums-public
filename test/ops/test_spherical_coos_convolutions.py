import torch

from point_frustums.ops.spherical_coos_convolutions import Conv2dSpherical


def test_non_square_kernels_keep_the_featuremap_size():
    x = torch.randn(1, 1, 8, 12)
    assert Conv2dSpherical(1, 1, kernel_size=(1, 5))(x).shape == x.shape
    assert Conv2dSpherical(1, 1, kernel_size=(3, 1))(x).shape == x.shape
