import torch

from point_frustums.ops.spherical_coos_convolutions import Conv2dSpherical


def test_strided_convolution_is_centered_on_every_stride_th_node():
    conv = Conv2dSpherical(1, 1, kernel_size=3, stride=2, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 1] = 1.0  # identity at the kernel center
    x = torch.randn(1, 1, 8, 12)
    torch.testing.assert_close(conv(x), x[..., ::2, ::2])


def test_non_square_kernels_keep_the_featuremap_size():
    x = torch.randn(1, 1, 8, 12)
    assert Conv2dSpherical(1, 1, kernel_size=(1, 5))(x).shape == x.shape
    assert Conv2dSpherical(1, 1, kernel_size=(3, 1))(x).shape == x.shape
