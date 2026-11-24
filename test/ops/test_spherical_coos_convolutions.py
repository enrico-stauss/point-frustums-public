import torch

from point_frustums.ops.spherical_coos_convolutions import Conv2dSpherical, UpsampleSpherical


def test_strided_convolution_is_centered_on_every_stride_th_node():
    conv = Conv2dSpherical(1, 1, kernel_size=3, stride=2, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 1] = 1.0  # identity at the kernel center
    x = torch.randn(1, 1, 8, 12)
    torch.testing.assert_close(conv(x), x[..., ::2, ::2])


def test_upsampling_inverts_the_strided_sampling_of_linear_signals():
    rows, cols = torch.meshgrid(torch.arange(8.0), torch.arange(12.0), indexing="ij")
    x = (2.0 * rows + 3.0 * cols)[None, None]
    up = UpsampleSpherical((2, 2))(x[..., ::2, ::2])
    assert up.shape == x.shape
    # Exact where the interpolation does not wrap (last column) or repeat (last row)
    torch.testing.assert_close(up[..., :-1, :-1], x[..., :-1, :-1])


def test_upsampling_is_circular_in_azimuth_and_replicates_in_polar_direction():
    x = torch.tensor([[[[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]]]])
    up = UpsampleSpherical((2, 2))(x)
    # The last column is interpolated between the last and the first node
    torch.testing.assert_close(up[0, 0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 2.0]))
    # The last row repeats the last input row
    torch.testing.assert_close(up[0, 0, 3], up[0, 0, 2])


def test_non_square_kernels_keep_the_featuremap_size():
    x = torch.randn(1, 1, 8, 12)
    assert Conv2dSpherical(1, 1, kernel_size=(1, 5))(x).shape == x.shape
    assert Conv2dSpherical(1, 1, kernel_size=(3, 1))(x).shape == x.shape
