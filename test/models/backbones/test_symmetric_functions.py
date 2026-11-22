import torch

from point_frustums.models.backbones.frustum_encoder import symmetrize_mean, symmetrize_std


def _layout(n_points: int = 50, n_frustums: int = 12):
    torch.manual_seed(0)
    # Sorted frustum indices, every second frustum is empty
    i_frustum = torch.sort(torch.randint(0, n_frustums // 2, (n_points,)) * 2).values
    i_unique, i_inv, counts = torch.unique(i_frustum, return_inverse=True, return_counts=True)
    counts_padded = torch.zeros(n_frustums, dtype=torch.long)
    counts_padded[i_unique] = counts
    return i_frustum, i_unique, i_inv, counts_padded


def test_mean_and_std_per_frustum():
    i_frustum, i_unique, i_inv, counts_padded = _layout()
    n_frustums = counts_padded.shape[0]
    pc = torch.randn(i_frustum.shape[0], 4, dtype=torch.float64, requires_grad=True)
    mean, context = symmetrize_mean(pc, n_frustums, i_frustum, i_unique, i_inv, counts_padded, {})
    std, _ = symmetrize_std(pc, n_frustums, i_frustum, i_unique, i_inv, counts_padded, context)
    for frustum in range(n_frustums):
        points = pc[i_frustum == frustum]
        expected_mean = points.mean(0) if len(points) else torch.zeros(4, dtype=pc.dtype)
        expected_std = points.std(0, unbiased=False) if len(points) else torch.zeros(4, dtype=pc.dtype)
        torch.testing.assert_close(mean[frustum], expected_mean)
        torch.testing.assert_close(std[frustum], expected_std)

    def mean_and_std(x):
        m, ctx = symmetrize_mean(x, n_frustums, i_frustum, i_unique, i_inv, counts_padded, {})
        s, _ = symmetrize_std(x, n_frustums, i_frustum, i_unique, i_inv, counts_padded, ctx)
        return torch.cat([m, s], dim=1)

    torch.autograd.gradcheck(mean_and_std, (pc,))


def test_std_gradient_is_finite_for_identical_points():
    i_frustum = torch.tensor([0, 0, 0, 1])
    i_unique, i_inv = torch.tensor([0, 1]), torch.tensor([0, 0, 0, 1])
    counts_padded = torch.tensor([3, 1, 0])
    pc = torch.ones(4, 2, requires_grad=True)
    std, _ = symmetrize_std(pc, 3, i_frustum, i_unique, i_inv, counts_padded, {})
    std.sum().backward()
    assert torch.equal(std, torch.zeros_like(std))
    assert torch.isfinite(pc.grad).all()
