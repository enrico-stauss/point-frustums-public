import pytest
import torch

from point_frustums.config_dataclasses.point_frustums import (
    ConfigDecorate,
    ConfigDiscretize,
    ConfigReduce,
    ConfigTransformerFrustumEncoder,
    ConfigVectorize,
    DecoratorFunction,
)
from point_frustums.models.backbones import FrustumEncoder
from point_frustums.models.backbones.frustum_encoder import TransformerFrustumEncoder

N_AZI, N_POL = 8, 4


def make_tfe(max_points=None, dropout: float = 0.0) -> TransformerFrustumEncoder:
    torch.manual_seed(0)
    return TransformerFrustumEncoder(
        n_splits_azi=N_AZI,
        n_splits_pol=N_POL,
        n_channels_in=6,
        n_channels_embedding=8,
        n_channels_projection=12,
        n_heads=2,
        n_encoders=2,
        max_points_per_frustum=max_points,
        dropout=dropout,
    )


def frustum_layout(counts: list[int], frustums: list[int]):
    """Sorted points with the given number of points in each of the given (nonempty) frustums."""
    i_unique = torch.tensor(frustums)
    counts = torch.tensor(counts)
    i_inv = torch.repeat_interleave(torch.arange(len(counts)), counts)
    return i_unique, i_inv, counts


@pytest.mark.parametrize("training", [True, False])
def test_select_points(training: bool):
    tfe = make_tfe(max_points=5).train(training)
    _, i_inv, counts = frustum_layout([1, 5, 12, 3], [0, 3, 9, 30])
    point_index = tfe._select_points(i_inv, counts)  # pylint: disable=protected-access
    # Each point is selected at most once and the selection remains sorted by frustum
    assert point_index.unique().numel() == point_index.numel()
    assert (i_inv[point_index].diff() >= 0).all()
    # At most max_points_per_frustum per frustum, all points of the smaller frustums
    assert torch.equal(torch.bincount(i_inv[point_index], minlength=len(counts)), counts.clamp(max=5))


def test_output_is_invariant_to_the_point_order():
    tfe = make_tfe().eval()
    i_unique, i_inv, counts = frustum_layout([1, 5, 12, 3], [0, 3, 9, 30])
    pc = torch.randn(int(counts.sum()), 6)
    out = tfe(pc, i_unique=i_unique, i_inv=i_inv, counts=counts)
    assert out.shape == (N_AZI * N_POL, 12)
    # Empty frustums are zero before the final FFN, hence equal; nonempty ones differ
    assert not torch.allclose(out[i_unique], out[1].expand(len(i_unique), -1))

    # Shuffle the points within their frustum
    permutation = torch.sort(i_inv.float() + torch.rand(len(i_inv))).indices
    out_shuffled = tfe(pc[permutation], i_unique=i_unique, i_inv=i_inv, counts=counts)
    torch.testing.assert_close(out, out_shuffled)


def test_frustum_encoder_with_tfe_backward():
    discretize = ConfigDiscretize(
        n_splits_azi=N_AZI, n_splits_pol=N_POL, fov_azi_deg=(-180, 180), fov_pol_deg=(60, 100)
    )
    encoder = FrustumEncoder(
        channels_in=["x", "y", "z", "intensity", "timestamp", "radial", "polar", "azimuthal"],
        discretize=discretize,
        decorate=ConfigDecorate(
            functions=[
                DecoratorFunction(id="relative_angle", channel="azimuthal", std=0.29),
                DecoratorFunction(id="distance_to_mean", channel="radial", std=0.55),
            ],
            channels_out=["z", "intensity", "radial", "delta_azi", "delta_r"],
        ),
        vectorize=ConfigVectorize(layers=[16]),
        symmetrize=[],
        reduce=ConfigReduce(layers=[16]),
        transformer_frustum_encoder=ConfigTransformerFrustumEncoder(
            n_channels_embedding=8, n_channels_projection=16, n_heads=2, n_encoders=1, max_points_per_frustum=None
        ),
    )
    batch = []
    for n_points in (300, 50):
        radial = 1 + 30 * torch.rand(n_points)
        polar = torch.deg2rad(60 + 40 * torch.rand(n_points))
        azimuthal = torch.deg2rad(-180 + 360 * torch.rand(n_points))
        xyz = torch.stack(
            [
                radial * torch.sin(polar) * torch.cos(azimuthal),
                radial * torch.sin(polar) * torch.sin(azimuthal),
                radial * torch.cos(polar),
            ],
            dim=1,
        )
        batch.append(torch.cat([xyz, torch.rand(n_points, 2), radial[:, None], polar[:, None], azimuthal[:, None]], 1))
    features = encoder.train()(batch)
    assert features.shape == (2, 16, N_POL, N_AZI)
    features.sum().backward()
    assert encoder.tfe.decoder.grad is not None and encoder.tfe.decoder.grad.abs().sum() > 0


def test_checkpointing_gives_the_same_gradients():
    i_unique, i_inv, counts = frustum_layout([1, 5, 12, 3], [0, 3, 9, 30])
    pc = torch.randn(int(counts.sum()), 6)
    grads = []
    for checkpointing in (False, True):
        tfe = make_tfe(dropout=0.2).train()
        tfe.checkpointing = checkpointing
        torch.manual_seed(1)
        tfe(pc, i_unique=i_unique, i_inv=i_inv, counts=counts).sum().backward()
        grads.append([p.grad.clone() for p in tfe.parameters()])
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b)


def test_feed_forward_dropout_leaves_the_identity_path_intact():
    ffn = TransformerFrustumEncoder.FeedForward(16, dropout=0.5).train()
    # With a residual branch that outputs zero, the block is the identity even while dropout is active
    torch.nn.init.zeros_(ffn.network[-1].weight)
    torch.nn.init.zeros_(ffn.network[-1].bias)
    x = torch.randn(100, 16)
    torch.testing.assert_close(ffn(x), x, rtol=0, atol=0)


@pytest.mark.parametrize("zero_init_residual", [True, False])
def test_zero_init_residual_starts_the_blocks_as_identity(zero_init_residual: bool):
    tfe = TransformerFrustumEncoder(
        n_splits_azi=N_AZI,
        n_splits_pol=N_POL,
        n_channels_in=8,
        n_channels_embedding=8,
        n_channels_projection=12,
        n_heads=2,
        n_encoders=2,
        dropout=0.2,
        zero_init_residual=zero_init_residual,
    ).train()
    _, i_inv, counts = frustum_layout([1, 5, 12, 3], [0, 3, 9, 30])
    x = torch.randn(int(counts.sum()), 8)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(counts, dim=0), (1, 0))
    out = tfe._encoder_block(x, cu_seqlens, i_inv, 0, 0.2)  # pylint: disable=protected-access
    assert torch.equal(out, x) == zero_init_residual


@pytest.mark.parametrize("qk_norm", [True, False])
def test_qk_norm_makes_the_attention_invariant_to_the_query_and_key_scale(qk_norm: bool):
    tfe = TransformerFrustumEncoder(
        n_splits_azi=N_AZI,
        n_splits_pol=N_POL,
        n_channels_in=8,
        n_channels_embedding=8,
        n_channels_projection=12,
        n_heads=2,
        n_encoders=1,
        dropout=0.0,
        zero_init_residual=False,
        qk_norm=qk_norm,
    ).eval()
    i_unique, i_inv, counts = frustum_layout([1, 5, 12, 3], [0, 3, 9, 30])
    pc = torch.randn(int(counts.sum()), 8)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(counts, dim=0), (1, 0))

    def outputs():
        block = tfe._encoder_block(pc, cu_seqlens, i_inv, 0, 0.0)  # pylint: disable=protected-access
        return block, tfe(pc, i_unique=i_unique, i_inv=i_inv, counts=counts)

    block, out = outputs()
    # Scale the query and key projections of the self attention and the key projection of the cross attention (as
    #   the weights may grow during training): with the normalization, the logits and hence the outputs do not change
    with torch.no_grad():
        tfe.qkv_self_attention[0].weight[:16] *= 50
        tfe.qkv_self_attention[0].bias[:16] *= 50
        tfe.kv_cross_attention.weight[:12] *= 50
        tfe.kv_cross_attention.bias[:12] *= 50
    block_scaled, out_scaled = outputs()
    assert torch.allclose(block, block_scaled, atol=1e-4) == qk_norm
    assert torch.allclose(out, out_scaled, atol=1e-4) == qk_norm
