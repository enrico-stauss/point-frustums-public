import itertools

import pytest
import torch

import point_frustums  # noqa: F401  # pylint: disable=unused-import  # loads the C++ extension
from point_frustums.ops.segment_attention import (
    SegmentAttentionFunction,
    segment_attention,
    segment_attention_reference,
    segment_ids_from_cu_seqlens,
    segment_softmax_pooling,
)

DEVICES = (
    ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])
)
COUNTS = [1, 5, 40, 3, 130, 17, 2]
MASK32 = 0xFFFFFFFF


def _layout(counts=COUNTS):
    counts = torch.tensor(counts)
    return torch.nn.functional.pad(counts.cumsum(0), (1, 0)), int(counts.sum())


def _hash(x: torch.Tensor) -> torch.Tensor:
    """Port of segment_attention::Hash on int64 tensors (products wrap, only the lower 32 bits are kept)."""
    x = x & MASK32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & MASK32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & MASK32
    return x ^ (x >> 16)


def _dropout_scale(seed: int, h: torch.Tensor, i: torch.Tensor, j: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Port of segment_attention::DropoutScale."""
    x = _hash(seed ^ _hash(h + 0x9E3779B9))
    x = _hash(x ^ i)
    x = _hash(x + j * 0x85EBCA6B)
    u = (x >> 8).double() / 16777216.0
    return torch.where(u < p_drop, 0.0, 1.0 / (1.0 - p_drop))


def _reference_with_dropout(q, k, v, cu_seqlens, p_drop, seed):
    out = []
    bounds = cu_seqlens.tolist()
    scale = q.shape[-1] ** -0.5
    for start, end in itertools.pairwise(bounds):
        q_s, k_s, v_s = (x[start:end].transpose(0, 1) for x in (q, k, v))  # [H, n, D]
        p = torch.softmax(q_s @ k_s.transpose(1, 2) * scale, dim=-1)
        index = torch.arange(start, end)
        h, i, j = torch.meshgrid(torch.arange(q.shape[1]), index, index, indexing="ij")
        out.append(((p * _dropout_scale(seed, h, i, j, p_drop)) @ v_s).transpose(0, 1))
    return torch.cat(out)


def _inputs(device, n, n_heads=2, head_dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n_heads, head_dim, generator=generator).to(device).requires_grad_() for _ in range(3)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("head_dim", [4, 16, 64])
def test_matches_dense_reference(device, head_dim):
    cu_seqlens, n = _layout()
    q, k, v = _inputs(device, n, head_dim=head_dim)
    out = segment_attention(q, k, v, cu_seqlens)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)

    q_ref, k_ref, v_ref = (x.detach().cpu().double().requires_grad_() for x in (q, k, v))
    ref = segment_attention_reference(q_ref, k_ref, v_ref, cu_seqlens)
    grads_ref = torch.autograd.grad(ref, (q_ref, k_ref, v_ref), grad_out.cpu().double())

    torch.testing.assert_close(out.detach().cpu().double(), ref, atol=1e-5, rtol=1e-5)
    for grad, grad_ref in zip(grads, grads_ref):
        torch.testing.assert_close(grad.cpu().double(), grad_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_is_reproduced_in_the_backward_pass(device):
    cu_seqlens, n = _layout()
    q, k, v = _inputs(device, n)
    p_drop, seed = 0.3, 1234
    segment_ids = segment_ids_from_cu_seqlens(cu_seqlens).to(device)
    out = SegmentAttentionFunction.apply(q, k, v, segment_ids, cu_seqlens.to(device), 0.25, p_drop, seed)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)

    q_ref, k_ref, v_ref = (x.detach().cpu().double().requires_grad_() for x in (q, k, v))
    ref = _reference_with_dropout(q_ref, k_ref, v_ref, cu_seqlens, p_drop, seed)
    grads_ref = torch.autograd.grad(ref, (q_ref, k_ref, v_ref), grad_out.cpu().double())

    torch.testing.assert_close(out.detach().cpu().double(), ref, atol=1e-5, rtol=1e-5)
    for grad, grad_ref in zip(grads, grads_ref):
        torch.testing.assert_close(grad.cpu().double(), grad_ref, atol=1e-5, rtol=1e-5)


def test_dropout_rate():
    index = torch.arange(200)
    h, i, j = torch.meshgrid(torch.arange(2), index, index, indexing="ij")
    kept = _dropout_scale(42, h, i, j, 0.1) > 0
    assert abs(1 - kept.double().mean().item() - 0.1) < 0.005


@pytest.mark.parametrize("device", DEVICES)
def test_segment_softmax_pooling(device):
    cu_seqlens, n = _layout()
    segment_ids = segment_ids_from_cu_seqlens(cu_seqlens).to(device)
    _, k, v = _inputs(device, n, n_heads=1, head_dim=8)
    query = torch.randn(len(COUNTS), 1, 8, device=device)
    out = segment_softmax_pooling(query, k, v, segment_ids, n_segments=len(COUNTS))

    bounds = cu_seqlens.tolist()
    for s, (start, end) in enumerate(itertools.pairwise(bounds)):
        ref = torch.nn.functional.scaled_dot_product_attention(
            query[s][:, None], k[start:end].transpose(0, 1), v[start:end].transpose(0, 1)
        )  # [H, 1, D]
        torch.testing.assert_close(out[s], ref[:, 0], atol=1e-5, rtol=1e-5)
