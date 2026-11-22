"""
Exact attention within segments of a packed sequence (block-diagonal attention), e.g. the points within a frustum.

Equivalent to the xFormers memory_efficient_attention with a BlockDiagonalMask or flash attention "varlen", but
available on CPU, CUDA and MPS from one shared implementation (csrc/segment_attention/segment_attention_utils.h). The
attention matrix is never stored, the memory is linear in the number of points regardless of the segment sizes.
"""

import itertools
import math

import torch

from point_frustums.ops import metal


def _forward(q, k, v, segment_ids, cu_seqlens, scale, dropout_p, seed):
    if q.device.type == "mps":
        return metal.segment_attention_forward(q, k, v, segment_ids, cu_seqlens, scale, dropout_p, seed)
    return torch.ops.point_frustums.segment_attention_forward(q, k, v, segment_ids, cu_seqlens, scale, dropout_p, seed)


def _backward(grad_out, q, k, v, out, lse, segment_ids, cu_seqlens, scale, dropout_p, seed):
    if q.device.type == "mps":
        return metal.segment_attention_backward(
            grad_out, q, k, v, out, lse, segment_ids, cu_seqlens, scale, dropout_p, seed
        )
    return torch.ops.point_frustums.segment_attention_backward(
        grad_out, q, k, v, out, lse, segment_ids, cu_seqlens, scale, dropout_p, seed
    )


class SegmentAttentionFunction(torch.autograd.Function):  # pylint: disable=abstract-method
    @staticmethod
    def forward(ctx, q, k, v, segment_ids, cu_seqlens, scale, dropout_p, seed):  # pylint: disable=arguments-differ
        out, lse = _forward(q, k, v, segment_ids, cu_seqlens, scale, dropout_p, seed)
        ctx.save_for_backward(q, k, v, out, lse, segment_ids, cu_seqlens)
        ctx.hyper = (scale, dropout_p, seed)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # pylint: disable=arguments-differ
        q, k, v, out, lse, segment_ids, cu_seqlens = ctx.saved_tensors
        grad_q, grad_k, grad_v = _backward(
            grad_out.contiguous(), q, k, v, out, lse, segment_ids, cu_seqlens, *ctx.hyper
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None


def segment_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, n_points: int | None = None) -> torch.Tensor:
    counts = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.repeat_interleave(torch.arange(len(counts), device=cu_seqlens.device), counts, output_size=n_points)


def segment_attention(  # pylint: disable=too-many-arguments
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
    scale: float | None = None,
    dropout_p: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Every point attends to all points of its segment.
    :param q: [N, H, D], sorted by segment
    :param k: [N, H, D]
    :param v: [N, H, D]
    :param cu_seqlens: The start of every segment followed by N [S + 1]
    :param segment_ids: The segment of every point [N], derived from cu_seqlens if not provided
    :param scale: Defaults to 1 / sqrt(D)
    :param dropout_p: Dropout on the attention probabilities
    :param generator: Used to draw the seed of the dropout mask
    :return: [N, H, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int64).contiguous()
    if segment_ids is None:
        segment_ids = segment_ids_from_cu_seqlens(cu_seqlens, q.shape[0])
    segment_ids = segment_ids.to(device=q.device, dtype=torch.int64).contiguous()
    seed = 0
    if dropout_p > 0.0:
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    dtype = q.dtype
    q, k, v = (x.float().contiguous() for x in (q, k, v))
    out = SegmentAttentionFunction.apply(q, k, v, segment_ids, cu_seqlens, float(scale), float(dropout_p), seed)
    return out.to(dtype)


def segment_attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Dense reference implementation (one SDPA call per segment), only meant for testing."""
    out = []
    bounds = cu_seqlens.tolist()
    for start, end in itertools.pairwise(bounds):
        # [n, H, D] -> [H, n, D]
        q_s, k_s, v_s = (x[start:end].transpose(0, 1) for x in (q, k, v))
        out.append(torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s, scale=scale).transpose(0, 1))
    return torch.cat(out, dim=0)


def segment_softmax_pooling(
    query: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    segment_ids: torch.Tensor,
    n_segments: int,
    scale: float | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Attention of a single query per segment to the points of the segment (cross attention), exact and with linear
    memory from scatter operations, so no kernel is required.
    :param query: [S, H, D]
    :param k: [N, H, D]
    :param v: [N, H, D]
    :param segment_ids: [N]
    :param n_segments: S
    :param scale: Defaults to 1 / sqrt(D)
    :return: [S, H, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    # [N, H]
    scores = (query[segment_ids] * k).sum(-1) * scale
    index = segment_ids[:, None].expand_as(scores)
    maximum = scores.new_full((n_segments, scores.shape[1]), -torch.inf)
    maximum = maximum.scatter_reduce(0, index, scores, reduce="amax", include_self=True).detach()
    weights = torch.exp(scores - maximum[segment_ids])
    normalizer = weights.new_zeros((n_segments, scores.shape[1])).index_add(0, segment_ids, weights)
    weights = weights / normalizer[segment_ids]
    weights = torch.nn.functional.dropout(weights, p=dropout_p, training=training)
    return query.new_zeros(query.shape).index_add(0, segment_ids, weights[..., None] * v)
