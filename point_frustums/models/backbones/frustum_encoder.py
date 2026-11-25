from collections.abc import Mapping, Sequence
from dataclasses import asdict
from functools import cached_property
from typing import Optional, Literal, Dict

import torch
import torch.utils.checkpoint
from torch import nn

from point_frustums.config_dataclasses.point_frustums import (
    ConfigDiscretize,
    ConfigVectorize,
    ConfigDecorate,
    ConfigReduce,
    ConfigTransformerFrustumEncoder,
)
from point_frustums.ops.segment_attention import segment_attention, segment_softmax_pooling


@torch.jit.script
def symmetrize_max(  # pylint: disable=too-many-arguments
    pc: torch.Tensor,
    n_frustums: int,
    i_frustum: torch.Tensor,
    i_unique: torch.Tensor,
    i_inv: torch.Tensor,
    counts_padded: torch.Tensor,
    context: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the max via the torch.scatter_reduce OP.
    :param pc:
    :param n_frustums:
    :param i_frustum:
    :param i_inv:
    :param counts_padded:
    :param context:
    :return:
    """
    n_points, n_channels = pc.shape
    # Prepare the output tensor and the index tensor for the scatter op
    i_frustum = i_frustum[:, None].expand(n_points, n_channels)
    # Initialize the output tensor with zeros and fill by the maximum value per frustum
    frustum_max = pc.new_zeros((n_frustums, n_channels))
    frustum_max = torch.scatter_reduce(input=frustum_max, dim=0, index=i_frustum, reduce="amax", src=pc)
    return frustum_max, context


@torch.jit.script
def symmetrize_mean(  # pylint: disable=too-many-arguments
    pc: torch.Tensor,
    n_frustums: int,
    i_frustum: torch.Tensor,
    i_unique: torch.Tensor,
    i_inv: torch.Tensor,
    counts_padded: torch.Tensor,
    context: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the mean via the torch.index_add OP as method to compute the frustum-wise sum.
    :param pc:
    :param n_frustums:
    :param i_frustum:
    :param i_inv:
    :param counts_padded:
    :param context:
    :return:
    """

    if "mean" in context:
        return context["mean"], context

    n_channels = pc.shape[1]
    # index_add instead of scatter_add: on MPS, scatter_add with sorted (hence clustered) indices is >10x slower
    mean = pc.new_zeros((n_frustums, n_channels)).index_add(0, i_frustum, pc)

    # Empty frustums have a sum of zero; dividing by the clamped count avoids boolean indexing (a device sync)
    mean = mean.div(counts_padded.clamp(min=1)[:, None])

    context["mean"] = mean
    return mean, context


@torch.jit.script
def symmetrize_std(  # pylint: disable=too-many-arguments
    pc: torch.Tensor,
    n_frustums: int,
    i_frustum: torch.Tensor,
    i_unique: torch.Tensor,
    i_inv: torch.Tensor,
    counts_padded: torch.Tensor,
    context: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the std via the torch.index_add OP as method to compute the frustum-wise sum.
    :param pc:
    :param n_frustums:
    :param i_frustum:
    :param i_inv:
    :param counts_padded:
    :param context:
    :return:
    """
    if "std" in context:
        return context["std"], context

    if "mean" not in context:
        _, context = symmetrize_mean(
            pc=pc,
            n_frustums=n_frustums,
            i_frustum=i_frustum,
            i_unique=i_unique,
            i_inv=i_inv,
            counts_padded=counts_padded,
            context=context,
        )
    mean = context["mean"]
    n_channels = pc.shape[1]

    pc = pc.sub(mean[i_frustum, :])
    pc = pc.square()

    variance = pc.new_zeros((n_frustums, n_channels)).index_add(0, i_frustum, pc)
    variance = variance.div(counts_padded.clamp(min=1)[:, None])
    # The square root is only differentiable for a positive variance, a zero variance gets a zero gradient
    positive = variance > 0
    std = torch.where(positive, torch.sqrt(torch.where(positive, variance, torch.ones_like(variance))), 0.0)

    context["std"] = std
    return std, context


class TransformerFrustumEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    class FeedForward(nn.Module):
        def __init__(self, n_channels, expansion_factor: int = 4, dropout: float = 0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(n_channels, expansion_factor * n_channels),
                nn.GELU(),
                nn.Linear(expansion_factor * n_channels, n_channels),
            )
            self.norm = nn.LayerNorm(n_channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x) -> torch.Tensor:
            # Pre-LN residual block: the dropout applies to the residual branch only, the identity path stays intact
            return x + self.dropout(self.network(self.norm(x)))

    def __init__(  # pylint: disable=too-many-arguments
        self,
        n_splits_azi: int,
        n_splits_pol: int,
        n_channels_in: int,
        n_channels_embedding: int,
        n_channels_projection: int,
        n_heads: int = 1,
        n_encoders: int = 1,
        max_points_per_frustum: int | None = None,
        forward_expansion: int = 4,
        dropout: float = 0.1,
        checkpointing: bool = False,
        zero_init_residual: bool = True,
        qk_norm: bool = True,
    ):
        """
        Enrich points with information from the local context by applying a sequence of n encoder blocks. The
        semantically richer pointcloud is then projected to the output space and reduced to a pseudo-image by applying
        cross attention with a learnable query/decoder parameter.

        Attention is restricted to the points within a frustum (block-diagonal attention). It is evaluated exactly on the
        packed points, without padding and without storing the attention matrix (point_frustums.ops.segment_attention,
        a kernel for CPU, CUDA and MPS), so the memory is linear in the number of points regardless of how unevenly
        they are distributed over the frustums. The cross attention of the row queries reduces each frustum with a
        segment softmax.
        Optionally, at most max_points_per_frustum points per frustum enter the encoder: a random subset during
        training, an evenly spaced one otherwise.
        :param n_splits_azi:
        :param n_splits_pol:
        :param n_channels_in:
        :param n_channels_embedding:
        :param n_channels_projection:
        :param n_heads:
        :param n_encoders:
        :param max_points_per_frustum:
        :param forward_expansion:
        :param dropout:
        :param checkpointing: Recompute the activations of the encoder blocks in the backward pass instead of storing
            them; these dominate the memory (several per point and channel) while being cheap to evaluate.
        :param zero_init_residual: Initialize the last layer of each residual branch (the output projection of the
            self attention and the second layer of the feed forward networks) with zeros, so that each block starts as
            the identity and the encoder initially passes the embedded points through; the attention is phased in as
            it learns to contribute.
        :param qk_norm: Apply a LayerNorm to the queries and keys (per head) before the dot product, in the self and in
            the cross attention. This bounds the attention logits, which otherwise can grow with the weights and
            saturate the softmax, especially at high learning rates.
        """
        super().__init__()

        self.checkpointing = checkpointing
        self.n_splits_azi = n_splits_azi
        self.n_splits_pol = n_splits_pol
        self.n_splits = n_splits_azi * n_splits_pol

        self.n_channels_in = n_channels_in
        self.n_channels_embedding = n_channels_embedding
        self.n_channels_projection = n_channels_projection

        self.n_heads = n_heads
        self.n_encoders = n_encoders
        self.max_points_per_frustum = max_points_per_frustum
        self.forward_expansion = forward_expansion
        self.dropout = dropout

        assert self.n_channels_embedding % self.n_heads == 0, "Embedding dimension not divisible by `n_heads`."
        self.n_channels_per_head = self.n_channels_embedding // self.n_heads
        assert self.n_channels_per_head in (4, 8, 16, 32, 64), "The channels per head must be one of 4, 8, 16, 32, 64."

        self.fc_pre = None
        if self.n_channels_in != self.n_channels_embedding:
            self.fc_pre = nn.Linear(self.n_channels_in, self.n_channels_embedding)

        self.qkv_self_attention = nn.ModuleList()
        self.out_self_attention = nn.ModuleList()
        self.ffn_self_attention = nn.ModuleList()
        self.norm_self_attention = nn.ModuleList()
        self.q_norm_self_attention = nn.ModuleList()
        self.k_norm_self_attention = nn.ModuleList()
        for _ in range(self.n_encoders):
            self.qkv_self_attention.append(nn.Linear(self.n_channels_embedding, 3 * self.n_channels_embedding))
            self.out_self_attention.append(nn.Linear(self.n_channels_embedding, self.n_channels_embedding))
            self.ffn_self_attention.append(
                self.FeedForward(self.n_channels_embedding, expansion_factor=forward_expansion, dropout=dropout)
            )
            self.norm_self_attention.append(nn.LayerNorm(self.n_channels_embedding))
            self.q_norm_self_attention.append(nn.LayerNorm(self.n_channels_per_head) if qk_norm else nn.Identity())
            self.k_norm_self_attention.append(nn.LayerNorm(self.n_channels_per_head) if qk_norm else nn.Identity())

        self.norm_cross_attention = nn.LayerNorm(self.n_channels_embedding)
        self.kv_cross_attention = nn.Linear(self.n_channels_embedding, 2 * self.n_channels_projection)
        self.q_norm_cross_attention = nn.LayerNorm(self.n_channels_projection) if qk_norm else nn.Identity()
        self.k_norm_cross_attention = nn.LayerNorm(self.n_channels_projection) if qk_norm else nn.Identity()
        # One learnable query per row of the featuremap (the encoding is azimuth invariant)
        self.decoder = nn.Parameter(torch.empty(self.n_splits_pol, self.n_channels_projection))
        nn.init.normal_(self.decoder, std=0.02)
        self.ffn_cross_attention = self.FeedForward(
            self.n_channels_projection, expansion_factor=forward_expansion, dropout=dropout
        )

        if zero_init_residual:
            residual_outputs = [
                *self.out_self_attention,
                *(ffn.network[-1] for ffn in self.ffn_self_attention),
                self.ffn_cross_attention.network[-1],
            ]
            for layer in residual_outputs:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _select_points(self, i_inv: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Select at most max_points_per_frustum points per frustum.
        :param i_inv: The index of the (nonempty) frustum of each point, the points are sorted by frustum [N]
        :param counts: The number of points per nonempty frustum [I]
        :return: The indices of the selected points, still sorted by frustum [S]
        """
        n_points, n_max = i_inv.shape[0], self.max_points_per_frustum
        # The position of each point within its frustum
        position = torch.arange(n_points, device=i_inv.device) - (torch.cumsum(counts, dim=0) - counts)[i_inv]
        if self.training:
            # Shuffle the points within their frustum: sort by the frustum index plus a random fraction. The points
            # remain grouped by frustum, hence keep their positions; the first max_points_per_frustum are selected.
            point_index = torch.sort(i_inv.float() + torch.rand(n_points, device=i_inv.device)).indices
            return point_index[position < n_max]
        # Evenly spaced subset: every stride-th point of a frustum
        stride = (counts.float() / n_max).clamp(min=1.0)[i_inv]
        selected = (torch.remainder(position.float(), stride) < 1.0) & (position.float() < stride * n_max)
        return torch.nonzero(selected).squeeze(-1)

    def _encoder_block(  # pylint: disable=too-many-arguments
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        segment_ids: torch.Tensor,
        i_encoder: int,
        dropout_p: float,
    ) -> torch.Tensor:
        n_points = x.shape[0]
        qkv, out = self.qkv_self_attention[i_encoder], self.out_self_attention[i_encoder]
        # [N, 3E] -> 3 x [N, H, D]
        q, k, v = (
            qkv(self.norm_self_attention[i_encoder](x))
            .reshape(n_points, 3, self.n_heads, self.n_channels_per_head)
            .unbind(1)
        )
        q, k = self.q_norm_self_attention[i_encoder](q), self.k_norm_self_attention[i_encoder](k)
        attention = segment_attention(q, k, v, cu_seqlens=cu_seqlens, segment_ids=segment_ids, dropout_p=dropout_p)
        x = out(attention.reshape(n_points, self.n_channels_embedding)) + x
        return self.ffn_self_attention[i_encoder](x)

    def forward(  # pylint: disable=too-many-locals
        self,
        pc: torch.Tensor,
        i_unique: torch.Tensor,
        i_inv: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enrich the pointcloud with the local context and then apply a reduction to the pseudo-image by performing cross
        attention with the decoder query parameter.
        :param pc: The (decorated) pointcloud, sorted by frustum [N, F]
        :param i_unique: The indices of the nonempty frustums [I]
        :param i_inv: The index into i_unique for each point [N]
        :param counts: The number of points per nonempty frustum [I]
        :return: The featuremap [M, L] (M = n_splits)
        """
        # For brevity, the comments use:
        #   N := n_points, I := n_nonempty_frustums, M := n_frustums
        #   H := n_heads, D := n_channels_per_head, E := n_channels_embedding, L := n_channels_projection
        if self.fc_pre is not None:
            pc = self.fc_pre(pc)

        x, segment_ids = pc, i_inv
        if self.max_points_per_frustum is not None and int(counts.max()) > self.max_points_per_frustum:
            point_index = self._select_points(i_inv, counts)
            x, segment_ids = pc[point_index], i_inv[point_index]
            counts = counts.clamp(max=self.max_points_per_frustum)
        n_points, n_frustums = x.shape[0], counts.shape[0]
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(counts, dim=0), (1, 0))
        dropout_p = self.dropout if self.training else 0.0

        # SELF ATTENTION (THE POINTS ARE ENRICHED WITH THE CONTEXT OF THEIR FRUSTUM)
        for i_encoder in range(self.n_encoders):
            if self.checkpointing and self.training and torch.is_grad_enabled():
                # The CPU RNG state is restored on recomputation, hence the same dropout seed is drawn
                x = torch.utils.checkpoint.checkpoint(
                    self._encoder_block, x, cu_seqlens, segment_ids, i_encoder, dropout_p, use_reentrant=False
                )
            else:
                x = self._encoder_block(x, cu_seqlens, segment_ids, i_encoder, dropout_p)

        # CROSS ATTENTION (THE ROW QUERY ATTENDS TO THE ENRICHED POINTS OF EACH FRUSTUM)
        # [N, 2L] -> 2 x [N, 1, L]
        k, v = self.kv_cross_attention(self.norm_cross_attention(x)).reshape(n_points, 2, 1, -1).unbind(1)
        # The query of each nonempty frustum is the one of its row [I, 1, L]
        query = self.decoder[torch.div(i_unique, self.n_splits_azi, rounding_mode="floor")][:, None, :]
        # The normalization applies to the logits only, the residual connection adds the unnormalized query
        attention = segment_softmax_pooling(
            self.q_norm_cross_attention(query),
            self.k_norm_cross_attention(k),
            v,
            segment_ids=segment_ids,
            n_segments=n_frustums,
            dropout_p=dropout_p,
            training=self.training,
        )

        feats = pc.new_zeros((self.n_splits, self.n_channels_projection))
        feats[i_unique, :] = (attention + query)[:, 0, :]
        # Apply feed forward network (norm, fully connected network, residual connection, dropout)
        return self.ffn_cross_attention(feats)


def _create_ffn(n_channels: int, layers: Sequence[int], dropout: float = 0.1):
    network = []
    for layer in layers:
        network.append(nn.Linear(n_channels, layer))
        network.append(nn.GELU())
        n_channels = layer
    network.append(nn.LayerNorm(n_channels))
    network.append(nn.Dropout(dropout))
    return nn.Sequential(*network)


def decorator_distance_to_mean(
    pc_channel: torch.Tensor,
    n_frustums: int,
    counts_padded: torch.Tensor,
    i_frustum: torch.Tensor,
    i_unique: torch.Tensor,
    i_inv: torch.Tensor,
    std: Optional[float] = 1.0,
):
    """
    :param pc_channel:
    :param n_frustums:
    :param counts_padded:
    :param i_frustum:
    :param i_unique:
    :param i_inv:
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :return:
    """
    mean = torch.scatter_add(pc_channel.new_zeros((n_frustums,)), dim=0, index=i_frustum, src=pc_channel)
    mask = counts_padded != 0
    mean[mask] = mean[mask].div(counts_padded[mask])
    # Subset to the number of non-empty frustums (i_frustum) and then map the frustum mean value to the points (i_inv)
    pc_channel = pc_channel - mean[i_unique][i_inv]
    return (pc_channel / std)[:, None]


def decorator_relative_angle(
    pc_channel: torch.Tensor, delta_rad: float, std: float | None = 1.0, start_rad: float = 0.0
):
    """
    Returns the (rad) angle between the center of the frustum and the point in the respective dimension.
    :param pc_channel:
    :param delta_rad: The angular width of the frustums
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :param start_rad: The start of the field of view, i.e. the boundary of the first frustum
    :return:
    """
    # Normalize output by shifting to the frustum center and dividing by an (empirically estimated) standard deviation
    angle = ((pc_channel - start_rad) % delta_rad - (delta_rad / 2)) / delta_rad
    angle = angle / std
    return angle[:, None]


def decorator_distance_to_mean_local(  # pylint: disable=too-many-arguments
    pc_x: torch.Tensor,
    pc_y: torch.Tensor,
    azimuth_frustum: torch.Tensor,
    n_frustums: int,
    counts_padded: torch.Tensor,
    i_frustum: torch.Tensor,
    component: str,
    std: float | None = 1.0,
):
    """
    The distance of the point to the mean of its frustum in the horizontal plane, rotated into the frame of the frustum
    (by its azimuth) instead of the fixed x/y axes: the longitudinal component along the center ray of the frustum and
    the lateral component perpendicular to it. Unlike the x/y distances, their meaning does not depend on the azimuth,
    as the translation equivariant convolutions along the azimuth require.
    :param pc_x: The x coordinates of the points [N]
    :param pc_y: The y coordinates of the points [N]
    :param azimuth_frustum: The azimuth of the center of the frustum of each point [N]
    :param n_frustums:
    :param counts_padded:
    :param i_frustum:
    :param component: "longitudinal" or "lateral"
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :return:
    """
    counts = counts_padded.clamp(min=1)
    mean_x = pc_x.new_zeros((n_frustums,)).index_add(0, i_frustum, pc_x).div(counts)[i_frustum]
    mean_y = pc_y.new_zeros((n_frustums,)).index_add(0, i_frustum, pc_y).div(counts)[i_frustum]
    delta_x, delta_y = pc_x - mean_x, pc_y - mean_y
    cos, sin = torch.cos(azimuth_frustum), torch.sin(azimuth_frustum)
    if component == "longitudinal":
        distance = cos * delta_x + sin * delta_y
    elif component == "lateral":
        distance = -sin * delta_x + cos * delta_y
    else:
        raise ValueError(f"Invalid component {component}, choose from 'longitudinal' and 'lateral'.")
    return (distance / std)[:, None]


def decorator_distance_to_min(pc_channel: torch.Tensor, n_frustums: int, i_frustum: torch.Tensor, std: float = 1.0):
    """
    The distance of the point to the minimum of its frustum, e.g. for the radial channel, how far the point lies behind
    the closest surface within the frustum.
    :param pc_channel: [N]
    :param n_frustums:
    :param i_frustum:
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :return:
    """
    minimum = pc_channel.new_zeros((n_frustums,)).scatter_reduce(
        0, i_frustum, pc_channel, reduce="amin", include_self=False
    )
    return ((pc_channel - minimum[i_frustum]) / std)[:, None]


class FrustumEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    symmetric_functions = {
        "max": symmetrize_max,
        "mean": symmetrize_mean,
        "std": symmetrize_std,
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        channels_in: Sequence[str],
        discretize: ConfigDiscretize,
        decorate: ConfigDecorate,
        vectorize: ConfigVectorize,
        symmetrize: Sequence[Literal["max", "mean", "std"]],
        reduce: ConfigReduce,
        transformer_frustum_encoder: Optional[ConfigTransformerFrustumEncoder] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels_in = channels_in
        self.discretize = discretize
        self.decorate = decorate
        self.vectorize = vectorize
        self.symmetrize = symmetrize
        self.reduce = reduce
        self.tfe = transformer_frustum_encoder
        self.dropout = dropout

        if self.tfe is not None:
            self.tfe = TransformerFrustumEncoder(
                n_splits_azi=self.discretize.n_splits_azi,
                n_splits_pol=self.discretize.n_splits_pol,
                n_channels_in=self.n_channels["decorated"],
                **asdict(self.tfe),
            )

        self.decorator_keep_channels = self.get_decorator_keep_channels()

        self.ffn_vectorize = _create_ffn(self.n_channels["decorated"], self.vectorize.layers, dropout=dropout)
        self.ffn_reduce = _create_ffn(self.n_channels["squashed"], self.reduce.layers, dropout=dropout)

    @cached_property
    def n_channels(self) -> Mapping:
        """
        The number of channels returned by each of the four stages {input, decorated, squashed, reduced}.
        :return:
        """
        stages = {"input": len(self.channels_in)}

        # Determine the output dimensionality of the decorator
        if self.decorate.n_channels_out is not None:
            stages["decorated"] = self.decorate.n_channels_out
        else:
            stages["decorated"] = stages["input"]

        # Determine the output dimensionality of the vectorizer
        if self.tfe is not None and self.vectorize.n_channels_out is not None:
            assert self.tfe.n_channels_projection == self.vectorize.n_channels_out
            stages["squashed"] = self.vectorize.n_channels_out
        elif self.vectorize.n_channels_out is not None:
            stages["squashed"] = self.vectorize.n_channels_out
        elif self.tfe is not None:
            stages["squashed"] = self.tfe.n_channels_projection
        else:
            stages["squashed"] = stages["decorated"]

        # Determine how many symmetric functions are applied to the vectorized pointcloud
        n_symmetric_functions = len(self.symmetrize)
        if self.tfe is not None:
            n_symmetric_functions += 1
        stages["squashed"] *= n_symmetric_functions

        # Determine the output dimensionality of the dimension reduction
        if self.reduce.n_channels_out is not None:
            stages["reduced"] = self.reduce.n_channels_out
        else:
            stages["reduced"] = stages["squashed"]
        return stages

    def get_decorator_keep_channels(self):
        """Evaluates the indices of the input channels that are kept after applying the decorators in correct order."""
        channels = []
        keep_channels = set(self.channels_in).intersection(self.decorate.channels_out)
        for channel in self.channels_in:
            if channel in keep_channels:
                channels.append(self.channels_in.index(channel))
        return channels

    def filter_fov(self, pc: torch.Tensor) -> torch.Tensor:
        channel_index_azi, channel_index_pol = self.channels_in.index("azimuthal"), self.channels_in.index("polar")
        mask = (
            (pc[:, channel_index_azi] >= self.discretize.fov_azi[0])
            & (pc[:, channel_index_azi] < self.discretize.fov_azi[1])
            & (pc[:, channel_index_pol] >= self.discretize.fov_pol[0])
            & (pc[:, channel_index_pol] < self.discretize.fov_pol[1])
        )
        return pc[mask, ...]

    def get_frustum_index(self, pc: torch.Tensor):
        channel_index_azi, channel_index_pol = self.channels_in.index("azimuthal"), self.channels_in.index("polar")
        i_azi = ((pc[:, channel_index_azi] - self.discretize.fov_azi[0]) // self.discretize.delta_azi).to(torch.int64)
        i_pol = ((pc[:, channel_index_pol] - self.discretize.fov_pol[0]) // self.discretize.delta_pol).to(torch.int64)
        return i_azi + i_pol * self.discretize.n_splits_azi

    @staticmethod
    def sort_by_frustum_index(pc: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices, sorting_indices = torch.sort(indices)
        pc = pc[sorting_indices, :]
        return pc, indices

    def get_frustum_stats(self, indices):
        i_unique, i_inv, counts = indices.unique_consecutive(return_inverse=True, return_counts=True)
        counts_padded = counts.new_zeros((self.discretize.n_splits,))
        counts_padded[i_unique] = counts
        return i_unique, i_inv, counts, counts_padded

    def apply_decorators(
        self,
        pc: torch.Tensor,
        counts_padded: torch.Tensor,
        i_frustum: torch.Tensor,
        i_unique: torch.Tensor,
        i_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the configured decorator functions to obtain invariant features.
        :param pc:
        :param counts_padded:
        :param i_frustum:
        :param i_unique:
        :param i_inv:
        :return:
        """
        decorated_channels = []
        for fn in self.decorate.functions:
            pc_channel = pc[:, self.channels_in.index(fn.channel)]
            match fn.id:
                case "relative_angle":
                    delta_rad = {
                        "azimuthal": self.discretize.delta_azi,
                        "polar": self.discretize.delta_pol,
                    }[fn.channel]
                    start_rad = {
                        "azimuthal": self.discretize.fov_azi[0],
                        "polar": self.discretize.fov_pol[0],
                    }[fn.channel]
                    decorated_channels.append(
                        decorator_relative_angle(
                            pc_channel=pc_channel,
                            delta_rad=delta_rad,
                            std=fn.std,
                            start_rad=start_rad,
                        )
                    )
                case "distance_to_mean":
                    decorated_channels.append(
                        decorator_distance_to_mean(
                            pc_channel=pc_channel,
                            n_frustums=self.discretize.n_splits,
                            counts_padded=counts_padded,
                            i_frustum=i_frustum,
                            i_unique=i_unique,
                            i_inv=i_inv,
                            std=fn.std,
                        )
                    )
                case "distance_to_mean_local":
                    # The channel selects the component: x -> longitudinal, y -> lateral (w.r.t. the frustum center)
                    i_azi = torch.remainder(i_frustum, self.discretize.n_splits_azi)
                    azimuth_frustum = (
                        self.discretize.fov_azi[0] + (i_azi.to(pc.dtype) + 0.5) * self.discretize.delta_azi
                    )
                    decorated_channels.append(
                        decorator_distance_to_mean_local(
                            pc_x=pc[:, self.channels_in.index("x")],
                            pc_y=pc[:, self.channels_in.index("y")],
                            azimuth_frustum=azimuth_frustum,
                            n_frustums=self.discretize.n_splits,
                            counts_padded=counts_padded,
                            i_frustum=i_frustum,
                            component={"x": "longitudinal", "y": "lateral"}[fn.channel],
                            std=fn.std,
                        )
                    )
                case "distance_to_min":
                    decorated_channels.append(
                        decorator_distance_to_min(
                            pc_channel=pc_channel, n_frustums=self.discretize.n_splits, i_frustum=i_frustum, std=fn.std
                        )
                    )
                case _:
                    raise NotImplementedError(f"Decorator {fn.id} not implemented.")
        return torch.cat((pc[:, self.decorator_keep_channels], *decorated_channels), dim=1)

    def apply_symmetric_functions(  # pylint: disable=too-many-arguments
        self,
        pc: torch.Tensor,
        n_frustums: int,
        i_frustum: torch.Tensor,
        i_unique: torch.Tensor,
        i_inv: torch.Tensor,
        counts_padded: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        pseudo_images = []
        context = {}
        for function in self.symmetrize:
            pseudo_image, context = self.symmetric_functions[function](
                pc=pc,
                n_frustums=n_frustums,
                i_frustum=i_frustum,
                i_unique=i_unique,
                i_inv=i_inv,
                counts_padded=counts_padded,
                context=context,
            )
            pseudo_images.append(pseudo_image)
        return pseudo_images

    def vectorize_and_squash(  # pylint: disable=too-many-arguments
        self,
        pc: torch.Tensor,
        i_frustum: torch.Tensor,
        i_unique: torch.Tensor,
        i_inv: torch.Tensor,
        counts: torch.Tensor,
        counts_padded: torch.Tensor,
    ) -> torch.Tensor:
        # If the number of symmetric functions (other than the TFE is nonzero, apply vectorization and symmetric fn
        featuremap = []
        if len(self.symmetrize) > 0:
            # Apply PointNet-like encoder to entire PC and then reduce by symmetric function(s)
            featuremap.extend(
                self.apply_symmetric_functions(
                    pc=self.ffn_vectorize(pc),
                    n_frustums=self.discretize.n_splits,
                    i_frustum=i_frustum,
                    i_unique=i_unique,
                    i_inv=i_inv,
                    counts_padded=counts_padded,
                )
            )
        if self.tfe is not None:
            # Apply the TFE to the pc to enrich points with local context and project to output dimension
            featuremap.append(self.tfe(pc=pc, i_unique=i_unique, i_inv=i_inv, counts=counts))

        return torch.cat(featuremap, dim=1)

    def forward(self, batch: list[torch.Tensor]) -> torch.Tensor:
        batch_size = len(batch)
        features = batch[0].new_empty((batch_size, self.discretize.n_splits, self.n_channels["squashed"]))

        for i, pc in enumerate(batch):
            pc = self.filter_fov(pc)
            i_frustum = self.get_frustum_index(pc)
            pc, i_frustum = self.sort_by_frustum_index(pc, i_frustum)
            i_unique, i_inv, counts, counts_padded = self.get_frustum_stats(i_frustum)
            # Apply decorators to entire PC
            decorated = self.apply_decorators(
                pc,
                counts_padded=counts_padded,
                i_frustum=i_frustum,
                i_unique=i_unique,
                i_inv=i_inv,
            )
            # Squash irregular number of points to regular array by applying symmetric function
            # PyTorch automatically assigns tensors C-contiguous which matches the creation of the global index
            features[i, :, :] = self.vectorize_and_squash(decorated, i_frustum, i_unique, i_inv, counts, counts_padded)

        # Apply dimension reduction to the featuremap
        features = self.ffn_reduce(features)
        # Reshape featuremap to form a pseudo image
        features = features.permute(0, 2, 1).reshape(
            batch_size,
            self.n_channels["reduced"],
            self.discretize.n_splits_pol,
            self.discretize.n_splits_azi,
        )
        return features
