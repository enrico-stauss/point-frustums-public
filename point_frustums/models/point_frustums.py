# pylint: disable=too-many-lines
import os
import random
from copy import deepcopy
from functools import cached_property
from math import prod, exp, isclose, ceil
from typing import Literal, Optional, Any

import pytorch_lightning.loggers
import torch
from ot.unbalanced import sinkhorn_unbalanced_translation_invariant
from torch import nn, optim
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from point_frustums.config_dataclasses.dataset import Annotations
from point_frustums.config_dataclasses.point_frustums import (
    ModelOutputSpecification,
    TargetAssignment,
    Losses,
    Predictions,
    Logging,
)
from point_frustums.geometry.coordinate_system_conversion import sph_to_cart_torch, cart_to_sph_torch
from point_frustums.geometry.quaternion import rotate_2d
from point_frustums.geometry.rotation_matrix import (
    rotation_matrix_from_spherical_coordinates,
    rotation_matrix_to_rotation_6d,
    rotation_matrix_from_rotation_6d,
)
from point_frustums.geometry.utils import get_corners_3d, get_featuremap_projection_boundaries, iou_vol_3d
from point_frustums.metrics.detection.featuremap_level_assignments import (
    FeaturemapLevelAssignments,
    MissedTargets,
    PreassignmentOutflows,
)
from point_frustums.metrics.detection.nds import NuScenesDetectionScore
from point_frustums.ops.nms import nms_3d
from point_frustums.utils.custom_types import Targets
from point_frustums.utils.logging import log_pointcloud
from point_frustums.utils.plotting import render_target_assignment
from .backbones import PointFrustumsBackbone
from .base_models import Detection3DModel
from .base_runtime import Detection3DRuntime
from .heads import PointFrustumsHead
from .necks import PointFrustumsNeck
from .training.losses import vfl
from .training.target_assignment import match_target_projection_to_receptive_field


class PointFrustumsModel(Detection3DModel):
    def __init__(self, backbone: PointFrustumsBackbone, neck: PointFrustumsNeck, head: PointFrustumsHead):
        super().__init__(backbone, neck, head)


class PointFrustums(Detection3DRuntime):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        *args,
        model: PointFrustumsModel,
        target_assignment: TargetAssignment,
        losses: Losses,
        predictions: Predictions,
        logging: Logging,
        **kwargs,
    ):
        super().__init__(*args, model=model, **kwargs)
        self.discretization = deepcopy(self.model.backbone.lidar.discretize)
        self.strides = deepcopy(self.model.neck.fpn.strides)
        self.fpn_layers = deepcopy(self.model.neck.fpn.layers)
        self.fpn_extra_layers = deepcopy(self.model.neck.fpn.extra_layers)
        self.target_assignment = target_assignment
        self.losses = losses
        self.predictions = predictions
        self.logging = logging
        self.featuremap_parametrization = self._register_featuremap_parametrization()

        self.nds_train: Optional[NuScenesDetectionScore] = None
        self.nds_val: Optional[NuScenesDetectionScore] = None
        self._annotations = None

        self.featuremap_assignment_count = FeaturemapLevelAssignments(num_levels=len(self.model.neck.fpn.strides))
        self.preassignment_outflows = PreassignmentOutflows()
        self.missed_targets = MissedTargets()
        self._register_featuremap_arange()
        self._register_featuremap_bounds()

    def _register_featuremap_bounds(self) -> None:
        featuremap_bounds = torch.tensor([0] + self.featuremap_parametrization.layer_sizes_flat).cumsum(0)
        self.register_buffer("_fm_idx_bounds_lower", featuremap_bounds[None, :-1])
        self.register_buffer("_fm_idx_bounds_upper", featuremap_bounds[None, 1:])

    def _register_featuremap_arange(self) -> None:
        self.register_buffer("_fm_arange", torch.arange(self.featuremap_parametrization.total_size))

    def setup(self, *args, **kwargs):
        # Required to access the dataloader at setup time:
        # https://github.com/Lightning-AI/pytorch-lightning/issues/10430#issuecomment-1487753339
        self.trainer.fit_loop.setup_data()
        self.nds_val = NuScenesDetectionScore(self.annotations)
        self.nds_train = NuScenesDetectionScore(self.annotations)
        if bool(self.trainer.fast_dev_run) is False:  # NOQA
            self._setup_logger()

    @property
    def annotations(self) -> Annotations:
        """
        Access the Annotations config class of the (NuScenes) DataModules dataset object.
        :return:
        """
        if self._annotations is None:
            self._annotations = self.trainer.datamodule.dataset.annotations  # NOQA: Unresolved attribute reference...
        return self._annotations

    def _setup_logger(self):
        for logger in self.loggers:
            if isinstance(logger, pytorch_lightning.loggers.TensorBoardLogger):
                losses_ids = self.losses.losses + ["sum"]
                layout = {
                    "Losses": {f"Loss/{l}": ["Multiline", [f"Loss/{l}/train", f"Loss/{l}/val"]] for l in losses_ids}
                }
                layout.update(self.nds_train.tensorboard_custom_scalars)
                logger.experiment.add_custom_scalars(layout)
            elif isinstance(logger, pytorch_lightning.loggers.WandbLogger):
                logger.experiment.watch(self.model, log_freq=100, log="all")
                logger.experiment.log_code(
                    ".", include_fn=lambda p: os.path.relpath(p).split("/")[0] in ("configs", "point_frustums")
                )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["Annotations"] = self.annotations.serialize()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._annotations = Annotations(**checkpoint["Annotations"])

    def _evaluate_receptive_fields(self) -> tuple[dict[str, float], dict[str, float]]:
        def get_rf(k: list, s: list) -> int:
            """
            https://distill.pub/2019/computing-receptive-fields/#other-network-operations
            Araujo, et al., "Computing Receptive Fields of Convolutional Neural Networks", Distill, 2019.
            :param k:
            :param s:
            :return:
            """
            n_layers = len(k)
            assert n_layers == len(s)

            r = 0
            for n_layers in range(0, n_layers):
                r += (k[n_layers] - 1) * prod(s[0:n_layers])
            return int(r + 1)

        def get_erf(rf: int) -> float:
            """
            arXiv:1701.04128
            :param rf:
            :return:
            """
            return rf * exp(-0.43)

        layers = []

        kernels = {}
        strides_pol = {}
        strides_azi = {}

        _kernels_prev = []
        _strides_pol_prev = []
        _strides_azi_prev = []

        for layer, specs in self.fpn_layers.items():
            layers.append(layer)
            kernels[layer] = _kernels_prev + specs.n_blocks * [3]
            layer_stride_pol, layer_stride_azi = specs.stride
            strides_pol[layer] = _strides_pol_prev + [layer_stride_pol] + (specs.n_blocks - 1) * [1]
            strides_azi[layer] = _strides_azi_prev + [layer_stride_azi] + (specs.n_blocks - 1) * [1]

            _kernels_prev = kernels[layer]
            _strides_pol_prev = strides_pol[layer]
            _strides_azi_prev = strides_azi[layer]

        for extra_layer, specs in self.fpn_extra_layers.items():
            layers.append(extra_layer)
            kernels[extra_layer] = _kernels_prev + [3]
            strides_pol[extra_layer] = _strides_pol_prev + [specs.stride[0]]
            strides_azi[extra_layer] = _strides_azi_prev + [specs.stride[1]]

            _kernels_prev = kernels[extra_layer]
            _strides_pol_prev = strides_pol[extra_layer]
            _strides_azi_prev = strides_azi[extra_layer]

        n = min(self.model.head.n_convolutions_classification, self.model.head.n_convolutions_regression)
        for layer in layers:
            kernels[layer].extend(n * [3])
            strides_pol[layer].extend(n * [1])
            strides_azi[layer].extend(n * [1])

        receptive_fields_pol = {l: get_erf(get_rf(kernels[l], strides_pol[l])) for l in layers}
        receptive_fields_azi = {l: get_erf(get_rf(kernels[l], strides_azi[l])) for l in layers}

        return receptive_fields_pol, receptive_fields_azi

    def _register_featuremap_parametrization(self) -> ModelOutputSpecification:
        """
        Register the parameter-set that describe the geometric relation between feature vectors and spherical
        coordinates. They are used to map targets to the feature map and predictions back.
        Those parameters need not be optimized but are required (on the correct device) to run the model and shall
        therefore be registered as buffers.

        Torch does not currently provide adequate methods to do so.
        Requirement specification:
        - Parameters need to be managed automatically together with the model (device, fp precision, etc.)
        - Multi-layer featuremaps require a sequence of parameters (one for each layer)
        - Parameters should possibly be grouped semantically
        Possible approaches:
        - Combine parameters into dedicated module -> But: the purpose of a Module is a different one
        - Use a NestedTensor -> But: NestedTensor is still in prototype stage
        - Use BufferDict -> But: not yet available
        Temporary solution:
        - Iteratively register all parameters on the top level and separately
        - Keep a reference to each parameter in a regular dict

        :return:
        """
        layers_rfs_pol, layers_rfs_azi = self._evaluate_receptive_fields()
        layers_rfs = list(zip(layers_rfs_pol.values(), layers_rfs_azi.values()))
        reference_boxes_centered = 0.5 * torch.tensor(layers_rfs).repeat(1, 2) * torch.tensor([-1, -1, 1, 1])

        discretize = self.discretization

        size_per_layer = []
        size_per_layer_flat = []
        feat_rfs_center = []
        stride_per_layer = []
        for layer, stride in self.strides.items():
            assert isclose(discretize.n_splits_pol % stride[0], 0), (
                f"The polar input featuremap resolution {discretize.n_splits_pol} is not divisible by the stride "
                f"{stride[0]} on layer {layer}."
            )
            assert isclose(discretize.n_splits_azi % stride[1], 0), (
                f"The azimuthal input featuremap resolution {discretize.n_splits_azi} is not divisible by the "
                f"stride {stride[1]} on layer {layer}."
            )
            stride_per_layer.append(stride)
            size_per_layer.append((int(discretize.n_splits_pol / stride[0]), int(discretize.n_splits_azi / stride[1])))
            size_per_layer_flat.append(int(discretize.n_splits / prod(stride)))

            # Get the node position w.r.t. the original input data shape
            nodes_pol = torch.arange(0, discretize.n_splits_pol - 1e-8, step=stride[0])
            nodes_azi = torch.arange(0, discretize.n_splits_azi - 1e-8, step=stride[1])

            # Create meshgrid representation for easy indexing (in row-major ordering)
            nodes_azi, nodes_pol = torch.meshgrid(nodes_azi, nodes_pol, indexing="xy")
            nodes_azi, nodes_pol = nodes_azi.flatten(), nodes_pol.flatten()
            # Merge to obtain the [x, y] coordinates of the centers
            feat_rfs_center.append(torch.stack((nodes_azi, nodes_pol), dim=1))

        stride_per_layer = torch.tensor(stride_per_layer)

        feat_rfs_center = torch.cat(feat_rfs_center, dim=0)
        self.register_buffer("feat_rfs_centers", feat_rfs_center)

        feat_rfs = reference_boxes_centered.repeat_interleave(torch.tensor(size_per_layer_flat), dim=0)
        feat_rfs += feat_rfs_center.repeat(1, 2)
        self.register_buffer("feat_rfs", feat_rfs)

        feat_rfs_sizes = torch.tensor(layers_rfs).repeat_interleave(torch.tensor(size_per_layer_flat), dim=0)
        self.register_buffer("feat_rfs_sizes", feat_rfs_sizes)

        # Scale-scale-shift grid centers to angular center coordinates
        feat_center_pol_azi = feat_rfs_center[:, [1, 0]]
        feat_center_pol_azi /= torch.tensor([discretize.n_splits_pol, discretize.n_splits_azi])
        feat_center_pol_azi *= torch.tensor([discretize.range_pol, discretize.range_azi])
        feat_center_pol_azi += torch.tensor([discretize.fov_pol[0], discretize.fov_azi[0]])
        frustum_width_per_layer = stride_per_layer * torch.tensor([discretize.delta_pol, discretize.delta_azi])
        feat_center_pol_azi += (frustum_width_per_layer / 2).repeat_interleave(torch.tensor(size_per_layer_flat), dim=0)
        self.register_buffer("feat_center_pol_azi", feat_center_pol_azi)

        return ModelOutputSpecification(
            strides=list(self.strides.values()),
            layer_sizes=size_per_layer,
            layer_sizes_flat=size_per_layer_flat,
            total_size=sum(size_per_layer_flat),
        )

    @property
    def receptive_fields(self):
        """
        The bounding box corresponding to the receptive field of the feature vector.
        :return:
        """
        return self.get_buffer("feat_rfs")

    @property
    def receptive_fields_sizes(self):
        """
        The receptive field size of the feature vector.
        :return:
        """
        return self.get_buffer("feat_rfs_sizes")

    @property
    def receptive_fields_centers(self):
        """
        The featuremap coordinate/index corresponding to the featuremap location.
        :return:
        """
        return self.get_buffer("feat_rfs_centers")

    @property
    def feature_vectors_angular_centers(self) -> torch.Tensor:
        """
        The {theta, phi}-coordinate corresponding to the featuremap location.
        :return:
        """
        return self.get_buffer("feat_center_pol_azi")

    @cached_property
    def top_k(self):
        return torch.tensor(self.predictions.top_k, device=self.device)

    @cached_property
    def n_detections(self):
        return torch.tensor(self.predictions.n_detections, device=self.device)

    @staticmethod
    def _one_hot(label: torch.Tensor, num_classes: int, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert label.dim() == 1
        one_hot = F.one_hot(label.clip(min=0), num_classes=num_classes)  # pylint: disable=not-callable
        if index is not None:
            one_hot = one_hot[index, ...]
            mask = torch.ge(index, torch.zeros_like(index))
            one_hot[~mask, :] = 0
        return one_hot.float()

    def _encode_center(self, x: torch.Tensor, *, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the M spherical center coordinates w.r.t. the angular position of the feature vectors. Scale the angular
        delta furthermore with the distance.
        :param x:
        :param idx_feat:
        :return:
        """
        center_pol_azi = self.feature_vectors_angular_centers
        if idx_feat is not None:
            center_pol_azi = center_pol_azi[idx_feat, :]
        x = x.clone()
        # TODO: One approach to improve the convergence of the angular center could be to scale with radial distance.
        #   1. Either I scale the encoding component, then I have to adjust the decoding equivalently, this I can
        #      motivate with the analogy of using the length of the opposite leg to the angle rather than the angle
        #   2. Or I just scale the loss (only possible if I use reduction=None and apply the reduction explicitly)
        #   Option 1 is equivalent to  using the distance in a cartesian COOS, whilst option two still predicts an
        #   angle but penalizes more distant targets stronger. In either way, I'll multiply with the radial distance
        #   and probably divide with a factor in order to balance the different loss components. One last thing would be
        #   consider removing the scaling with the frustum angle. This would imply that in a more finely resolved
        #   discretization, the angular loss components become less relevant.
        x[..., 1] = (x[..., 1] - center_pol_azi[..., 0]) * x[..., 0]
        x[..., 2] = (x[..., 2] - center_pol_azi[..., 1]) * x[..., 0]
        return x

    def _decode_center(self, x: torch.Tensor, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode the M spherical center coordinates w.r.t. the angular position of the feature vectors. Scale the angular
        delta furthermore with the distance.
        :param x:
        :param idx_feat:
        :return:
        """
        center_pol_azi = self.feature_vectors_angular_centers
        if idx_feat is not None:
            center_pol_azi = center_pol_azi[idx_feat, :]
        rad, pol, azi = x.detach().tensor_split(3, dim=-1)
        rad = rad.clamp(min=1e-3)
        return torch.cat((rad, pol / rad + center_pol_azi[..., [0]], azi / rad + center_pol_azi[..., [1]]), dim=-1)

    @staticmethod
    def _encode_wlh(x: torch.Tensor) -> torch.Tensor:
        return x.log()

    @staticmethod
    def _decode_wlh(x: torch.Tensor) -> torch.Tensor:
        return x.exp()

    def _encode_orientation(self, x: torch.Tensor, *, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the provided M orientation matrices w.r.t. the angular coordinates on the featuremap and subsequently in
        form of the 6D encoding suggested in: http://arxiv.org/abs/1812.07035
        :param x: The orientation matrices [M, 3, 3]
        :param idx_feat: The indices of the feature vectors corresponding to the provided orientations [M]
        :return:
        """
        theta, phi = self.feature_vectors_angular_centers.unbind(dim=-1)
        if idx_feat is not None:
            theta = theta[idx_feat]
            phi = phi[idx_feat]
        # Transpose along the last 2 dimensions to invert the direction of rotation
        rotation_matrices = rotation_matrix_from_spherical_coordinates(theta, phi).transpose(-1, -2)
        x = torch.einsum("...ij,...jk->...ik", rotation_matrices, x)
        return rotation_matrix_to_rotation_6d(x)

    def _decode_orientation(self, x: torch.Tensor, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode the provided M vectors encoded in form of the 6D encoding (http://arxiv.org/abs/1812.07035) to the
        matrix form and then rotate them back according to the angular coordinates on the featuremap.
        :param x: The encoded representation of the orientation [M, 6]
        :param idx_feat: The indices of the feature vectors corresponding to the provided orientations [M]
        :return:
        """
        theta, phi = self.feature_vectors_angular_centers.unbind(dim=-1)
        if idx_feat is not None:
            theta = theta[idx_feat]
            phi = phi[idx_feat]
        x = rotation_matrix_from_rotation_6d(x)
        rotation_matrices = rotation_matrix_from_spherical_coordinates(theta, phi)
        return torch.einsum("...ij,...jk->...ik", rotation_matrices, x)

    def _encode_velocity(
        self, x: torch.Tensor, ego_velocity: torch.Tensor, *, idx_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode the velocity as [normal, tangential] velocity by rotating it by the negative of the azimuthal position.
        :param x:
        :param ego_velocity:
        :param idx_feat:
        :return:
        """
        x = x - ego_velocity
        phi = self.feature_vectors_angular_centers[:, 1]
        if idx_feat is not None:
            phi = phi[idx_feat]
        # Invert the rotation by negating the sign of phi, this is possible only in the 2D case
        return rotate_2d(phi=-phi, x=x[..., [0, 1]])

    def _decode_velocity(
        self, x: torch.Tensor, ego_velocity: torch.Tensor, idx_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode the [normal, tangential] velocity into cartesian [x, y] components by rotating it by the azimuthal
        position.
        :param x:
        :param ego_velocity:
        :param idx_feat:
        :return:
        """
        phi = self.feature_vectors_angular_centers[:, 1]
        if idx_feat is not None:
            phi = phi[idx_feat]

        x = rotate_2d(phi=phi, x=x)
        return x + ego_velocity[..., [0, 1]]

    def _assign_target_index_to_features(
        self,
        targets: Targets,
        target_labels: torch.Tensor,
        target_corners: torch.Tensor,
        feat_labels: torch.Tensor,
        feat_centers: torch.Tensor,
        feat_corners: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assign a target to each feature vector of the concatenated featuremaps. Apply to a single sample from the batch.
        :param targets:
        :param target_corners:
        :param target_labels:
        :param feat_labels:
        :param feat_centers:
        :param feat_corners:
        :return: An integer tensor where -1 represents background and all values >= 0 correspond to one of the targets.
        """
        if target_labels.numel() == 0:
            return torch.full((self.featuremap_parametrization.total_size,), -1, device=self.device)

        targets_spherical_projection = get_featuremap_projection_boundaries(
            centers=targets["center"],
            wlh=targets["wlh"],
            orientation=targets["orientation"],
            fov_pol=self.discretization.fov_pol,
            fov_azi=self.discretization.fov_azi,
            delta_pol=self.discretization.delta_pol,
            delta_azi=self.discretization.delta_azi,
        )
        binary_pre_mapping = match_target_projection_to_receptive_field(
            targets_projections=targets_spherical_projection,
            rf_centers=self.receptive_fields_centers,
            rf_sizes=self.receptive_fields_sizes,
            layer_sizes_flat=self.featuremap_parametrization.layer_sizes_flat,
            base_featuremap_width=self.discretization.n_splits_azi,
            upper_bound_multiplier=self.target_assignment.upper_bound_multiplier,
            lower_bound_multiplier=self.target_assignment.lower_bound_multiplier,
        )

        # Get the indices that map between feature vectors and targets
        #   Column 0: Indices of feature vectors, use to obtain a subset of predictions that can be mapped to targets
        #   Column 1: Indices of targets, use to broadcast targets to the assigned subset of feature vectors
        idx = torch.nonzero(binary_pre_mapping).unbind(-1)

        # Evaluate pairwise IoU between the matched subset of predictions corresponding targets
        iou = torch.zeros_like(binary_pre_mapping, dtype=torch.float)
        iou[idx] = iou_vol_3d(feat_corners[idx[0], ...], target_corners[idx[1], ...])[0]
        # Initialize the center distance to all zeros
        center_distance = torch.zeros_like(binary_pre_mapping, dtype=torch.float)
        # Calculate the cartesian distances between predictions and mapped targets (prev.: the distance of radius)
        center_distance[idx] = torch.abs(feat_centers[idx[0], :].norm(dim=-1) - targets["center"][idx[1], :].norm(dim=-1))
        # Normalize the distance by a factor based on the target box size
        center_distance[idx] = center_distance[idx].div(targets["wlh"][idx[1], :].norm(dim=-1))
        # Squash the relative distance to the range [0, 1) by application of tanh
        #   The division by the empirical factor kappa shifts the relevant range into dynamic range of tanh
        center_distance[idx] = center_distance[idx].div(self.target_assignment.kappa).tanh()

        # Broadcast one-hot encoded labels s.t. the first 2 dimensions match the shape of the binary mapping and
        # evaluate the focal loss
        shape = binary_pre_mapping.shape + (self.annotations.n_classes,)
        cost_classification = sigmoid_focal_loss(
            feat_labels[:, None, :].expand(shape),
            target_labels[None, :, :].expand(shape),
            reduction="none",
        ).sum(dim=-1)

        # Evaluate the focal loss against the all-zero labels to use as classification cost for the supplementary
        # background class
        background_labels = torch.zeros_like(feat_labels)
        cost_background = sigmoid_focal_loss(feat_labels, background_labels, reduction="none").sum(dim=-1).tanh()
        # Shift by the factors beta and gamma to ensure that the pre-assigned background will have no smaller cost than
        # any point on the foreground ever
        cost_background += self.target_assignment.beta + self.target_assignment.gamma
        cost_background += 1e-1

        # Combine classification/regression and background cost
        cost_classification = cost_classification.tanh()
        min_fg_cost = cost_classification[binary_pre_mapping].max() + 1e-2
        cost_classification[~binary_pre_mapping] = cost_classification[~binary_pre_mapping].clamp(min=min_fg_cost)

        # Initialize to the bias which ensures that the background is never assigned lower cost than the foreground
        cost = (self.target_assignment.beta + self.target_assignment.gamma) * (~binary_pre_mapping).float() + 2e-1
        # Add the scaled classification cost
        cost += self.target_assignment.alpha * cost_classification
        # Add the scaled IoU cost
        cost += self.target_assignment.beta * (1 - iou)
        # Add the scaled center distance cost
        cost += self.target_assignment.gamma * center_distance
        # Concatenate the pre-determined background class cost
        cost = torch.cat((cost, cost_background[:, None]), dim=1)

        # Set the demand to ones as each prediction requires to be assigned to exactly one class (object or background)
        demand = torch.ones((self.featuremap_parametrization.total_size,), device=self.device).float()
        # Based on the IoU, evaluate how many predictions each of the targets can supply, constrain by min_k and top_k
        supply = torch.topk(iou, dim=0, k=self.target_assignment.max_k).values.sum(dim=0)
        supply = torch.clamp(supply, min=self.target_assignment.min_k)
        # Append the unassigned supply for the background class (note: tensor.sum returns a 0D tensor)
        unassigned_supply = self.featuremap_parametrization.total_size - supply.sum()
        supply = torch.cat((supply, unassigned_supply[None]))

        # Evaluate the transport plan PI of the optimal transport problem (Comprehensive read: arXiv:1803.00567 p 62-84)
        pi = sinkhorn_unbalanced_translation_invariant(
            a=demand,
            b=supply,
            M=cost,
            reg=self.target_assignment.epsilon,
            reg_m=(1.0, 1.0),  # Relax the supply marginal and mildly relax also the demand marginal
            reg_type="kl",
            stopThr=self.target_assignment.threshold,
            numItermax=self.target_assignment.max_iter,
        )

        # Rescale s.t. the maximum pi for each target equals 1.
        scale_factor, _ = pi.max(dim=0)
        pi = pi / scale_factor[None, :]
        # Select the target with the highest value of PI for each patch (if the last row -> background)
        _, match_index = pi.max(dim=1)

        # Assign entries that matched the last/background row to the background i.e. -1
        foreground_index = match_index != binary_pre_mapping.shape[1]
        match_index[~foreground_index] = -1

        if self.trainer.training:
            self.preassignment_outflows((~binary_pre_mapping[foreground_index, match_index[foreground_index]]).sum())
            self.missed_targets(binary_pre_mapping.shape[1] - (match_index.unique().numel() - 1))

        if self.logging.render_target_assignment:
            render_target_assignment(
                target_corners=target_corners,
                assigned_corners=feat_corners[foreground_index, ...],
                target_index=match_index[foreground_index],
                wandb_logger=[l for l in self.loggers if isinstance(l, pytorch_lightning.loggers.WandbLogger)][0],
            )

        return match_index

    @staticmethod
    def _broadcast_targets(targets: list[torch.Tensor], indices: list[torch.Tensor]) -> torch.Tensor:
        """
        Index-select targets that correspond to the foreground predictions and concatenate samples from the batch.
        :param targets:
        :param indices:
        :return:
        """
        return torch.cat([t[i, ...] for i, t in zip(indices, targets)], dim=0)

    def _loss_center(
        self,
        predictions_center_fg: torch.Tensor,
        foreground_idx: torch.Tensor,
        targets_center: list[torch.Tensor],
        targets_indices_fg: list[torch.Tensor],
    ) -> dict[Literal["center_radial", "center_polar", "center_azimuthal"], torch.Tensor]:
        """
        Evaluate the Smooth L1 loss on each of the coordinate components.

        Firstly, broadcast the targets to match the predictions that were assigned foreground, then apply the encoding
        to the targets and lastly compute the loss.
        :param predictions_center_fg:
        :param foreground_idx:
        :param targets_center:
        :param targets_indices_fg:
        :return:
        """
        # Evaluate the loss imposed on the center coordinate
        targets_center_fg = cart_to_sph_torch(self._broadcast_targets(targets_center, targets_indices_fg))
        targets_center_fg = self._encode_center(targets_center_fg, idx_feat=foreground_idx)
        center_radial = F.smooth_l1_loss(
            input=predictions_center_fg[..., 0],
            target=targets_center_fg[..., 0],
            reduction="sum",
            **self.losses.center_radial.kwargs,
        )
        center_polar = F.smooth_l1_loss(
            input=predictions_center_fg[..., 1],
            target=targets_center_fg[..., 1],
            reduction="sum",
            **self.losses.center_polar.kwargs,
        )
        center_azimuthal = F.smooth_l1_loss(
            input=predictions_center_fg[..., 2],
            target=targets_center_fg[..., 2],
            reduction="sum",
            **self.losses.center_azimuthal.kwargs,
        )
        return {"center_radial": center_radial, "center_polar": center_polar, "center_azimuthal": center_azimuthal}

    def _loss_wlh(
        self,
        predictions_wlh_fg: torch.Tensor,
        targets_wlh: list[torch.Tensor],
        targets_indices_fg: list[torch.Tensor],
    ) -> dict[Literal["wlh"], torch.Tensor]:
        targets_wlh_fg = self._encode_wlh(self._broadcast_targets(targets_wlh, targets_indices_fg))
        wlh = F.smooth_l1_loss(
            predictions_wlh_fg,
            targets_wlh_fg,
            reduction="sum",
            **self.losses.wlh.kwargs,
        )
        return {"wlh": wlh}

    def _loss_orientation(
        self,
        predictions_orientation_fg: torch.Tensor,
        foreground_idx: torch.Tensor,
        targets_orientation: list[torch.Tensor],
        targets_indices_fg: list[torch.Tensor],
    ) -> dict[Literal["orientation"], torch.Tensor]:
        targets_orientation_fg = self._broadcast_targets(targets_orientation, targets_indices_fg)
        targets_orientation_fg = self._encode_orientation(targets_orientation_fg, idx_feat=foreground_idx)
        orientation = nn.functional.smooth_l1_loss(
            predictions_orientation_fg,
            targets_orientation_fg,
            reduction="sum",
            **self.losses.orientation.kwargs,
        )
        return {"orientation": orientation}

    def _loss_velocity(
        self,
        predictions_velocity_fg: torch.Tensor,
        ego_velocities: list[torch.Tensor],
        foreground_idx: torch.Tensor,
        targets_velocity: list[torch.Tensor],
        targets_indices_fg: list[torch.Tensor],
    ) -> dict[Literal["velocity"], torch.Tensor]:
        targets_velocity_fg = self._broadcast_targets(targets_velocity, targets_indices_fg)

        # Broadcast the ego velocity per sample to match the foreground vector
        ego_velocities = [v[None, :].repeat(i.numel(), 1) for i, v in zip(targets_indices_fg, ego_velocities)]
        ego_velocities = torch.cat(ego_velocities, dim=0)

        targets_velocity_fg = self._encode_velocity(targets_velocity_fg, ego_velocities, idx_feat=foreground_idx)

        # Make sure that NaN velocity values which may be introduced by the dataloader are filtered out
        data_nan = torch.isnan(targets_velocity_fg)
        targets_velocity_fg = targets_velocity_fg[~data_nan]
        predictions_velocity_fg = predictions_velocity_fg[~data_nan]

        velocity = nn.functional.smooth_l1_loss(
            predictions_velocity_fg,
            targets_velocity_fg,
            reduction="sum",
            **self.losses.velocity.kwargs,
        )

        return {"velocity": velocity}

    def _log_featuremap_assignments(self, indices: torch.Tensor) -> None:
        if not self.trainer.training:
            return

        lower, upper = self.get_buffer("_fm_idx_bounds_lower"), self.get_buffer("_fm_idx_bounds_upper")
        assignments = (indices[:, None] >= lower) * (indices[:, None] < upper)
        self.featuremap_assignment_count(assignments.sum(dim=0))
        counts = self.featuremap_assignment_count.compute()
        self.log_dict({f"TargetAssignments/Level {k}": v for k, v in enumerate(counts)}, reduce_fx="sum")  # NOQA
        missed_targets = self.missed_targets.compute()
        outflows = self.preassignment_outflows.compute()
        self.log("TargetAssignments/Missed Targets", missed_targets, reduce_fx="sum")
        self.log("TargetAssignments/Number of Outflows", outflows, reduce_fx="sum")

    @staticmethod
    def _flatten_output_channels_last(output: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Flatten the last 2 dimensions of all predictions featuremaps, swap dimensions  and concatenate
        output_flat = {}
        for head, layers in output.items():
            output_flat[head] = torch.cat([l.flatten(start_dim=2).transpose(-2, -1) for l in layers.values()], dim=-2)
        return output_flat

    def get_losses(
        self,
        output: dict[
            Literal["class", "attribute", "center", "wlh", "orientation", "velocity", "vfl"], dict[str, torch.Tensor]
        ],
        targets: list[Targets],
        ego_velocities: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        output = self._flatten_output_channels_last(output=output)
        feat_center_decoded = sph_to_cart_torch(self._decode_center(output["center"]))
        feat_wlh_decoded = self._decode_wlh(output["wlh"])
        feat_orientation_decoded = self._decode_orientation(output["orientation"])

        feat_corners = get_corners_3d(
            centers=feat_center_decoded,
            wlh=feat_wlh_decoded,
            orientation=feat_orientation_decoded,
        )

        targets_indices = []
        indices = []
        foreground = []
        targets_label = []
        targets_attribute = []
        targets_iou = []
        for i in range(output["class"].shape[0]):
            # Assign the index of a target/the background to each prediction
            with torch.no_grad():
                target_labels = self._one_hot(targets[i]["class"], self.annotations.n_classes)
                target_corners = get_corners_3d(targets[i]["center"], targets[i]["wlh"], targets[i]["orientation"])
                index = self._assign_target_index_to_features(
                    targets=targets[i],
                    target_labels=target_labels,
                    target_corners=target_corners,
                    feat_labels=output["class"][i, ...],
                    feat_centers=feat_center_decoded[i, ...],
                    feat_corners=feat_corners[i, ...],
                )
            indices.append(index)
            fg = torch.ge(index, 0)
            foreground.append(fg)
            target_indices = index[fg]
            targets_indices.append(target_indices)

            if target_labels.numel() == 0:
                targets_label.append(target_labels.new_full(index.shape + (self.annotations.n_classes,), 0).float())
                targets_attribute.append(
                    targets[i]["attribute"].new_full(index.shape + (self.annotations.n_attributes,), 0).float()
                )
                targets_iou.append(target_labels.new_full(index.shape + (self.annotations.n_classes,), 0).float())
                continue

            target_labels = target_labels[index, :]
            target_labels[~fg, :] = 0
            targets_label.append(target_labels)
            targets_attribute.append(self._one_hot(targets[i]["attribute"], self.annotations.n_attributes, index))
            iou = torch.zeros_like(index, dtype=torch.float)
            iou[fg] = iou_vol_3d(feat_corners[i, fg, ...], target_corners[target_indices, ...])[0]
            # Take the one-hot encoded and broadcasted target labels from this sample and scale with the IoU
            targets_iou.append(targets_label[-1] * iou[:, None])

        foreground = torch.stack(foreground, dim=0)
        foreground_idx = foreground.nonzero()[:, 1]
        targets_label = torch.stack(targets_label, dim=0)
        targets_attribute = torch.stack(targets_attribute, dim=0)
        targets_iou = torch.stack(targets_iou, dim=0).detach()

        self._log_featuremap_assignments(foreground_idx)

        # Evaluate the losses that are applied to the entire featuremap (class/attribute/IoU)
        losses = {
            "class": sigmoid_focal_loss(output["class"], targets_label, reduction="sum"),
            "attribute": sigmoid_focal_loss(output["attribute"], targets_attribute, reduction="sum"),
            "vfl": vfl(output["vfl"], targets_iou, reduction="sum", **self.losses.vfl.kwargs),
        }

        # Evaluate the losses that are applied only to the foreground (center/wlh/orientation/velocity)
        if len(foreground_idx) != 0:
            # Evaluate the loss imposed on the center coordinate
            losses.update(
                self._loss_center(
                    predictions_center_fg=output["center"][foreground, ...],
                    foreground_idx=foreground_idx,
                    targets_center=[t["center"] for t in targets],
                    targets_indices_fg=targets_indices,
                )
            )
            # Evaluate the loss imposed on the size
            losses.update(
                self._loss_wlh(
                    predictions_wlh_fg=output["wlh"][foreground, ...],
                    targets_wlh=[t["wlh"] for t in targets],
                    targets_indices_fg=targets_indices,
                )
            )
            # Evaluate the loss imposed on the orientation
            losses.update(
                self._loss_orientation(
                    predictions_orientation_fg=output["orientation"][foreground, ...],
                    foreground_idx=foreground_idx,
                    targets_orientation=[t["orientation"] for t in targets],
                    targets_indices_fg=targets_indices,
                )
            )
            # Evaluate the loss imposed on the velocity
            losses.update(
                self._loss_velocity(
                    predictions_velocity_fg=output["velocity"][foreground, ...],
                    ego_velocities=ego_velocities,
                    foreground_idx=foreground_idx,
                    targets_velocity=[t["velocity"] for t in targets],
                    targets_indices_fg=targets_indices,
                )
            )

        n = max(1, len(foreground_idx))
        losses = {loss: value / n for loss, value in losses.items()}
        return losses  # NOQA: Expected type 'dict[Literal[...], torch.Tensor]', got 'dict[str, torch.Tensor]' instead

    def sum_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = []
        for key, val in losses.items():
            if key == "class":
                key = "label"
            loss_config = getattr(self.losses, key)
            if not loss_config.active or self.trainer.current_epoch < loss_config.from_epoch:
                continue
            loss.append(loss_config.weight * val)
        return torch.stack(loss, dim=0).sum(dim=0)

    @staticmethod
    def _subset_sample_mask(score: torch.Tensor, sample_mask: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
        score = score.clone()
        score[~sample_mask] = 0
        sample_mask[score.argsort(descending=True)[top_k:]] = False
        return sample_mask

    def get_detections(
        self,
        output: dict[
            Literal["class", "attribute", "center", "wlh", "orientation", "velocity", "vfl"], dict[str, torch.Tensor]
        ],
        ego_velocities: list[torch.Tensor],
    ) -> list[dict[Literal["score", "class", "attribute", "center", "wlh", "orientation", "velocity"], torch.Tensor]]:
        """
        Select confident outputs are detections and transform to the coordinate system of the training targets.
        :param output:
        :param ego_velocities:
        :return:
        """
        output = self._flatten_output_channels_last(output=output)

        score_vfl_factor = self.predictions.score_vfl_factor
        loss_config = getattr(self.losses, "vfl")
        if not loss_config.active or self.trainer.current_epoch < loss_config.from_epoch:
            score_vfl_factor = 0.0

        score = torch.sigmoid(output["class"] + score_vfl_factor * output["vfl"])
        batch_size = score.shape[0]

        # Get the indices of outputs that surpass the confidence threshold:
        # -> column 0: the sample index
        # -> column 1: the featuremap index
        # -> column 2: the class label index
        indices = torch.ge(score, self.predictions.score_threshold).nonzero().unbind(-1)

        # Subset the network outputs
        output = {head: tensor[indices[0], indices[1], ...] for head, tensor in output.items()}

        # Decode the box quantities
        attribute = output["attribute"]
        center = sph_to_cart_torch(self._decode_center(output["center"], idx_feat=indices[1]))
        wlh = self._decode_wlh(output["wlh"])
        orientation = self._decode_orientation(output["orientation"], idx_feat=indices[1])
        # Broadcast the ego_velocities to match the subset of predictions
        n_per_sample = [torch.eq(indices[0], i).count_nonzero() for i, _ in enumerate(ego_velocities)]
        ego_velocity = torch.cat([v[None, :].repeat(n, 1) for v, n in zip(ego_velocities, n_per_sample)], dim=0)
        velocity = self._decode_velocity(output["velocity"], ego_velocity=ego_velocity, idx_feat=indices[1])

        # Subset and flatten the score, now corresponds to the columns of indices
        score = score[indices]
        batch_detections = []
        for sample_index in range(batch_size):
            sample_detections = {}

            sample_mask = torch.eq(indices[0], sample_index)
            # Subset sample output to the configured number of top predictions that should be considered for NMS
            sample_mask = self._subset_sample_mask(score, sample_mask, top_k=torch.min(sample_mask.sum(), self.top_k))

            # Subset the sample subset by performing NMS and then selecting only the top duplicate-free detections
            boxes = get_corners_3d(
                centers=center[sample_mask, ...],
                wlh=wlh[sample_mask, ...],
                orientation=orientation[sample_mask, ...],
            )
            # Write NMS results directly to the sample mask
            sample_mask[sample_mask.clone()] = nms_3d(
                labels=indices[2][sample_mask],
                scores=score[sample_mask],
                boxes=boxes,
                iou_threshold=self.predictions.nms_threshold,
            )

            # Subset the duplicate-free outputs to the configured number of detections
            sample_mask = self._subset_sample_mask(score, sample_mask, torch.min(sample_mask.sum(), self.n_detections))

            # The score already corresponds to the detections subset
            sample_detections["score"] = score[sample_mask]
            # The class and center are in the intermediate sample top subset format
            sample_detections["class"] = indices[2][sample_mask]
            sample_detections["center"] = center[sample_mask, ...]
            # The remaining quantities are the subset of the network output that surpasses the initial score threshold
            sample_detections["attribute"] = attribute[sample_mask, ...].max(dim=-1).indices
            sample_detections["wlh"] = wlh[sample_mask, ...]
            sample_detections["orientation"] = orientation[sample_mask, ...]
            sample_detections["velocity"] = velocity[sample_mask, ...]

            batch_detections.append(sample_detections)

        return batch_detections  # NOQA: Cannot infer return type Tensor in the above lines

    def _post_loss(self, step_output, batch, mode: Literal["train", "val"]):
        for key, value in step_output["losses"].items():
            self.log(f"Loss/{key}/{mode}", value, batch_size=len(batch["metadata"]))

    def _post_pointcloud(self, batch, detections, logging_frequency: int = 1, prefix: Optional[str] = None):
        if logging_frequency is None:
            return

        if isinstance(prefix, str):
            prefix = prefix + " "
        else:
            prefix = ""

        lower_z_threshold = 0.0
        if self.annotations.coos == "EGO":
            lower_z_threshold = 0.1

        batch_size = len(batch["metadata"])
        for i in range(batch_size):
            if random.random() > (1 / logging_frequency):
                continue

            if self.annotations.coos == "LIDAR_TOP":
                lower_z_threshold = 0.1 - batch["metadata"][i]["LIDAR_TOP"]["translation"][-1]

            tag = prefix + "Sample: " + batch["metadata"][i]["sample_token"]
            for logger in self.loggers:
                log_pointcloud(
                    logger,
                    data=batch.get("lidar")["LIDAR_TOP"][i],
                    augmentations_log=batch["metadata"][i].get("augmentations", {}),
                    targets=batch.get("targets")[i],
                    detections=detections[i],
                    label_enum=self.annotations.classes,
                    tag=tag,
                    step=self.global_step,
                    lower_z_threshold=lower_z_threshold,
                )

    TRAIN_STEP_OUTPUT = dict[Literal["loss", "losses", "network_output"], Any]

    def training_step(self, batch, batch_idx) -> TRAIN_STEP_OUTPUT:
        network_output = self.model(lidar=batch.get("lidar"), camera=batch.get("camera"), radar=batch.get("radar"))
        # Put the EGO velocity per sample into a list
        ego_velocities = [sample["velocity"] for sample in batch["metadata"]]
        # Evaluate the losses and sum
        losses = self.get_losses(output=network_output, targets=batch.get("targets"), ego_velocities=ego_velocities)
        losses["sum"] = self.sum_losses(losses)
        return {"loss": losses["sum"], "losses": losses, "network_output": network_output}

    def on_train_batch_end(self, outputs: TRAIN_STEP_OUTPUT, batch: Any, batch_idx: int):
        self._post_loss(step_output=outputs, batch=batch, mode="train")
        if (batch_idx % self.trainer.log_every_n_steps) == 0:  # NOQA
            ego_velocities = [sample["velocity"] for sample in batch["metadata"]]
            # Extract the detections from the network output and decode
            detections = self.get_detections(output=outputs["network_output"], ego_velocities=ego_velocities)
            nds = self.nds_train(detections, batch["targets"])
            self.log_dict(self.nds_train.log_dict(nds, mode="train"))
            del nds
            self.nds_train.reset()
            modified_frequency = self.logging.frequency_log_train_sample / self.trainer.log_every_n_steps  # NOQA
            self._post_pointcloud(batch, detections, logging_frequency=modified_frequency, prefix="Train")

    VAL_STEP_OUTPUT = dict[Literal["losses", "detections", "network"], Any]

    def validation_step(self, batch, batch_idx) -> VAL_STEP_OUTPUT:
        network_output = self.model(lidar=batch.get("lidar"), camera=batch.get("camera"), radar=batch.get("radar"))
        # Put the EGO velocity per sample into a list
        ego_velocities = [sample["velocity"] for sample in batch["metadata"]]
        # Evaluate the losses and sum
        losses = self.get_losses(output=network_output, targets=batch.get("targets"), ego_velocities=ego_velocities)
        losses["sum"] = self.sum_losses(losses)
        detections = self.get_detections(output=network_output, ego_velocities=ego_velocities)
        self.nds_val.update(batch_detections=detections, batch_targets=batch["targets"])
        return {"losses": losses, "detections": detections}

    def on_validation_batch_end(self, outputs: VAL_STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self._post_loss(step_output=outputs, batch=batch, mode="val")
        self._post_pointcloud(
            batch=batch,
            detections=outputs["detections"],
            logging_frequency=self.logging.frequency_log_val_sample,
            prefix="Val",
        )

    def on_validation_epoch_end(self) -> None:
        nds = self.nds_val.compute()
        self.log_dict(self.nds_val.log_dict(nds, mode="val"), on_epoch=True)
        self.nds_val.reset()

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self, *args, **kwargs):
        lr = 0.5e-2
        optimizer = optim.AdamW(params=self.parameters(), lr=lr, weight_decay=0.01, amsgrad=False)
        steps_per_epoch = ceil(len(self.trainer.train_dataloader) / self.trainer.accumulate_grad_batches)
        steps_total = self.trainer.estimated_stepping_batches
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
            #pct_start=0.15,
            #div_factor=2e1,
            #final_div_factor=1e5,
        )
        # n = 2
        # interval = ceil(self.trainer.max_epochs / (n + 1))
        # milestones = [i * interval for i in range(1, n + 1) if i < self.trainer.max_epochs]
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #   optimizer=optimizer,
        #   T_0=ceil(2.5*steps_per_epoch),
        #   T_mult=2,
        #   eta_min=1e-7)
        # lr_scheduler = optim.lr_scheduler.ConstantLR(
        #   optimizer=optimizer,
        #   total_iters=self.trainer.max_epochs,
        #   factor=1.0)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
