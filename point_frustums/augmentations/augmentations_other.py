from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import cached_property
from math import ceil
from typing import Optional

import torch

from point_frustums.config_dataclasses.point_frustums import ConfigDiscretize
from point_frustums.utils.custom_types import Targets
from .augmentations import Augmentation


@dataclass(frozen=True)
class NormalizationParameters:
    channels: list[int]
    mean: list[float]
    std: list[float]

    @cached_property
    def tensor_mean(self) -> torch.Tensor:
        return torch.tensor(self.mean)

    @cached_property
    def tensor_std(self) -> torch.Tensor:
        return torch.tensor(self.std)

    @property
    def log(self):
        return {"mean": self.tensor_mean, "std": self.tensor_std, "channels": self.channels}


def de_normalize(data: torch.Tensor, channels: list[int], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data[:, channels] = data[:, channels].mul(std).add(mean)
    return data


def normalize(data: torch.Tensor, channels: list[int], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data[:, channels] = data[:, channels].sub(mean).div(std)
    return data


class Normalize(Augmentation):
    def __init__(
        self,
        lidar: Optional[NormalizationParameters] = None,
        camera: Optional[NormalizationParameters] = None,
        radar: Optional[NormalizationParameters] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._lidar = lidar
        self._camera = camera
        self._radar = radar
        for modality in self.apply_to:
            assert getattr(self, f"_{modality}") is not None

    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        metadata["augmentations"][str(self)] = {"camera": self._camera.log}
        return normalize(data, self._camera.channels, self._camera.tensor_mean, self._camera.tensor_std)

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        metadata["augmentations"][str(self)] = {"lidar": self._lidar.log}
        return normalize(data, self._lidar.channels, self._lidar.tensor_mean, self._lidar.tensor_std)

    def targets(self, targets: Targets):
        return targets

    def metadata(self, metadata: MutableMapping):
        return metadata


class SubsampleData(Augmentation):
    """
    Randomly subsample the data: drop a random fraction of the points, drawn uniformly from [0, drop_ratio] for each
    sample, so that the point density seen during training includes the one of the full point clouds seen during
    evaluation.
    """

    def __init__(self, drop_ratio, n_max=1_000_000, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= drop_ratio <= 1
        self.drop_ratio = drop_ratio
        self._n_max = n_max

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        """
        Remove a fraction of the points, the remaining points keep their order.
        The global torch RNG is used: the DataLoader seeds it differently for each worker and epoch, whereas a
        generator owned by the augmentation is copied into each worker with the same state, so each worker (and each
        epoch) would draw the same sequence.
        :param data:
        :param metadata:
        :return:
        """
        n_points = data.shape[0]
        keep_ratio = 1 - self.drop_ratio * torch.rand(()).item()
        n_keep = ceil(keep_ratio * min(n_points, self._n_max))
        indices = torch.randperm(n_points)[:n_keep].sort().values
        if metadata is not None:
            metadata["augmentations"][str(self)] = {"keep_ratio": keep_ratio}
        return data[indices, ...]

    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def targets(self, targets: Targets):
        raise SyntaxError("Subsampling should not be applied to targets.")

    def metadata(self, metadata: MutableMapping):
        return metadata


class KeepSweeps(Augmentation):
    """
    Keep only the points of the n most recent lidar sweeps (deterministic, apply before the normalization).

    The timestamp channel holds the time of each point w.r.t. the reference sweep: the rotation time within the sweep
    in [0, period) plus the age of the sweep (see NuScenes.add_timestamps), hence the points of the n most recent sweeps
    have a timestamp below n * period. With n = 1, the pointcloud is a single scan as used by range view detectors.
    """

    def __init__(self, n_sweeps: int = 1, timestamp_channel: int = 4, period: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        assert n_sweeps >= 1
        self._n_sweeps = n_sweeps
        self._timestamp_channel = timestamp_channel
        self._period = period

    def lidar(self, data: torch.Tensor, metadata: MutableMapping | None):
        return data[data[:, self._timestamp_channel] < self._n_sweeps * self._period]

    def camera(self, data: torch.Tensor, metadata: MutableMapping | None):
        raise NotImplementedError()

    def radar(self, data: torch.Tensor, metadata: MutableMapping | None):
        raise NotImplementedError()

    def targets(self, targets: Targets):
        return targets

    def metadata(self, metadata: MutableMapping):
        return metadata


class AssignFrustumsToTargets(Augmentation):
    """
    Determine the (full resolution) frustums that observe each target, for the points_in_box target assignment.

    A frustum observes a target if it contains a lidar point within the 3D box of the target. If a frustum contains
    points of several targets, the target of the closest point is kept (the surface that is visible in the frustum).
    The result is stored in the sample metadata, the points and targets are not modified:
        metadata["frustum_assignment"] = {
            "frustum": the frustum index (row-major, as in FrustumEncoder.get_frustum_index) [K],
            "target": the index of the target [K],
            "radial": the distance of the closest point of the target in the frustum [K],
            "discretization": [n_splits_azi, n_splits_pol, *fov_azi_deg, *fov_pol_deg] to check the consistency with
                the model,
        }

    Place it after the geometric augmentations (they change the frustums of points and targets alike) and before the
    normalization (the raw coordinates are required). The point-in-box relation itself is invariant to the rigid
    augmentations and to subsampling.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        discretize: ConfigDiscretize,
        sensor: str = "LIDAR_TOP",
        dimension_mapping_lidar: dict[str, int] | None = None,
        **kwargs,
    ):
        kwargs.setdefault("apply_to", ["lidar", "targets"])
        super().__init__(**kwargs)
        self.discretize = discretize
        self.sensor = sensor
        self.dims = dimension_mapping_lidar or {"x": 0, "y": 1, "z": 2, "radial": 5, "polar": 6, "azimuthal": 7}

    def assign(self, points: torch.Tensor, targets: Targets) -> dict[str, torch.Tensor]:
        d = self.discretize
        xyz = points[:, [self.dims["x"], self.dims["y"], self.dims["z"]]]
        polar, azimuthal = points[:, self.dims["polar"]], points[:, self.dims["azimuthal"]]
        radial = points[:, self.dims["radial"]]
        in_fov = (azimuthal >= d.fov_azi[0]) & (azimuthal < d.fov_azi[1])
        in_fov &= (polar >= d.fov_pol[0]) & (polar < d.fov_pol[1])
        i_azi = torch.div(azimuthal - d.fov_azi[0], d.delta_azi, rounding_mode="floor").long()
        i_pol = torch.div(polar - d.fov_pol[0], d.delta_pol, rounding_mode="floor").long()
        frustum = i_azi + i_pol * d.n_splits_azi

        frustums, target_indices, ranges = [], [], []
        for j, (center, wlh, orientation) in enumerate(zip(targets["center"], targets["wlh"], targets["orientation"])):
            # Coarse selection by the bounding sphere, then the exact test in the frame of the box (x: length,
            #   y: width, z: height as in get_corners_3d)
            half_size = wlh[[1, 0, 2]] / 2
            candidates = torch.nonzero(in_fov & ((xyz - center).norm(dim=-1) <= half_size.norm())).squeeze(-1)
            local = (xyz[candidates] - center) @ orientation
            inside = candidates[(local.abs() <= half_size).all(dim=-1)]
            frustums.append(frustum[inside])
            target_indices.append(torch.full_like(inside, j))
            ranges.append(radial[inside])

        discretization = torch.tensor([d.n_splits_azi, d.n_splits_pol, *d.fov_azi_deg, *d.fov_pol_deg])
        if len(frustums) == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return {"frustum": empty, "target": empty, "radial": empty.float(), "discretization": discretization}
        frustums, target_indices, ranges = torch.cat(frustums), torch.cat(target_indices), torch.cat(ranges)
        # Keep the closest point per frustum: sort by range and keep the first occurrence of each frustum
        order = torch.argsort(ranges)
        frustums, target_indices, ranges = frustums[order], target_indices[order], ranges[order]
        unique, inverse = torch.unique(frustums, return_inverse=True)
        first = torch.full_like(unique, len(frustums)).scatter_reduce(
            0, inverse, torch.arange(len(frustums)), reduce="amin"
        )
        return {
            "frustum": frustums[first],
            "target": target_indices[first],
            "radial": ranges[first].float(),
            "discretization": discretization,
        }

    def forward(self, lidar, camera, radar, targets, metadata):  # pylint: disable=too-many-arguments
        if metadata is None:
            metadata = {}
        if lidar is not None and targets is not None:
            metadata["frustum_assignment"] = self.assign(lidar[self.sensor], targets)
        return lidar, camera, radar, targets, metadata

    def lidar(self, data: torch.Tensor, metadata: MutableMapping | None):
        return data

    def camera(self, data: torch.Tensor, metadata: MutableMapping | None):
        return data

    def radar(self, data: torch.Tensor, metadata: MutableMapping | None):
        return data

    def targets(self, targets: Targets):
        return targets

    def metadata(self, metadata: MutableMapping):
        return metadata
