import torch

from point_frustums.augmentations import AssignFrustumsToTargets, KeepSweeps
from point_frustums.config_dataclasses.point_frustums import ConfigDiscretize
from point_frustums.geometry.coordinate_system_conversion import cart_to_sph_torch


def test_keep_sweeps():
    points = torch.zeros(5, 8)
    points[:, 4] = torch.tensor([0.0, 0.049, 0.05, 0.12, 0.3])
    kept = KeepSweeps(apply_to=["lidar"], n_sweeps=1).lidar(points, metadata={})
    assert kept.shape[0] == 2
    kept = KeepSweeps(apply_to=["lidar"], n_sweeps=3).lidar(points, metadata={})
    assert kept.shape[0] == 4


def test_assign_frustums_to_targets():
    discretize = ConfigDiscretize(n_splits_azi=360, n_splits_pol=40, fov_azi_deg=(-180, 180), fov_pol_deg=(70, 110))
    # Two boxes along the x-axis (the second behind the first) and one to the left; length 4, width 2, height 2
    targets = {
        "center": torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
        "wlh": torch.tensor([[2.0, 4.0, 2.0]] * 3),
        "orientation": torch.eye(3).repeat(3, 1, 1),
    }
    xyz = torch.tensor(
        [
            [9.0, 0.1, 0.1],  # box 0
            [19.0, 0.1, 0.1],  # box 1, same frustum as the point of box 0 but further away
            [0.1, 10.5, 0.2],  # box 2
            [15.0, 0.1, 0.1],  # between the boxes: no target
        ]
    )
    sph = cart_to_sph_torch(xyz)
    points = torch.cat([xyz, torch.zeros(4, 2), sph], dim=1)
    augmentation = AssignFrustumsToTargets(discretize=discretize)
    _, _, _, _, metadata = augmentation(
        lidar={"LIDAR_TOP": points}, camera=None, radar=None, targets=targets, metadata={}
    )
    result = metadata["frustum_assignment"]
    i_azi = ((sph[:, 2] + torch.pi) // discretize.delta_azi).long()
    i_pol = ((sph[:, 1] - discretize.fov_pol[0]) // discretize.delta_pol).long()
    frustum = i_azi + i_pol * 360
    assert frustum[0] == frustum[1]
    assignment = dict(zip(result["frustum"].tolist(), result["target"].tolist()))
    # The closest point (of box 0) decides in the shared frustum; the point between the boxes is not assigned
    assert assignment == {int(frustum[0]): 0, int(frustum[2]): 2}
