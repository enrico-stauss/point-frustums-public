import torch

from point_frustums.augmentations import KeepSweeps


def test_keep_sweeps():
    points = torch.zeros(5, 8)
    points[:, 4] = torch.tensor([0.0, 0.049, 0.05, 0.12, 0.3])
    kept = KeepSweeps(apply_to=["lidar"], n_sweeps=1).lidar(points, metadata={})
    assert kept.shape[0] == 2
    kept = KeepSweeps(apply_to=["lidar"], n_sweeps=3).lidar(points, metadata={})
    assert kept.shape[0] == 4
