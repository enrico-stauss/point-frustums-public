import pytest
import torch

from point_frustums.augmentations import RandomFlipHorizontal, RandomRotate
from point_frustums.geometry.coordinate_system_conversion import cart_to_sph_torch
from point_frustums.geometry.rotation_matrix import rotation_matrix_from_axis_angle

# The lidar channels are [x, y, z, intensity, timestamp, radial, polar, azimuthal]
MAPPING = {"x": 0, "y": 1, "z": 2, "azimuthal": 7}


def make_sample(n_boxes: int = 6, n_points_per_box: int = 50):
    """Boxes with a random yaw and points sampled inside of them, plus points in the surrounding."""
    torch.manual_seed(0)
    center = torch.cat([torch.rand(n_boxes, 2) * 60 - 30, torch.rand(n_boxes, 1) * 2 - 1], dim=1)
    wlh = torch.rand(n_boxes, 3) * 3 + 0.5
    yaw = torch.rand(n_boxes) * 2 * torch.pi - torch.pi
    orientation = rotation_matrix_from_axis_angle(axis=torch.tensor([[0, 0, 1.0]]).expand(n_boxes, 3), angle=yaw)
    # Points inside the boxes: in the box frame (x: length, y: width, z: height), rotated and moved to the center
    local = (torch.rand(n_boxes, n_points_per_box, 3) - 0.5) * 0.98 * wlh[:, None, [1, 0, 2]]
    inside = torch.einsum("bij,bnj->bni", orientation, local) + center[:, None, :]
    xyz = torch.cat([inside.reshape(-1, 3), torch.rand(500, 3) * 80 - 40], dim=0)
    lidar = torch.cat([xyz, torch.rand(xyz.shape[0], 2), cart_to_sph_torch(xyz)], dim=1)
    targets = {
        "class": torch.zeros(n_boxes, dtype=torch.long),
        "center": center,
        "wlh": wlh,
        "orientation": orientation,
        "attribute": torch.zeros(n_boxes, dtype=torch.long),
        # A velocity equal to the center vector must be transformed like the center
        "velocity": center.clone(),
    }
    metadata = {"velocity": center[0].clone()}
    return lidar, targets, metadata


def points_in_boxes(xyz: torch.Tensor, targets: dict) -> torch.Tensor:
    """[N, B] mask of the points that are inside each box."""
    local = torch.einsum("bji,nbj->nbi", targets["orientation"], xyz[:, None, :] - targets["center"][None])
    half_size = targets["wlh"][:, [1, 0, 2]] / 2
    return (local.abs() <= half_size[None]).all(dim=-1)


def apply(augmentation, lidar, targets, metadata):
    if hasattr(augmentation, "refresh"):
        augmentation.probability = 1.0
        augmentation.refresh()
    lidar_out, _, _, targets_out, metadata_out = augmentation(
        lidar={"LIDAR_TOP": lidar.clone()},
        camera=None,
        radar=None,
        targets={k: v.clone() for k, v in targets.items()},
        metadata={k: v.clone() for k, v in metadata.items() if isinstance(v, torch.Tensor)},
    )
    return lidar_out["LIDAR_TOP"], targets_out, metadata_out


AUGMENTATIONS = {
    "flip": lambda: RandomFlipHorizontal(apply_to=["lidar", "targets"], dimension_mapping_lidar=MAPPING, probability=1),
    "rotate": lambda: RandomRotate(apply_to=["lidar", "targets"], dimension_mapping_lidar=MAPPING, probability=1),
}


@pytest.mark.parametrize("name", AUGMENTATIONS)
def test_points_stay_in_their_boxes(name: str):
    lidar, targets, metadata = make_sample()
    mask = points_in_boxes(lidar[:, :3], targets)
    assert mask[:300].any(dim=1).all()
    for _ in range(5):
        lidar_aug, targets_aug, _ = apply(AUGMENTATIONS[name](), lidar, targets, metadata)
        # Each point is inside the same boxes as before; the box sizes and the range of the points do not change
        assert torch.equal(points_in_boxes(lidar_aug[:, :3], targets_aug), mask)
        torch.testing.assert_close(targets_aug["wlh"], targets["wlh"])
        torch.testing.assert_close(lidar_aug[:, 2:7], lidar[:, 2:7])
        # The orientation remains a proper rotation about the z-axis
        rot = targets_aug["orientation"]
        torch.testing.assert_close(rot @ rot.transpose(1, 2), torch.eye(3).expand_as(rot), atol=1e-5, rtol=0)
        torch.testing.assert_close(torch.linalg.det(rot), torch.ones(len(rot)), atol=1e-5, rtol=0)
        torch.testing.assert_close(rot[:, 2, 2], torch.ones(len(rot)), atol=1e-5, rtol=0)


@pytest.mark.parametrize("name", AUGMENTATIONS)
def test_spherical_coordinates_and_velocities_are_consistent(name: str):
    lidar, targets, metadata = make_sample()
    lidar_aug, targets_aug, metadata_aug = apply(AUGMENTATIONS[name](), lidar, targets, metadata)
    # The spherical coordinates match the transformed cartesian ones
    torch.testing.assert_close(lidar_aug[:, 5:], cart_to_sph_torch(lidar_aug[:, :3]), atol=1e-4, rtol=0)
    # Object and ego velocity are transformed like position vectors
    torch.testing.assert_close(targets_aug["velocity"], targets_aug["center"], atol=1e-5, rtol=0)
    torch.testing.assert_close(metadata_aug["velocity"], targets_aug["center"][0], atol=1e-5, rtol=0)
    if name == "flip":
        torch.testing.assert_close(targets_aug["center"], targets["center"] * torch.tensor([1.0, -1.0, 1.0]))
        # Applied twice, the flip is the identity
        lidar_twice, targets_twice, _ = apply(AUGMENTATIONS[name](), lidar_aug, targets_aug, metadata_aug)
        torch.testing.assert_close(lidar_twice, lidar)
        torch.testing.assert_close(targets_twice["orientation"], targets["orientation"], atol=1e-5, rtol=0)
