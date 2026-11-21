import pytest
import torch

import point_frustums  # noqa: F401  (registers torch.ops.point_frustums)
from point_frustums.geometry.rotation_matrix import rotation_matrix_from_axis_angle
from point_frustums.geometry.utils import get_corners_3d, iou_vol_3d
from point_frustums.ops import metal
from point_frustums.ops.nms import nms_3d

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")


def random_boxes(n: int, spread: float = 3.0, yaw_only: bool = False, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(n, 3, generator=generator) * spread
    wlh = torch.rand(n, 3, generator=generator) * 4 + 0.3
    if yaw_only:
        axis = torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3)
    else:
        axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=generator), dim=-1)
    angle = torch.rand(n, generator=generator) * 2 * torch.pi
    orientation = rotation_matrix_from_axis_angle(axis=axis, angle=angle)
    return get_corners_3d(centers, wlh, orientation).float().contiguous()


@pytest.mark.parametrize("yaw_only", [True, False])
def test_iou_box3d_matches_cpu(yaw_only: bool):
    boxes1, boxes2 = random_boxes(50, yaw_only=yaw_only, seed=1), random_boxes(70, yaw_only=yaw_only, seed=2)
    vol_cpu, iou_cpu = torch.ops.point_frustums.iou_box3d(boxes1, boxes2)
    vol_mps, iou_mps = metal.iou_box3d(boxes1.to("mps"), boxes2.to("mps"))
    # Same algorithm with explicit fma on both backends
    torch.testing.assert_close(vol_mps.cpu(), vol_cpu, rtol=0, atol=1e-6)
    torch.testing.assert_close(iou_mps.cpu(), iou_cpu, rtol=0, atol=1e-6)


def test_iou_box3d_pairwise_matches_cpu():
    boxes = random_boxes(100, seed=3)
    # Identical boxes, boxes sharing a face and random pairs
    shifted = boxes + (boxes[:, 1] - boxes[:, 0])[:, None, :]
    boxes1 = torch.cat([boxes, boxes, boxes]).contiguous()
    boxes2 = torch.cat([boxes, shifted, random_boxes(100, seed=4)]).contiguous()
    vol_cpu, iou_cpu = torch.ops.point_frustums.iou_box3d_pairwise(boxes1, boxes2)
    vol_mps, iou_mps = metal.iou_box3d_pairwise(boxes1.to("mps"), boxes2.to("mps"))
    torch.testing.assert_close(vol_mps.cpu(), vol_cpu, rtol=0, atol=1e-6)
    torch.testing.assert_close(iou_mps.cpu(), iou_cpu, rtol=0, atol=1e-6)
    torch.testing.assert_close(iou_mps[:100].cpu(), torch.ones(100), rtol=0, atol=1e-5)
    torch.testing.assert_close(iou_mps[100:200].cpu(), torch.zeros(100), rtol=0, atol=1e-5)


def test_iou_vol_3d_dispatches_to_metal():
    boxes1, boxes2 = random_boxes(20, seed=5), random_boxes(30, seed=6)
    iou_cpu, _ = iou_vol_3d(boxes1[:, None], boxes2[None, :])
    iou_mps, _ = iou_vol_3d(boxes1[:, None].to("mps"), boxes2[None, :].to("mps"))
    assert iou_mps.device.type == "mps"
    torch.testing.assert_close(iou_mps.cpu(), iou_cpu, rtol=0, atol=1e-6)


@pytest.mark.parametrize("distance_threshold", [1e9, 0.5])
def test_nms_matches_cpu(distance_threshold: float):
    generator = torch.Generator().manual_seed(7)
    boxes = random_boxes(400, spread=10.0, yaw_only=True, seed=7)
    # Add near-duplicates of the first half
    boxes[200:] = boxes[:200] + torch.randn(200, 1, 3, generator=generator) * 0.3
    labels = torch.randint(0, 3, (400,), dtype=torch.int32, generator=generator)
    labels[200:] = labels[:200]
    scores = torch.rand(400, generator=generator)
    keep_cpu = nms_3d(labels, scores, boxes, 0.4, distance_threshold)
    keep_mps = nms_3d(labels.to("mps"), scores.to("mps"), boxes.to("mps"), 0.4, distance_threshold)
    assert keep_mps.device.type == "mps"
    assert torch.equal(keep_mps.cpu(), keep_cpu)
    assert keep_cpu.sum() < 400
