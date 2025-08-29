from typing import Optional

import numpy as np
import torch
import wandb
from matplotlib import figure, pyplot, colormaps
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Arrow
from pytorch_lightning.loggers.wandb import WandbLogger

from point_frustums.config_dataclasses.dataset import Labels
from point_frustums.geometry.utils import get_corners_3d
from point_frustums.utils.custom_types import Boxes, Targets

COLORS = {
    "Blue": np.array([65, 47, 215]),
    "DarkBlue": np.array([22, 1, 43]),
    "Yellow": np.array([254, 239, 58]),
    "Red": np.array([231, 52, 50]),
    "Green": np.array([40, 238, 133]),
    "LightBlue": np.array([153, 204, 255]),
    "MintGreen": np.array([179, 255, 204]),
}


def get_bev_bbox(center: torch.Tensor, wlh: torch.Tensor, orientation: torch.Tensor) -> PatchCollection:
    device = center.device
    boxes_corners = (
        wlh[:, [1, 0]][:, None, :]
        .repeat(1, 4, 1)
        .mul(torch.tensor([[1, -1], [1, 1], [-1, 1], [-1, -1]], device=device, dtype=torch.float32)[None, ...])
        .mul(0.5)
    )

    assert orientation.ndim == 3 and orientation.size(1) == 3 and orientation.size(2) == 3
    rotation_matrices = orientation[:, :2, :2]

    # [N, 2, 2] x [N, 4, 2]
    boxes_corners = torch.einsum("ijk,ilk->ilj", rotation_matrices, boxes_corners)
    boxes_corners += center[:, None, [0, 1]]
    boxes_corners = boxes_corners.cpu()

    boxes_polygons = []
    for j in range(boxes_corners.size(0)):
        box_corners = boxes_corners[j, ...]
        boxes_polygons.append(Polygon(box_corners))

    return PatchCollection(boxes_polygons, alpha=0.6, linewidth=0.1)


def get_bev_velocity_arrows(velocity: torch.Tensor, center: torch.Tensor) -> PatchCollection:
    velocity, center = velocity.cpu(), center.cpu()
    velocities_arrows = []
    for i in range(velocity.size(0)):
        x, y = center[i, ..., [0, 1]].tolist()
        v_x, v_y = velocity[i, ..., [0, 1]].tolist()
        velocities_arrows.append(Arrow(x, y, v_x, v_y))

    return PatchCollection(velocities_arrows, alpha=1, linewidth=0.1)


def _white_grid(ax, x_label, y_label):
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.grid(alpha=0.1)
    ax.title.set_color("white")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.set_xlabel(x_label, c="white")
    ax.set_ylabel(y_label, c="white")
    return ax


def plot_pointcloud_bev(
    points: torch.Tensor,
    targets: Targets,
    ego_velocity: Optional[torch.Tensor] = None,
    detections: Optional[dict[str, torch.Tensor]] = None,
    figure_size: tuple[float, float] = (5.0, 5.0),
    plotting_range: int = 50,
    white_grid: bool = False,
) -> figure:
    fig, ax = pyplot.subplots(figsize=figure_size)

    bboxes_tg = get_bev_bbox(targets["center"], targets["wlh"], targets["orientation"])
    velocities_tg = get_bev_velocity_arrows(targets["velocity"], targets["center"])
    bboxes_tg.set(color=COLORS["Green"] / 255, alpha=0.8, edgecolor=COLORS["MintGreen"] / 255, linewidth=0.4)
    velocities_tg.set(color=COLORS["MintGreen"] / 255, alpha=1, linewidth=0.01)
    ax.add_collection(bboxes_tg)
    ax.add_collection(velocities_tg)

    mean_x, mean_y = points.mean(dim=0)[[0, 1]].tolist()
    ax.scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.6, color=COLORS["Yellow"] / 255, marker=".", lw=0)

    if ego_velocity is not None:
        v_x, v_y = ego_velocity[[0, 1]].tolist()
        ego_velocity_arrow = PatchCollection([Arrow(mean_x, mean_y, v_x, v_y)], alpha=1, linewidth=0.1)
        ego_velocity_arrow.set(color=COLORS["Blue"] / 255, alpha=0.8, linewidth=0.05)

    if detections is not None:
        bboxes_gt = get_bev_bbox(detections["center"], detections["wlh"], detections["orientation"])
        velocities_gt = get_bev_velocity_arrows(detections["velocity"], detections["center"])
        bboxes_gt.set(color=COLORS["Red"] / 255, alpha=0.4, edgecolor=COLORS["Red"] / 255, linewidth=0.2)
        velocities_gt.set(color=COLORS["Red"] / 255, alpha=0.6, linewidth=0.01)
        ax.add_collection(bboxes_gt)
        ax.add_collection(velocities_gt)

    ax.set_xlim((mean_x - plotting_range, mean_x + plotting_range))
    ax.set_ylim((mean_y - plotting_range, mean_y + plotting_range))

    x_label, y_label = "x [m]", "y [m]"
    if white_grid:
        _white_grid(ax, x_label=x_label, y_label=y_label)
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    return fig


def create_boxes_log(boxes: Boxes, color: tuple[int, int, int]) -> tuple[list[dict], list[dict]]:
    n_boxes = boxes["class"].numel()
    if n_boxes == 0:
        return [], []

    corners = get_corners_3d(boxes["center"], boxes["wlh"], boxes["orientation"]).tolist()
    labels = boxes["class"].tolist()
    centers = boxes["center"].tolist()

    scores = None
    if "scores" in boxes:
        scores = boxes["score"].tolist()

    velocities = torch.nan_to_num(boxes["velocity"])
    if velocities.size(1) == 2:
        velocities = torch.nn.functional.pad(velocities, (0, 1, 0, 0))
    vel_end = (boxes["center"] + velocities).tolist()

    box_list = []
    velocity_list = []

    for i in range(n_boxes):
        box = {"corners": corners[i], "label": labels[i], "color": color}
        if scores is not None:
            box["score"] = scores[i]
        box_list.append(box)
        velocity_list.append({"start": centers[i], "end": vel_end[i]})
    return box_list, velocity_list


def plot_pointcloud_wandb(
    wandb_logger: WandbLogger,
    points: torch.Tensor,
    targets: Targets,
    tag: str = "Pointcloud",
    step: Optional[int] = None,
    ego_velocity: Optional[torch.Tensor] = None,
    detections: Optional[dict[str, torch.Tensor]] = None,
    label_enum: Optional[Labels] = None,
    lower_z_threshold: float = 0.1,
):
    boxes = []
    velocities = []
    if targets is not None:
        gt_boxes, gt_velocities = create_boxes_log(targets, COLORS["MintGreen"].tolist())
        boxes.extend(gt_boxes)
        velocities.extend(gt_velocities)

    if detections is not None:
        det_boxes, det_velocities = create_boxes_log(detections, (COLORS["Red"]).tolist())
        boxes.extend(det_boxes)
        velocities.extend(det_velocities)

    if label_enum is not None:
        for box in boxes:
            box["label"] = label_enum.from_index(box["label"]).name
    else:
        for box in boxes:
            box["label"] = str(box["label"])

    if ego_velocity is not None:
        velocities.append({"start": [0, 0, 0], "end": ego_velocity.tolist()})

    boxes = np.array(boxes)
    velocities = np.array(velocities)
    mask = points[:, 2] >= lower_z_threshold
    points = points[mask, :]
    intensity_rgb = colormaps["plasma"](points[:, 3].numpy() / 255)[:, :3] * 255
    points = np.concatenate((points[:, :3].numpy(), intensity_rgb), axis=1)

    pointcloud = wandb.Object3D.from_point_cloud(
        point_cloud_type="lidar/beta", points=points, boxes=boxes, vectors=velocities
    )
    wandb_logger.experiment.log({tag: pointcloud, "trainer/global_step": step}, commit=False)


def render_target_assignment(
    target_corners: torch.Tensor, assigned_corners: torch.Tensor, target_index: torch.Tensor, wandb_logger
):
    targets_list = []
    for i, target in enumerate(target_corners.cpu().tolist()):
        targets_list.append({"corners": target, "label": i, "color": COLORS["MintGreen"].tolist()})

    assignment_list = []
    target_index = target_index.cpu().tolist()
    for i, assignment in enumerate(assigned_corners.cpu().tolist()):
        assignment_list.append({"corners": assignment, "label": str(target_index[i]), "color": COLORS["Red"].tolist()})
    pointcloud = wandb.Object3D(
        {"type": "lidar/beta", "points": np.array([[0.0, 0.0, 0.0]]), "boxes": np.array(targets_list + assignment_list)}
    )
    wandb_logger.experiment.log({"Pointcloud": pointcloud}, commit=True)
