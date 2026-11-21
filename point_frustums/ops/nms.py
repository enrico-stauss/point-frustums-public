import torch

from point_frustums.ops import metal


def nms_3d(
    labels: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    iou_threshold: float = 0.5,
    distance_threshold: float = 0.1,
) -> torch.Tensor:
    """

    :param labels:
    :param scores:
    :param boxes:
    :param iou_threshold:
    :param distance_threshold: Evaluated before the IoU to reduce the computational load
    :return: A boolean tensor where True indicates that the prediction should be kept
    """
    if labels.numel() == 0:
        return torch.empty_like(labels).bool()

    if labels.device.type == "mps":
        # The C++ extension only provides CPU and CUDA kernels, MPS uses the equivalent Metal kernels
        return metal.nms(labels, scores, boxes, iou_threshold, distance_threshold)

    return torch.ops.point_frustums.nms(labels.int(), scores, boxes, iou_threshold, distance_threshold).bool()
