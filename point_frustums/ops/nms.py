import torch


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

    mps = False
    if labels.device.type == "mps":
        mps = True
        labels, scores, boxes = labels.int().cpu(), scores.cpu(), boxes.cpu()
    result = torch.ops.point_frustums.nms(labels.int(), scores, boxes, iou_threshold, distance_threshold).bool()

    if mps:
        result = result.to("mps")

    return result
