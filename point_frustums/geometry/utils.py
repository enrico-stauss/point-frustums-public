import numpy as np
import torch

from point_frustums.ops import metal
from .coordinate_system_conversion import cart_to_sph_torch


def deg_to_rad(deg):
    """
    Uses np.pi but works just fine with torch input
    :param deg:
    :return:
    """
    return (deg / 360) * 2 * np.pi


def rad_to_deg(rad):
    """
    Uses np.pi but works just fine with torch input
    :param rad:
    :return:
    """
    return (rad / (2 * np.pi)) * 360


def cart_to_sph_numpy(points: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    coos_sph = np.empty(points.shape, dtype=np.float32)
    xy = points[:, 0] ** 2 + points[:, 1] ** 2  # squared distance to observer in ground projection
    coos_sph[:, 0] = np.sqrt(xy + points[:, 2] ** 2)  # Radial distance
    coos_sph[:, 1] = np.arctan2(np.sqrt(xy), points[:, 2])  # Polar/inclination angle (defined from Z-axis down)
    coos_sph[:, 2] = np.arctan2(points[:, 1], points[:, 0])  # Azimuth angle
    return coos_sph


@torch.jit.script
def filter_fov_spherical(
    points: torch.Tensor,
    index_mapping: dict[str, int],
    fov_azimuthal: tuple[float, float],
    fov_polar: tuple[float, float],
):
    i_az, i_pol = index_mapping["azimuthal"], index_mapping["polar"]
    mask = (
        (points[:, i_az] >= fov_azimuthal[0])
        & (points[:, i_az] < fov_azimuthal[1])
        & (points[:, i_pol] >= fov_polar[0])
        & (points[:, i_pol] < fov_polar[1])
    )
    return points[mask, :]


def angle_to_neg_pi_to_pi(angle: torch.Tensor):
    """
    Shift all angles to be in the range (-pi, pi].
    :param angle:
    :return:
    """
    # Get the integer multiplier required to shift each angle to (-pi, pi]
    shift = torch.ceil(angle / (2 * torch.pi) - 0.5)
    return angle - shift * 2 * torch.pi


@torch.jit.script
def get_corners_3d(centers: torch.Tensor, wlh: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    """
    The corner ordering is/has to be compliant with the specification in pytorch3d.ops.box3d_overlap
    Calculate the 3D box corners. The coordinate system is aligned as:
        x -> forward
        y -> left
        z -> up
    :param centers: Cartesian coordinates of the centers [N, 3]
    :param wlh: Boxes width, length and height [N, 3]
    :param orientation: Orientation matrices [N, 3, 3]
    :return: Cartesian coordinates of the 8 corners [N, 8, 3]
    """
    unit_box_corners = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        device=centers.device,
        dtype=centers.dtype,
    )
    unit_box_corners -= 0.5
    # Scale each box; Take care that sizes are given in the format wlh and the unit box is xyz
    corners = (
        unit_box_corners * wlh.index_select(dim=-1, index=torch.tensor([1, 0, 2], device=centers.device))[..., None, :]
    )
    # Rotate each box individually -> (N,8,3):
    #   - keeps N boxes
    #   - contracts last dimension of rotation matrix with the 3 coordinate values of each point
    #   - produces M new corners in 3D
    # [N, 3, 3] x [N, 8, 3]
    # Shift all boxes to align with their respective center
    return torch.einsum("...jk,...lk->...lj", orientation, corners) + centers[..., None, :]


@torch.jit.script
def get_featuremap_projection_boundaries(
    centers: torch.Tensor,
    wlh: torch.Tensor,
    orientation: torch.Tensor,
    fov_pol: tuple[float, float],
    fov_azi: tuple[float, float],
    delta_pol: float,
    delta_azi: float,
) -> torch.Tensor:
    """
    Project the 3D box onto the 2D featuremap and return the 4 corners of the encapsulating 2D upright bounding box.
    :param centers:
    :param wlh:
    :param orientation:
    :param fov_pol:
    :param fov_azi:
    :param delta_pol:
    :param delta_azi:
    :return: torch.tensor([x1, y1, x2, y2])
    """
    # Get 8 corner points of 3D boxes [N, 8, 3]
    corners = get_corners_3d(centers=centers, wlh=wlh, orientation=orientation)
    # Transform cartesian corner coordinates to spherical coordinates
    corners = cart_to_sph_torch(corners)
    # Get the 3D box centers in spherical coordinates
    centers = cart_to_sph_torch(centers)

    # The below is required to rectify targets overlapping the angular ambiguity at +-pi
    # Given a target that has corners with negative and positive azimuthal angle, or speaking in terms of the
    #   featuremap some corners on the very left and some on the very right side.
    # -> Most likely the target does not span almost 360° but rather just overlaps the angular ambiguity.

    # Create a mask that indicates corners corresponding to such targets IFF the center is at least pi/2 from
    #   the center of the featuremap located at 0°.
    mask = ~(centers[:, None, 2].sign() == corners[:, :, 2].sign()) & (centers[:, None, 2].abs() > torch.pi / 2)
    fold_over = torch.nonzero(mask).unbind(-1)
    # Those masked corners are now either on the left or right side of the featuremap and have the opposite sign
    #   of the corresponding center. Now we need to shift them by +- 2 pi so that they exit the featuremap on
    #   the other side but have the same sign as the center angle.
    # This is done by adding SIGN(center) * 2 * pi to the angle:
    #   - If the center is on the left side (has negative angle) then the corners from the right side will be
    #       subtracted 2 pi and thus shifted to the left
    #   - If the center is on the right side (has positive angle) then the corners from the left side will be
    #       added 2 pi and thus shifted to the right
    corners[fold_over[0], fold_over[1], 2] += centers[fold_over[0], 2].sign() * torch.pi * 2
    # TODO: Make sure polar/vertical position is not flipped either

    # Now all corners are shifted s.t. the angular range is not [-pi, pi] but instead [0, 2 * pi] and equivalently
    #   for the polar (not problematic) angle
    corners[:, :, 1] -= fov_pol[0]
    corners[:, :, 2] -= fov_azi[0]

    # Get the minimum and maximum azimuthal/polar coordinate values per 3D box corner (note the dimension reorder)
    min_vals = corners[:, :, [2, 1]].min(dim=1).values
    max_vals = corners[:, :, [2, 1]].max(dim=1).values

    # Stack min/max in a [x, y, x, y] fashion
    boundaries = torch.cat((min_vals, max_vals), dim=1)

    # Compute box coordinates
    boundaries[:, [0, 2]] = boundaries[:, [0, 2]] / delta_azi
    boundaries[:, [1, 3]] = boundaries[:, [1, 3]] / delta_pol
    return boundaries


# @torch.jit.script
def iou_vol_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D IoU between 2 sets of input boxes defined by 8 corner points. If the input is of shape
    boxes1.shape==boxes2.shape==[N, 8, 3] computes the pairwise IoU of shape [N], else if boxes1.shape==[N, 1, 8, 3] and
    boxes2.shape==[1, M, 8, 3] broadcasts to the binary IoU of shape [N, M].
    :param boxes1:
    :param boxes2:
    :return: The IoU and intersecting volume between the input boxes
    """
    if boxes1.device.type == "mps":
        # The C++ extension only provides CPU and CUDA kernels, MPS uses the equivalent Metal kernels
        iou_box3d, iou_box3d_pairwise = metal.iou_box3d, metal.iou_box3d_pairwise
    else:
        iou_box3d, iou_box3d_pairwise = torch.ops.point_frustums.iou_box3d, torch.ops.point_frustums.iou_box3d_pairwise
    if boxes1.shape == boxes2.shape and boxes1.dim() == 3:
        vol, iou = iou_box3d_pairwise(boxes1, boxes2)
    elif boxes1.size(1) == 1 and boxes2.size(0) == 1 and boxes1.dim() == 4:
        boxes1, boxes2 = boxes1[:, 0, :, :], boxes2[0, :, :, :]
        vol, iou = iou_box3d(boxes1, boxes2)
    else:
        raise ValueError(
            f"Invalid input shapes {boxes1.shape}, {boxes2.shape}; please provide either both as "
            f"[N, 8, 3] or boxes1.shape==[N,1,8,3] and boxes2.shape==[1,M,8,3]."
        )
    iou = iou.nan_to_num(neginf=0.0, posinf=0.0).clamp(min=0.0, max=1.0).detach()

    return iou, vol
