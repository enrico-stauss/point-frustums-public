import math

import torch

from point_frustums.models.backbones.frustum_encoder import (
    decorator_distance_to_mean_local,
    decorator_distance_to_min,
    decorator_relative_angle,
)


def test_distance_to_mean_local_rotates_into_the_frustum_frame():
    # Two frustums: one looking along +x (azimuth 0), one along +y (azimuth pi / 2); each holds two points that are
    #   1 m apart along the ray and 0.2 m apart perpendicular to it
    x = torch.tensor([10.0, 11.0, -0.1, 0.1])
    y = torch.tensor([0.1, -0.1, 10.0, 11.0])
    i_frustum = torch.tensor([0, 0, 1, 1])
    azimuth = torch.tensor([0.0, 0.0, math.pi / 2, math.pi / 2])
    counts = torch.tensor([2, 2])
    longitudinal = decorator_distance_to_mean_local(x, y, azimuth, 2, counts, i_frustum, "longitudinal")[:, 0]
    lateral = decorator_distance_to_mean_local(x, y, azimuth, 2, counts, i_frustum, "lateral")[:, 0]
    torch.testing.assert_close(longitudinal, torch.tensor([-0.5, 0.5, -0.5, 0.5]))
    # Lateral is positive to the left of the ray (counter-clockwise), in both frustums alike
    torch.testing.assert_close(lateral, torch.tensor([0.1, -0.1, 0.1, -0.1]))


def test_distance_to_min():
    radial = torch.tensor([5.0, 7.0, 6.0, 20.0, 21.0])
    i_frustum = torch.tensor([0, 0, 0, 2, 2])
    out = decorator_distance_to_min(radial, n_frustums=3, i_frustum=i_frustum, std=2.0)[:, 0]
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0, 0.5, 0.0, 0.5]))


def test_relative_angle_respects_the_start_of_the_field_of_view():
    # Frustums of 1.5 starting at 0.25: [0.25, 1.75), [1.75, 3.25); the centers are 1.0 and 2.5
    angle = torch.tensor([1.0, 2.5, 0.25, 3.2])
    out = decorator_relative_angle(angle, delta_rad=1.5, start_rad=0.25)[:, 0]
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, -0.5, 0.7 / 1.5]), atol=1e-5, rtol=0)
