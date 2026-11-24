from functools import partial
from math import pi
from unittest.mock import Mock

import torch

from point_frustums.config_dataclasses.fpn import ConfigFPNLayer
from point_frustums.config_dataclasses.point_frustums import ConfigDiscretize
from point_frustums.geometry.coordinate_system_conversion import cart_to_sph_torch, sph_to_cart_torch
from point_frustums.geometry.rotation_matrix import random_rotation_matrix
from point_frustums.models.point_frustums import PointFrustums


# pylint: disable=protected-access


class TestPointFrustumsCoders:
    n = 1000
    module_mock = Mock()
    module_mock.feature_vectors_angular_centers = torch.arange(0, 2 * n).reshape((n, 2)) / n
    module_mock.discretization.delta_pol = 0.01
    module_mock.discretization.delta_azi = 0.01

    def test_center_coder(self):
        points_cartesian: torch.Tensor = torch.rand((self.n, 3))
        points_spherical = cart_to_sph_torch(points_cartesian)
        points_encoded = PointFrustums._encode_center(self.module_mock, points_spherical)
        points_decoded = PointFrustums._decode_center(self.module_mock, points_encoded)
        assert torch.allclose(points_spherical, points_decoded, atol=1e-6)

    def test_velocity_coder(self):
        ego_velocity = torch.rand(1, 3).repeat((self.n, 1))
        velocities_cartesian: torch.Tensor = torch.rand((self.n, 3))
        velocities_encoded = PointFrustums._encode_velocity(self.module_mock, velocities_cartesian, ego_velocity)
        velocities_decoded = PointFrustums._decode_velocity(self.module_mock, velocities_encoded, ego_velocity)
        assert torch.allclose(velocities_cartesian[:, [0, 1]], velocities_decoded, atol=1e-6)

    def test_orientation_coder(self):
        rotation_matrices = random_rotation_matrix(self.n)
        orientation_encoded = PointFrustums._encode_orientation(self.module_mock, rotation_matrices)
        orientation_decoded = PointFrustums._decode_orientation(self.module_mock, orientation_encoded)
        assert torch.allclose(rotation_matrices, orientation_decoded, atol=1e-6)

    def test_wlh_coder(self):
        wlh = torch.rand((self.n, 3))
        wlh_encoded = PointFrustums._encode_wlh(wlh)
        wlh_decoded = PointFrustums._decode_wlh(wlh_encoded)
        assert torch.allclose(wlh, wlh_decoded, atol=1e-6)


class TestPointFrustumsLosses:
    # Instantiate the mocking object
    module_mock = Mock()
    # Define the config to use for the mocked PointFrustums model
    #   -> 2 layers with 2 blocks each and downsampling in both directions
    #   -> Heads with 4 convolutional layers
    #   -> Discretization azimuthal [-180, 180]: 8 splits and polar [60, 100]: 4 splits
    #   -> A stride of 2 for the first layer and 4 for the second layer
    layer_config_kwargs = {"n_channels": 1, "downsampling_horizontal": True, "downsampling_vertical": True}
    module_mock.fpn_layers = {
        "l0": ConfigFPNLayer(n_blocks=2, **layer_config_kwargs),
        "l1": ConfigFPNLayer(n_blocks=2, **layer_config_kwargs),
    }
    module_mock.fpn_extra_layers = {}
    module_mock.model.head.n_convolutions_classification = 4
    module_mock.model.head.n_convolutions_regression = 4
    module_mock.discretization = ConfigDiscretize(
        n_splits_pol=4, n_splits_azi=8, fov_azi_deg=(-180, 180), fov_pol_deg=(60, 100)
    )
    module_mock.strides = {"l0": (2, 2), "l1": (4, 4)}

    # Patch the properties to access the angular receptive field center coordinates
    module_mock._evaluate_receptive_fields = partial(PointFrustums._evaluate_receptive_fields, self=module_mock)
    module_mock.register_buffer = partial(setattr, module_mock)
    fm_params = PointFrustums._register_featuremap_parametrization(module_mock)
    module_mock.feature_vectors_angular_centers = getattr(module_mock, "feat_center_pol_azi")

    layer_sizes = fm_params.layer_sizes_flat
    angular_centers = module_mock.feature_vectors_angular_centers.split_with_sizes(layer_sizes, dim=0)

    module_mock._broadcast_targets = PointFrustums._broadcast_targets
    module_mock._encode_center = partial(PointFrustums._encode_center, module_mock)
    module_mock.losses.center_radial.kwargs = {}
    module_mock.losses.center_polar.kwargs = {}
    module_mock.losses.center_azimuthal.kwargs = {}

    def test_center_los(self):
        """
        Test that the center loss equals zero for a well-defined case of 3 targets, located in the upper-left and
        lower-right frustum of the PointFrustums mock.
        :return:
        """
        # Set the foreground_idx to decide where predictions were made
        #   -> the first index corresponds to the upper left of the pseudo image
        #   -> The input frustums are 45° (azimuthal) x 10° (polar) wide.
        #   -> Node k of a layer with stride s is centered on the input frustum s * k (the strided convolutions sample
        #      every s-th frustum), i.e. at the center of that frustum.
        #   -> l0 (stride 2, 4x2 nodes): index 0 is centered on the input frustum (0, 0) at [-7/8 pi, 65°] and index 7
        #      (row 1, column 3) on the input frustum (2, 6) at [5/8 pi, 85°].
        #   -> l1 (stride 4, 2x1 nodes): index 8 (the first of l1) is centered on the input frustum (0, 0) at
        #      [-7/8 pi, 65°].
        foreground_idx = torch.tensor([0, 7, 8], dtype=torch.long)
        # Create some sample predictions (relative to the frustum center) in the upper-left frustum for layer l0 and l1
        predictions = torch.tensor([[10, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=torch.float)
        # Create some sample targets (in absolute, cartesian coordinates)
        targets = sph_to_cart_torch(
            torch.tensor(
                [
                    [10, torch.deg2rad(torch.tensor(65)), -7 / 8 * pi],  # upper-left (l0)
                    [10, torch.deg2rad(torch.tensor(65)), -7 / 8 * pi],  # upper-left (l1)
                    [5, torch.deg2rad(torch.tensor(85)), 5 / 8 * pi],  # lower-right (l0)
                ],
                dtype=torch.float,
            )
        )
        # Set the mapping from target indices to foreground predictions
        targets_indices_foreground = torch.tensor([0, 2, 1], dtype=torch.long)
        # Compute the loss
        losses = PointFrustums._loss_center(
            self.module_mock,
            predictions_center_fg=predictions,
            foreground_idx=foreground_idx,
            targets_center=[targets],
            targets_indices_fg=[targets_indices_foreground],
        )
        zero = torch.tensor(0.0)
        assert torch.allclose(losses["center_radial"], zero)
        assert torch.allclose(losses["center_polar"], zero)
        assert torch.allclose(losses["center_azimuthal"], zero)

        # Second case, the predictions were inaccurate, we'd expect all losses to be nozero
        predictions = torch.tensor([[11, 0.5, 0], [4, 0, 0.2], [10, 1, 0]], dtype=torch.float)
        losses = PointFrustums._loss_center(
            self.module_mock,
            predictions_center_fg=predictions,
            foreground_idx=foreground_idx,
            targets_center=[targets],
            targets_indices_fg=[targets_indices_foreground],
        )
        assert not torch.isclose(losses["center_radial"], zero).any()
        assert not torch.isclose(losses["center_polar"], zero).any()
        assert not torch.isclose(losses["center_azimuthal"], zero).any()
