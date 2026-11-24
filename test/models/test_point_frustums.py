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
        #   -> The angular coordinate with the discretization of 4 azimuthal and 2 polar splits (after a stride 2
        #      downsampling on the first layer) equals to an azimuthal position of [-3/4 pi, 70°].
        #   -> The angular coordinates with the discretization of 2 azimuthal and 1 polar split (after a stride 4
        #      downsampling on the second layer) equals to an azimuthal position of [-1/2 pi, 80°].
        #   -> The first layer outputs 8 predictions 4x2, so the first index of the second layer output equals 8.
        foreground_idx = torch.tensor([0, 7, 8], dtype=torch.long)
        # Create some sample predictions (relative to the frustum center) in the upper-left frustum for layer l0 and l1
        predictions = torch.tensor([[10, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=torch.float)
        # Create some sample targets (in absolute, cartesian coordinates)
        targets = sph_to_cart_torch(
            torch.tensor(
                [
                    [10, torch.deg2rad(torch.tensor(70)), -3 / 4 * pi],  # upper-left (l0)
                    [10, torch.deg2rad(torch.tensor(80)), -1 / 2 * pi],  # upper-left (l1)
                    [5, torch.deg2rad(torch.tensor(90)), 3 / 4 * pi],  # lower-right (l0)
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


class TestPointsInBoxAssignment:
    """The points_in_box target assignment: from the frustums observing a target to the feature vectors."""

    module_mock = Mock()
    layer_config_kwargs = {"n_channels": 1, "downsampling_horizontal": True, "downsampling_vertical": False}
    module_mock.fpn_layers = {
        "l0": ConfigFPNLayer(n_blocks=1, n_channels=1, downsampling_horizontal=False, downsampling_vertical=False),
        "l1": ConfigFPNLayer(n_blocks=1, **layer_config_kwargs),
    }
    module_mock.fpn_extra_layers = {}
    module_mock.model.head.n_convolutions_classification = 1
    module_mock.model.head.n_convolutions_regression = 1
    # 8 azimuthal x 4 polar frustums; l0 has stride 1 (4x8 nodes), l1 the stride (1, 2) (4x4 nodes)
    module_mock.discretization = ConfigDiscretize(
        n_splits_pol=4, n_splits_azi=8, fov_azi_deg=(-180, 180), fov_pol_deg=(60, 100)
    )
    module_mock.strides = {"l0": (1, 1), "l1": (1, 2)}
    module_mock._evaluate_receptive_fields = partial(PointFrustums._evaluate_receptive_fields, self=module_mock)
    module_mock.register_buffer = partial(setattr, module_mock)
    module_mock.featuremap_parametrization = PointFrustums._register_featuremap_parametrization(module_mock)
    module_mock.device = torch.device("cpu")
    module_mock.trainer.training = False
    module_mock.target_assignment.level_ranges = {"l0": (0.0, 15.0), "l1": (15.0, 80.0)}

    def assign(self, centers, frustum, target, radial):
        assignment = {
            "frustum": torch.tensor(frustum),
            "target": torch.tensor(target),
            "radial": torch.tensor(radial, dtype=torch.float),
            "discretization": torch.tensor([8, 4, -180, 180, 60, 100]),
        }
        return PointFrustums._assign_points_in_box(self.module_mock, {"center": torch.tensor(centers)}, assignment)

    def test_layers_by_range_and_closest_point(self):
        # Target 0 at 10 m -> l0 (stride 1): the frustums 9 (row 1, column 1) and 10 map to the l0 nodes 9 and 10
        # Target 1 at 20 m -> l1 (stride 2, offset 32): the frustum 20 (row 2, column 4) maps to the node
        #   32 + 2 * 4 + 2 = 42, the frustum 21 (column 5) to the closest node column round(2.5) = 2 as well; target 2
        #   (also at 20 m) has a closer point in the frustum 21 and wins the node 42
        index = self.assign(
            centers=[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
            frustum=[9, 10, 20, 21],
            target=[0, 0, 1, 2],
            radial=[9.0, 9.5, 19.0, 18.0],
        )
        assert index.shape == (32 + 16,)
        assert index[9] == 0 and index[10] == 0
        assert index[42] == 2
        assert (index >= 0).sum() == 3

    def test_azimuth_wraps_to_the_first_node(self):
        # The last column (7) is closest to the node column round(3.5) = 4 -> wraps to 0 on the 4 columns of l1
        index = self.assign(centers=[[20.0, 0.0, 0.0]], frustum=[7], target=[0], radial=[19.0])
        assert index[32 + 0] == 0
