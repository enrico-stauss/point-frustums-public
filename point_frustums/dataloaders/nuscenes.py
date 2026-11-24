import os
from enum import Enum
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache, cached_property
from functools import partial
from typing import Literal, Optional, NamedTuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from nuscenes import NuScenes as NuScDB
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from pytorch_lightning import LightningDataModule
from torch.utils._pytree import tree_map_only
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.io import read_image

from litdata import StreamingDataset, StreamingDataLoader, optimize

from point_frustums import ROOT_DIR
from point_frustums.augmentations import Augmentation, RandomAugmentation
from point_frustums.config_dataclasses.dataset import DatasetConfig, Sensor
from point_frustums.dataloaders.sampling import (
    IndexedStreamingDataset,
    SamplingConfig,
    count_targets_streaming,
    num_streaming_items,
    weighted_sampler,
)
from point_frustums.geometry.boxes import transform_boxes
from point_frustums.geometry.coordinate_system_conversion import cart_to_sph_torch
from point_frustums.geometry.quaternion import quaternion_to_rotation_matrix
from point_frustums.geometry.utils import cart_to_sph_numpy
from point_frustums.utils.custom_types import Targets
from point_frustums.utils import storage
from point_frustums.utils.environment_helpers import data_root_getter


def compact_lidar(pc: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    The storage format of a point cloud [N, 8] with the channels (x, y, z, intensity, timestamp, radial, polar,
    azimuthal): the cartesian coordinates in float32, the intensity (integers in [0, 255], exact) and the timestamp
    (< 1s, resolution <= 0.5ms) in float16; the spherical coordinates are not stored but recomputed on loading
    (expand_lidar). 16 instead of 32 bytes per point.
    :param pc: The point cloud [N, 8]
    :return:
    """
    assert pc.dim() == 2 and pc.shape[1] == 8, f"Expected a point cloud with 8 channels, got {tuple(pc.shape)}"
    return {"xyz": pc[:, :3].to(torch.float32).contiguous(), "intensity_timestamp": pc[:, 3:5].to(torch.float16)}


def expand_lidar(pc: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    """
    Restore the point cloud [N, 8] (x, y, z, intensity, timestamp, radial, polar, azimuthal) from the storage format
    (compact_lidar); point clouds that were stored in full are returned unchanged.
    :param pc:
    :return:
    """
    if isinstance(pc, torch.Tensor):
        return pc
    xyz = pc["xyz"].to(torch.float32)
    return torch.cat((xyz, pc["intensity_timestamp"].to(torch.float32), cart_to_sph_torch(xyz)), dim=1)


def _get_item_compact(dataset: Dataset, i: int) -> MutableMapping:
    """Load sample i of the dataset with the point clouds in the storage format (for the litdata optimization)."""
    data = dataset[i]
    if data.get("lidar") is not None:
        data["lidar"] = {sensor: compact_lidar(pc) for sensor, pc in data["lidar"].items()}
    return data


VISIBILITY_LEVELS = {
    "v0-40": 40,
    "v40-60": 60,
    "v60-80": 80,
    "v80-100": 100,
}
EMPTY_TARGET_TENSOR_OPTIONS: dict[str, dict] = {
    "class": {"size": (0,), "dtype": torch.int64},
    "attribute": {"size": (0,), "dtype": torch.int64},
    "center": {"size": (0, 3), "dtype": torch.float32},
    "wlh": {"size": (0, 3), "dtype": torch.float32},
    "orientation": {"size": (0, 4), "dtype": torch.float32},
    "velocity": {"size": (0, 3), "dtype": torch.float32},
}


class Sample(NamedTuple):
    lidar: Optional[dict[str, torch.Tensor]] = None
    camera: Optional[dict[str, torch.Tensor]] = None
    radar: Optional[dict[str, torch.Tensor]] = None
    targets: Optional[Targets] = None
    metadata: Optional[dict] = None

    def as_dict(self):
        return self._asdict()  # pylint: disable=no-member


class NuScenes(Dataset):
    def __init__(
        self,
        db: NuScDB,
        dataset: DatasetConfig,
        split: Literal["train", "test", "val"],
        load_annotations: bool = True,
        augmentations: Optional[list[Augmentation]] = None,
    ):
        """

        :param db: An instance of the nuscenes-devkit NuScenes class.
        :param dataset: The configuration object of the dataset.
        :param split: The dataset split to use, one of ['train', 'test', 'val'].
        :param augmentations: A dictionary of augmentations to apply to the data.
        """
        self.dataset = dataset
        self.db = db

        self.split = split
        self.load_annotations = load_annotations
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, i: int) -> MutableMapping:
        """
        Gets the data, targets and metadata to the given sample index i.
        ------------------------------------------------------------
        > lidar:
        >   SENSOR_ID: torch.Tensor | np.ndarray
        > camera:
        >   SENSOR_ID: torch.Tensor | np.ndarray
        > metadata:
        >   SENSOR_ID:
        >       modality: str
        >       rotation: List  # in quaternions
        >       translation: List  # xyz
        >   ...
        >   location: str
        >   sample_token: str
        >   scene_description: str
        >   scene_name: str
        >   timestamp: int  # unix timestamp in microseconds
        >   vehicle: str
        >   velocity: float  # in m/s
        > targets: point_frustums.utils.targets.Targets
        >   ...
        ------------------------------------------------------------
        :param i: The sample index to load.
        :return:
        """
        # Fetch the sample
        sample_token = self.sample_tokens[i]
        sample = self.db.get("sample", sample_token)
        # Fetch the scene
        scene_token = sample["scene_token"]
        scene = self.db.get("scene", scene_token)

        # Initialize the return dict and add the sensor independent metadata
        data = Sample().as_dict()
        data["metadata"] = {
            "sample_token": sample_token,
            "sample_index": i,
            **self.get_sample_meta(scene=scene, sample=sample),
        }

        # Get all sensor dependent (meta-)data
        for sensor, specifications in self.dataset.sensors.items():
            if not specifications.active:
                continue
            if data[specifications.modality] is None:
                data[specifications.modality] = {}
            data[specifications.modality][sensor] = self.get_data(specifications.modality, sample=sample, sensor=sensor)
            data["metadata"][sensor] = self.get_sensor_meta(data_token=sample["data"][sensor])

        # Load the targets in the COOS specified for the annotations
        if self.load_annotations:
            targets, metadata = self.get_targets(sample=sample, sensor=self.dataset.annotations.coos)
            data["targets"] = targets
            data["metadata"].update(metadata)

        # Apply augmentation to data and targets
        if self.augmentations is not None:
            with torch.inference_mode():
                data = self.apply_augmentations(Sample(**data)).as_dict()
        return data

    def apply_augmentations(self, sample: Sample) -> Sample:
        for augmentation in self.augmentations:
            if isinstance(augmentation, RandomAugmentation):
                augmentation.refresh()
            sample = augmentation(*sample)
        return Sample(*sample)

    @cached_property
    def can_bus_api(self) -> NuScenesCanBus:
        return NuScenesCanBus(dataroot=self.db.dataroot)

    @cached_property
    def sample_tokens(self):
        sample_tokens = []
        for scene in self.scenes:
            sample_tokens.extend(self.yield_samples_in_scene(scene["first_sample_token"]))
        return sample_tokens

    @cached_property
    def scenes(self):
        train_scenes, val_scenes, test_scenes = self.get_splits()
        if self.split == "train":
            scenes = self.get_scenes(train_scenes)
        elif self.split == "test":
            assert test_scenes is not None, "NuScenes Mini does not provide a test set!"
            scenes = self.get_scenes(test_scenes)
        elif self.split == "val":
            scenes = self.get_scenes(val_scenes)
        else:
            raise AttributeError
        return scenes

    def required_files(self, i: int) -> list[str]:
        """
        The sensor data files that loading sample i reads (all active sensors, including the previous lidar sweeps).
        :param i: The sample index
        :return:
        """
        sample = self.db.get("sample", self.sample_tokens[i])
        files = []
        for sensor, specifications in self.dataset.sensors.items():
            if not specifications.active:
                continue
            sample_data_token = sample["data"][sensor]
            n_files = specifications.sweeps if specifications.modality == "lidar" else 1
            for _ in range(n_files):
                files.append(self.db.get_sample_data_path(sample_data_token))
                sample_data_token = self.db.get("sample_data", sample_data_token)["prev"]
                if sample_data_token == "":
                    break
        return files

    def get_data(self, modality: str, sample: Mapping, sensor: str) -> torch.Tensor:
        if modality == "camera":
            data_path = self.db.get_sample_data_path(sample["data"][sensor])
            data = read_image(data_path).to(torch.float32)
        elif modality == "lidar":
            data = self.load_sweeps(sample=sample, sensor_id=sensor).to(torch.float32)
        else:
            raise NotImplementedError("For now, only 'camera' and 'lidar' data is supported.")
        return data

    @staticmethod
    def _translation_rotation(pose: dict, invert: bool = False) -> dict[Literal["rotation", "translation"], list]:
        translation = pose["translation"]
        rotation = pose["rotation"]

        if invert:
            translation = [-x for x in translation]
            rotation = [x if i == 0 else -x for i, x in enumerate(rotation)]

        return {"translation": translation, "rotation": rotation}

    def relevant_label(self, sample_annotation: Mapping) -> Optional[Enum]:
        """
        Return the class label of an annotation if it is to be used as target, else None. This is the single source of
        truth for which annotations become targets (used by get_targets and by the class counts for weighted sampling).
        :param sample_annotation: A record of the sample_annotation table
        :return:
        """
        visibility = self.db.get("visibility", sample_annotation["visibility_token"])["level"]
        if VISIBILITY_LEVELS[visibility] <= self.dataset.annotations.visibility:
            # The visibility of the object is below the threshold
            return None
        # None if the class is not in the list of classes to be considered
        return self.dataset.annotations.retrieve_class(sample_annotation["category_name"])

    def count_targets(self) -> torch.Tensor:
        """
        Count the targets per class in each sample without loading any sensor data.
        :return: Tensor of shape [N_samples, N_classes] in the order of self.sample_tokens
        """
        counts = torch.zeros((len(self), self.dataset.annotations.n_classes), dtype=torch.int64)
        for i, sample_token in enumerate(self.sample_tokens):
            for annotation_token in self.db.get("sample", sample_token)["anns"]:
                label = self.relevant_label(self.db.get("sample_annotation", annotation_token))
                if label is not None:
                    counts[i, label.value] += 1
        return counts

    def get_targets(  # pylint: disable=too-many-locals
        self,
        sample: Mapping,
        sensor: str,
    ) -> tuple[Targets, dict[str, list]]:
        """
        Load the targets in the COOS specified for the annotations and return them together with the transformations
        stored in the metadata.
        :param sample:
        :param sensor:
        :return:
        """
        if sensor == "EGO":
            # Use LIDAR_TOP as proxy sensor if annotations are to be loaded in EGO coos. Required to use the devkit API.
            sensor = "LIDAR_TOP"
        data_token = sample["data"][sensor]
        # Retrieve sensor & pose records
        sample_data = self.db.get("sample_data", data_token)
        ego = self.db.get("ego_pose", sample_data["ego_pose_token"])
        sensor = self.db.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

        # Add ego pose information to meta for later mapping back to global COOS
        metadata: dict[str, list] = {"translation": ego["translation"], "rotation": ego["rotation"]}

        # Make list of Box objects including coord system transforms.
        boxes_stacked = {"wlh": [], "center": [], "orientation": [], "velocity": [], "class": [], "attribute": []}

        for box in self.db.get_boxes(data_token):
            # sample["anns"] contains the same box.tokens as returned by get_boxes but the latter also interpolates
            # boxes for intermediate frames.
            sample_annotation = self.db.get("sample_annotation", box.token)

            label = self.relevant_label(sample_annotation)
            if label is None:
                continue

            # Add velocity information
            box.velocity = self.db.box_velocity(box.token)

            # Get the attributes
            if len(sample_annotation["attribute_tokens"]) == 0:
                # The object has no attributes (traffic cone, barrier, etc.)
                attribute = "void"
            else:
                # The object has exactly one attribute (pedestrian, car, etc.)
                (attribute,) = sample_annotation["attribute_tokens"]
                # Get the raw attribute name without the class type prefix
                attribute = self.db.get("attribute", attribute)["name"].split(".")[-1]
            # Retrieve the relevant row from the attributes table to access the integer label
            attribute = self.dataset.annotations.attributes.from_name(attribute)

            boxes_stacked["class"].append(label.value)
            boxes_stacked["wlh"].append(box.wlh)
            boxes_stacked["center"].append(box.center)
            boxes_stacked["orientation"].append(box.orientation.q)
            boxes_stacked["attribute"].append(attribute.value)
            boxes_stacked["velocity"].append(box.velocity)

        for key, value in boxes_stacked.items():
            if len(value) == 0:
                boxes_stacked[key] = torch.zeros(**EMPTY_TARGET_TENSOR_OPTIONS[key])
            elif key in ("class", "attribute"):
                boxes_stacked[key] = torch.tensor(value)
            else:
                # Convert float64 to float32
                dtype = dtype if (dtype := value[0].dtype) != np.float64 else np.float32
                # Stack and convert to torch.Tensor
                boxes_stacked[key] = torch.from_numpy(np.stack(value, axis=0, dtype=dtype))

        # Transform global -> EGO coos (parameters are in the EGO coos; reference: nuscenes.nuscenes.get_sample_data)
        boxes_stacked = transform_boxes(
            boxes_stacked, **self._translation_rotation(ego, invert=True), rotate_first=False
        )

        if self.dataset.annotations.coos != "EGO":
            # Transform from the EGO to the sensor coos (transformation parameters are in the sensor coos)
            boxes_stacked = transform_boxes(
                boxes_stacked, **self._translation_rotation(sensor, invert=True), rotate_first=False
            )

        # Convert orientation to matrix representation
        boxes_stacked["orientation"] = quaternion_to_rotation_matrix(boxes_stacked["orientation"])

        return boxes_stacked, metadata

    @lru_cache(maxsize=128)
    def load_pose_from_can(self, scene_name: str, message: str = "pose") -> Sequence[pd.DataFrame, pd.DatetimeIndex]:
        """
        Access the CAN data for the specified scene.
        Make use of caching as the same scene will be accessed over and over again but will always return the same data.
        :param scene_name:
        :param message:
        :return:
        """
        pose, indexer_timestamp = None, None
        try:
            # Load the pose data for the current scene from the CAN bus API
            pose = self.can_bus_api.get_messages(scene_name, message, print_warnings=False)
            pose = pd.DataFrame.from_records(pose)
            # Create a datetime index from the CAN log timestamps
            indexer_timestamp = pd.to_datetime(
                pose["utime"].array,
                utc=True,
                unit="us",
                origin="unix",
            )
        except KeyError as err:
            # The CAN Bus API returns an empty list for some scenes (e.g. scene-0061) which will produce an error when
            # trying to access the 'utime' key of the DataFrame
            assert "KeyError: 'utime'" == str(err)
        except Exception as err:  # pylint: disable=locally-disabled, broad-exception-caught
            # The CAN Bus API raises base exception if data is missing
            assert f"Error: {scene_name} does not have any CAN bus data!" == str(err)

        return pose, indexer_timestamp

    @staticmethod
    def interpolate_velocity(current_pose, next_pose):
        time = (next_pose["timestamp"] - current_pose["timestamp"]) * 1e-6
        translation = np.array(next_pose["translation"]) - np.array(current_pose["translation"])
        return translation / time

    def get_ego_velocity(self, scene, sample, proxy_sensor="LIDAR_TOP") -> np.ndarray:
        """
        Load the velocity of the EGO vehicle.
        If enabled, try to retrieve the velocity from the CAN data.
        If the latter fails or is disabled, fallback to interpolating the velocity from the EGO translation.
            -> First option: use the translation of the previous sample
            -> Second option: use the translation of the next sample
        If everything of the above fails, return an unphysical downward velocity of 1m/s.
        :param scene:
        :param sample:
        :param proxy_sensor: An arbitrary sensor, used to retrieve 2 consecutive samples if the velocity cannot be
            loaded from CAN data.
        :return:
        """
        velocity = None

        if self.dataset.load_can:
            # Try to load the velocity from CAN data
            pose, indexer_timestamp = self.load_pose_from_can(scene["name"], "pose")
            if pose is not None:
                # Convert the timestamp to a datetime object
                t = pd.to_datetime(sample["timestamp"], utc=True, unit="us", origin="unix")
                # Retrieve the velocity of the closest record by using the indexer and convert to m/s
                i = indexer_timestamp.get_indexer([t], method="nearest")
                (velocity,) = pose.loc[i].vel
                velocity = np.array(velocity, dtype=np.float32)

        # Try to interpolate the velocity based on the previous/next sample
        if velocity is None:
            # Falling back: First, load the current timestamp and position/translation
            data_record_current = self.db.get("sample_data", sample["data"][proxy_sensor])
            ego_pose_current = self.db.get("ego_pose", data_record_current["ego_pose_token"])
            if data_record_current["prev"] != "":
                # Fallback 1: Use the previous record if available
                data_record_prev = self.db.get("sample_data", data_record_current["prev"])
                ego_pose_prev = self.db.get("ego_pose", data_record_prev["ego_pose_token"])
                velocity = self.interpolate_velocity(ego_pose_prev, ego_pose_current)
            elif data_record_current["next"] != "":
                data_record_next = self.db.get("sample_data", data_record_current["next"])
                ego_pose_next = self.db.get("ego_pose", data_record_next["ego_pose_token"])
                velocity = self.interpolate_velocity(ego_pose_current, ego_pose_next)

        # Set the velocity to a dummy value
        if velocity is None:
            velocity = np.array([0, 0, -1], dtype=np.float32)
            logger.warning(
                "Could not load CAN data or interpolate from previous/next sample, substituted EGO velocity by v_z=1."
            )
        return velocity

    def get_sample_meta(self, scene, sample, proxy_sensor="LIDAR_TOP") -> MutableMapping:
        """
        :param scene:
        :param sample:
        :param proxy_sensor: A sensor is required in case the velocity has to estimated from the EGO translation
            because the EGO translation is evaluated at the timestamp of the sensor data record and not per sample.
        :return:
        """
        log = self.db.get("log", scene["log_token"])
        meta = {
            "vehicle": log["vehicle"],
            "location": log["location"],
            "timestamp": sample["timestamp"],
            "scene_name": scene["name"],
            "scene_description": scene["description"],
        }

        if self.dataset.load_velocity:
            velocity = self.get_ego_velocity(scene, sample, proxy_sensor)
            if self.dataset.annotations.coos != "EGO":
                # Get the sample token of the COOS in which annotations are provided
                ref_sample_data_token = sample["data"][self.dataset.annotations.coos]
                # Get the sample data
                sample_data = self.db.get("sample_data", ref_sample_data_token)
                # Get the token of the calibrated sensor record
                calibrated_sensor_data = self.db.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                # Load the inverse rotation between the calibrated sensor and the EGO vehicle as Quaternion
                rotation = Quaternion(calibrated_sensor_data["rotation"]).inverse
                # Transform to the COOS of the annotations
                velocity = rotation.inverse.rotate(velocity)
            meta["velocity"] = torch.from_numpy(velocity).float()

        return meta

    def get_sensor_meta(self, data_token):
        data_record = self.db.get("sample_data", data_token)
        calibrated_sensor_record = self.db.get("calibrated_sensor", data_record["calibrated_sensor_token"])
        meta = {
            "modality": data_record["sensor_modality"],
            "rotation": calibrated_sensor_record["rotation"],
            "translation": calibrated_sensor_record["translation"],
        }
        return meta

    @staticmethod
    def add_timestamps(
        points: np.ndarray, timestamp_reference: pd.Timestamp, timestamp_sample: pd.Timestamp, period: float = 1 / 20
    ):
        """
        Calculate a timestamp for each point and append it to the points array.

        Timestamps are approximated based on the rotation frequency of the sensor and the azimuthal angle of the
        measurement.
        :param points:
        :param timestamp_reference:
        :param timestamp_sample:
        :param period: The time the sensor takes for one complete rotation [ms]
        :return:
        """
        # Precompute angle as factor for timestamp generation
        theta = np.arctan2(points[1, :], points[0, :]) + np.pi
        # Generate timestamps based on mathematically negative rotation of the sensor (Velodyne HDL32E manual)
        timestamp_pts = period * (theta / (2 * np.pi))
        # Compute the time difference between the reference sweep and the current sweep
        delta_t = timestamp_reference - timestamp_sample
        # Offset the points timestamps w.r.t. the reference sweep
        timestamps = timestamp_pts + delta_t.total_seconds()
        # Overwrite the points with the representation that contains the timestamps
        return np.concatenate((points, timestamps[None, :]), axis=0).astype(np.float32)

    def get_sensor_to_global(self, sample_data: dict[str, str]) -> np.ndarray:
        """
        Transform the pointcloud from the current COOS to the COOS of the reference sweep.
        :param sample_data:
        :return:
        """
        # Load/construct the transformation matrix: (current) sensor2global
        ego = self.db.get("ego_pose", sample_data["ego_pose_token"])
        sensor = self.db.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        sensor2ego = transform_matrix(sensor["translation"], Quaternion(sensor["rotation"]), inverse=False)
        ego2global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
        return np.dot(ego2global, sensor2ego)

    def get_global_to_sensor(self, sample_data: Mapping) -> np.ndarray:
        """
        Get the transformation matrix from the global COOS to the COOS of the sensor.
        :param sample_data:
        :return:
        """
        ego = self.db.get("ego_pose", sample_data["ego_pose_token"])
        sensor = self.db.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        ego2sensor = transform_matrix(sensor["translation"], Quaternion(sensor["rotation"]), inverse=True)
        global2ego = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=True)
        return np.dot(ego2sensor, global2ego)

    def load_sweeps(self, sample: Mapping, sensor_id: str, min_distance: float = 1.0) -> torch.Tensor:
        """
        Load the configured number of (previous) sweeps in the COOS of the sample record.
        :param sample:
        :param sensor_id:
        :param min_distance:
        :return:
        """
        sweeps = []

        # Get the reference sample_data(_token)
        ref_sample_data_token = sample["data"][sensor_id]
        sample_data = self.db.get("sample_data", ref_sample_data_token)
        # Load the timestamp of the reference sweep
        timestamp_reference = pd.to_datetime(sample_data["timestamp"], utc=True, unit="us", origin="unix")
        # Load/construct the transformation matrix: global2sensor (reference)
        global2sensor = self.get_global_to_sensor(sample_data)

        # Initialize the sample_data_token to the reference token which is loaded first
        sample_data_token = ref_sample_data_token

        # The idea behind two separate, consecutive loops was to enable asynchronous loading. Furthermore, merging the
        # loops introduces an issue with the loop skipping upon error. The obvious workaround would be duplicate code.
        pc_files = []
        for _ in range(self.dataset.sensors[sensor_id].sweeps):
            file = self.db.get_sample_data_path(sample_data_token)
            pc_files.append((file, sample_data_token, sample_data))
            # Evaluate whether a previous sweep in the current scene exists
            if sample_data["prev"] == "":
                break
            # Overwrite the sample_data(_token) with the one from the previous sweep (that shall be loaded next)
            sample_data_token = sample_data["prev"]
            sample_data = self.db.get("sample_data", sample_data_token)

        for file, sample_data_token, sample_data in pc_files:
            try:
                pc = LidarPointCloud.from_file(file)
            except ValueError:
                logger.warning(f"The pointcloud in {file} could not be loaded due to a corrupted shape.")
                continue

            # Keep only points that are at least min_distance away (others are most likely artefacts)
            pc.points = pc.points[:, np.linalg.norm(pc.points[0:2, :], axis=0) >= min_distance]

            # All loaded sweeps that are not the reference sweep are transformed into the reference sweep frame.
            if ref_sample_data_token != sample_data_token:
                sensor2global = self.get_sensor_to_global(sample_data)
                # Apply fused transformation: (current) sensor2ego -> ego2global -> global2ego -> ego2sensor (reference)
                pc.transform(np.dot(global2sensor, sensor2global))

            # Extract the points array (4, N) from the data class
            pc = pc.points

            # Load the timestamp of the current sweep
            timestamp_sample = pd.to_datetime(sample_data["timestamp"], utc=True, unit="us", origin="unix")
            pc = self.add_timestamps(pc, timestamp_reference, timestamp_sample)

            sweeps.append(pc.astype(np.float32))

        sweeps = np.concatenate(sweeps, axis=1).T

        if self.dataset.annotations.coos == "EGO":
            sample_data = self.db.get("sample_data", ref_sample_data_token)
            sensor = self.db.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            sensor2ego = transform_matrix(sensor["translation"], Quaternion(sensor["rotation"]), inverse=False)
            sweeps_padded = np.concatenate([sweeps[:, :3], np.ones((sweeps.shape[0], 1))], axis=1)
            sweeps[:, :3] = np.einsum("ij,kj->ki", sensor2ego, sweeps_padded)[:, :3]

        sweeps = np.concatenate((sweeps, cart_to_sph_numpy(sweeps[:, :3])), axis=1)

        return torch.from_numpy(sweeps)

    def yield_samples_in_scene(self, sample_token):
        """
        Iterate through all samples in a scene and return the token for each sample.
        :param sample_token:
        :return:
        """
        while sample_token != "":
            yield sample_token
            sample_token = self.db.get("sample", sample_token)["next"]

    def get_scenes(self, scene_names):
        """
        Get the scene tokens for the given scene names.
        :param scene_names:
        :return:
        """
        return [self.scene_name_to_token(sn) for sn in scene_names]

    def scene_name_to_token(self, scene_name):
        """
        Get the token of a scene for the given scene name.
        :param scene_name:
        :return:
        """
        (token,) = self.db.field2token("scene", "name", scene_name)
        return self.db.get("scene", token)

    def get_splits(self) -> Sequence[Sequence, Sequence, Optional[Sequence]]:
        """
        Load the scene tokens for the scenes contained in the respective splits of the datasets.
        :return:
        """
        scenes_mapping = create_splits_scenes()
        if self.dataset.version == "v1.0-mini":
            train_scenes = scenes_mapping["mini_train"]
            val_scenes = scenes_mapping["mini_val"]
            test_scenes = None
        elif self.dataset.version == "v1.0-trainval":
            train_scenes = scenes_mapping["train"]
            val_scenes = scenes_mapping["val"]
            test_scenes = scenes_mapping["test"]
        else:
            raise KeyError("Illegal NuScenes version configuration, choose from ['v1.0-mini', 'v1.0-trainval']")
        return train_scenes, val_scenes, test_scenes


class NuScenesStreamingDataset(StreamingDataset):
    def __init__(
        self,
        *args,
        augmentations: Optional[list[Augmentation]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.augmentations = augmentations

    def __getitem__(self, index):
        data = super().__getitem__(index)
        # litdata deserializes tensors as zero-copy views into the read-only memory-mapped chunk file; the in-place
        # augmentations (and any later in-place OP) would crash with SIGBUS when writing to them, so copy them first.
        data = tree_map_only(torch.Tensor, torch.clone, data)
        if data.get("lidar") is not None:
            data["lidar"] = {sensor: expand_lidar(pc) for sensor, pc in data["lidar"].items()}
        # Apply augmentation to data and targets
        if self.augmentations is not None:
            with torch.inference_mode():
                data = self.apply_augmentations(Sample(**data)).as_dict()
        return data

    def apply_augmentations(self, sample: Sample) -> Sample:
        for augmentation in self.augmentations:
            if isinstance(augmentation, RandomAugmentation):
                augmentation.refresh()
            sample = augmentation(*sample)
        return Sample(*sample)


def partial_collate_fn(batch, sensors: dict[str, Sensor], load_annotations: bool) -> dict:
    """
    Custom collate function that handles the variable-sized point clouds and targets properly.

    Resources:
    - https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    - https://pytorch.org/docs/stable/_modules/torch/utils/data/_utils/collate.html#default_collate
    :param batch:
    :param sensors:
    :param load_annotations:
    :return:
    """
    output = {"metadata": []}
    for sensor, specification in sensors.items():
        if not specification.active:
            continue
        if specification.modality not in output:
            output[specification.modality] = {}
        output[specification.modality][sensor] = []
    if load_annotations:
        output["targets"] = []

    batch_size = len(batch)
    apply_default_collate_keys = []

    # Iterate samples in batch
    for i in range(batch_size):
        # Iterate top level (modalities, annotations, metadata)

        for element, value in batch[i].items():
            if value is None:
                continue
            match element:
                case "lidar" | "camera" | "radar":
                    # Iterate individual sensors
                    for sensor_id, data in value.items():
                        output[element][sensor_id].append(data)
                case "targets" | "metadata":
                    output[element].append(value)
                case _:
                    print(element)
                    if element not in output:
                        output[element] = []
                    output[element].append(value)
                    apply_default_collate_keys.append(element)

    for element in set(apply_default_collate_keys):
        if element in output:
            output[element] = default_collate(output[element])

    return output


class NuScenesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: DatasetConfig,
        augmentations: Optional[dict[Literal["train", "val", "test", "predict"], list[Augmentation]]] = None,
        predict_split: Literal["train", "val"] = "val",
        data_root: Optional[str] = None,
        streaming_data_root: Optional[str] = None,
        streaming_chunk_size: int = 50,
        streaming_cache_size: int | str = "10GB",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = True,
        sampling: Optional[SamplingConfig] = None,
        **dataloader_kwargs,
    ):
        """
        :param sampling: Weighted sampling of the training frames (see point_frustums.dataloaders.sampling); if None,
            the training frames are shuffled.
        """
        super().__init__()
        self.dataset = dataset
        self.sampling = sampling
        self.augmentations = augmentations
        self.predict_split = predict_split
        if streaming_data_root is not None and not storage.is_remote(streaming_data_root):
            streaming_data_root = os.path.abspath(os.path.expanduser(streaming_data_root))
        # Local directory or remote URL (s3://..., see point_frustums.utils.storage)
        self.streaming_data_root = streaming_data_root
        self.streaming_chunk_size = streaming_chunk_size
        self.streaming_cache_size = streaming_cache_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        # The CLI exposes all DataLoader arguments through the kwargs; the sampling is controlled by the datamodule
        for key in ("sampler", "batch_sampler"):
            if dataloader_kwargs.pop(key, None) is not None:
                raise ValueError(f"`{key}` is set by the datamodule, configure the training sampling via `sampling`.")
        self.dataloader_kwargs = dataloader_kwargs

        self.data_root = data_root_getter(
            self.dataset.name, self.dataset.version, root_dir=ROOT_DIR, data_root=data_root
        )
        self.active_modalities = set(sensor.modality for sensor in self.dataset.sensors.values() if sensor.active)

    @cached_property
    def nuscenes_database(self):
        return NuScDB(version=self.dataset.version, dataroot=self.data_root, verbose=False)

    @cached_property
    def nuscenes_database_test(self):
        version = self.dataset.version.rsplit("-")[0] + "-test"
        return NuScDB(version=version, dataroot=self.data_root, verbose=False)

    def nuscenes_split(self, split: Literal["train", "val", "test"]) -> "NuScenes":
        """The NuScenes dataset of a split as it is stored in the optimized (streaming) format."""
        return NuScenes(
            db=self.nuscenes_database_test if split == "test" else self.nuscenes_database,
            dataset=self.dataset,
            load_annotations=split in ("train", "val"),
            augmentations=None,
            split=split,
        )

    def optimize_split(
        self,
        output_dir: str,
        split: Literal["train", "val", "test"],
        indices: Optional[Sequence[int]] = None,
        append: bool = False,
        num_workers: Optional[int] = None,
        chunk_bytes: Optional[int | str] = None,
    ) -> list[str]:
        """
        Optimize (a subset of) the samples of a split into the litdata format.

        The items are stored in the order of the provided indices (with any number of workers and across appends), so
        that per-item information (e.g. the target counts for weighted sampling) can be derived without reading the
        optimized data.
        :param output_dir: Local directory or remote URL (s3://...)
        :param split:
        :param indices: The sample indices to process, defaults to all samples of the split
        :param append: Append to an existing optimized dataset in output_dir
        :param num_workers: Defaults to the number of CPUs
        :param chunk_bytes: Chunk the data by size instead of by the number of items (streaming_chunk_size)
        :return: The sample tokens of the stored items, in the order of the litdata indices
        """
        dataset = self.nuscenes_split(split)
        indices = list(range(len(dataset))) if indices is None else list(indices)
        logger.info(f"Optimizing {len(indices)} samples of the {split} split of the NuScenes dataset into {output_dir}")
        if not storage.is_remote(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        optimize(
            fn=partial(_get_item_compact, dataset),
            inputs=indices,
            output_dir=output_dir,
            chunk_size=None if chunk_bytes is not None else self.streaming_chunk_size,
            chunk_bytes=chunk_bytes,
            num_workers=num_workers or max(1, os.cpu_count()),
            mode="append" if append else None,
            keep_data_ordered=True,
            reorder_files=False,
        )
        return [dataset.sample_tokens[i] for i in indices]

    def prepare_data(self) -> None:
        if self.streaming_data_root is not None:
            splits = ["train", "val"] if self.dataset.version == "v1.0-mini" else ["train", "val", "test"]
            if storage.is_remote(self.streaming_data_root):
                # Remote datasets are built incrementally with point_frustums.dataloaders.build_streaming
                for split in ("train", "val"):
                    if not storage.exists(storage.join(self.streaming_data_root, split, "index.json")):
                        raise FileNotFoundError(
                            f"No optimized {split} split in {self.streaming_data_root}; build it with "
                            "point_frustums.dataloaders.build_streaming."
                        )
            else:
                os.makedirs(self.streaming_data_root, exist_ok=True)
                for split in splits:
                    if not os.path.exists(split_dir := os.path.join(self.streaming_data_root, split)):
                        # Create the optimized version of the split
                        self.optimize_split(output_dir=split_dir, split=split)

            self.train_dataloader = self.train_dataloader_streaming
            self.val_dataloader = self.val_dataloader_streaming
            self.test_dataloader = self.test_dataloader_streaming
            self.predict_dataloader = self.predict_dataloader_streaming

    def train_dataloader_standard(self) -> DataLoader:
        dataset = NuScenes(
            db=self.nuscenes_database,
            dataset=self.dataset,
            load_annotations=True,
            augmentations=self.augmentations.get("train"),
            split="train",
        )
        sampler = None
        if self.sampling is not None:
            sampler = weighted_sampler(dataset.count_targets(), self.sampling, self.dataset.annotations.class_aliases)
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=True)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    def val_dataloader_standard(self) -> DataLoader:
        dataset = NuScenes(
            db=self.nuscenes_database,
            dataset=self.dataset,
            load_annotations=True,
            augmentations=self.augmentations.get("val"),
            split="val",
        )
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=True)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    def test_dataloader_standard(self) -> DataLoader:
        """
        The test dataloader can only be used for actual testing by the NuScenes team as they do not provide the
        annotations.
        """
        dataset = NuScenes(
            db=self.nuscenes_database,
            dataset=self.dataset,
            load_annotations=False,
            augmentations=self.augmentations.get("test"),
            split="test",
        )
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=False)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    def predict_dataloader_standard(self) -> DataLoader:
        dataset = NuScenes(
            db=self.nuscenes_database,
            dataset=self.dataset,
            load_annotations=False,
            augmentations=self.augmentations.get("predict"),
            split="val",
        )
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=False)
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    def train_dataloader_streaming(self) -> StreamingDataLoader | DataLoader:
        input_dir = storage.join(self.streaming_data_root, "train")
        dataset = NuScenesStreamingDataset(
            augmentations=self.augmentations.get("train"),
            input_dir=input_dir,
            max_cache_size=self.streaming_cache_size,
        )
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=True)
        if self.sampling is not None:
            # The StreamingDataLoader does not support samplers; use random access through a map-style view instead.
            # The counts are in the order of the litdata indices (which differs from the order of the sample tokens).
            counts = count_targets_streaming(input_dir, n_classes=self.dataset.annotations.n_classes)
            return DataLoader(
                dataset=IndexedStreamingDataset(dataset, length=num_streaming_items(input_dir)),
                collate_fn=collate_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
                sampler=weighted_sampler(counts, self.sampling, self.dataset.annotations.class_aliases),
                drop_last=self.drop_last,
                pin_memory=self.pin_memory,
                **self.dataloader_kwargs,
            )
        dataloader = StreamingDataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    def val_dataloader_streaming(self) -> StreamingDataLoader:
        dataset = NuScenesStreamingDataset(
            augmentations=self.augmentations.get("val"),
            input_dir=storage.join(self.streaming_data_root, "val"),
            max_cache_size=self.streaming_cache_size,
        )
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=True)
        dataloader = StreamingDataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    def test_dataloader_streaming(self) -> StreamingDataLoader:
        dataset = NuScenesStreamingDataset(
            augmentations=self.augmentations.get("test"),
            input_dir=storage.join(self.streaming_data_root, "test"),
            max_cache_size=self.streaming_cache_size,
        )
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=False)
        dataloader = StreamingDataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    def predict_dataloader_streaming(self) -> StreamingDataLoader:
        dataset = NuScenesStreamingDataset(
            augmentations=self.augmentations.get("val"),
            input_dir=storage.join(self.streaming_data_root, "val"),
            max_cache_size=self.streaming_cache_size,
        )
        collate_fn = partial(partial_collate_fn, sensors=self.dataset.sensors, load_annotations=False)
        dataloader = StreamingDataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.dataloader_kwargs,
        )
        return dataloader

    train_dataloader = train_dataloader_standard
    val_dataloader = val_dataloader_standard
    test_dataloader = test_dataloader_standard
    predict_dataloader = predict_dataloader_standard
