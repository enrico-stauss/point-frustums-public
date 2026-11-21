import json
import os
from collections.abc import Sequence
from enum import Enum
from functools import cached_property
from itertools import chain
from math import isclose
from typing import Optional

import pytest
import torch
from nuscenes import NuScenes as NuScDB
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import config_factory
from pyquaternion import Quaternion

from point_frustums import ROOT_DIR
from point_frustums.metrics.detection.nds import NuScenesDetectionScore
from point_frustums.utils.environment_helpers import data_root_getter

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True, scope="module")
def float64_default_dtype():
    """Evaluate in FP64, but restore the default afterward so it does not leak into other test modules."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


class Labels:
    def __init__(self, labels: Sequence):
        self.labels = Enum("labels", labels, start=0)

    def from_index(self, index: int) -> Enum:
        """
        Based on the integer label of the class, return an Enum for the class.
        :param index:
        :return:
        """
        return self.labels(index)

    def from_name(self, name: str) -> Enum:
        """
        Based on the class alias, return the Enum for the class.
        :param name:
        :return:
        """
        return self.labels[name]


class Annotations:
    coos: str = "LIDAR_TOP"
    class_aliases: list[str] = [
        "pedestrian",
        "car",
        "bus",
        "bicycle",
        "truck",
        "trailer",
        "construction_vehicle",
        "traffic_cone",
        "barrier",
        "motorcycle",
    ]
    alias_to_class: dict[str, str] = {
        "pedestrian": "human.pedestrian",
        "car": "vehicle.car",
        "bus": "vehicle.bus",
        "bicycle": "vehicle.bicycle",
        "truck": "vehicle.truck",
        "trailer": "vehicle.trailer",
        "construction_vehicle": "vehicle.construction",
        "traffic_cone": "movable_object.trafficcone",
        "barrier": "movable_object.barrier",
        "motorcycle": "vehicle.motorcycle",
    }
    alias_to_category: dict[str, str] = {
        "pedestrian": "pedestrian",
        "car": "vehicle",
        "bus": "vehicle",
        "bicycle": "cycle",
        "truck": "vehicle",
        "trailer": "vehicle",
        "construction_vehicle": "vehicle",
        "traffic_cone": "other",
        "barrier": "other",
        "motorcycle": "cycle",
    }
    category_to_attributes: dict[str, list] = {
        "pedestrian": ["moving", "standing", "sitting_lying_down"],
        "vehicle": ["moving", "parked", "stopped"],
        "cycle": ["with_rider", "without_rider"],
        "other": ["void"],
    }

    @cached_property
    def classes(self) -> Labels:
        """
        Return an Enum name to integer mapping created from the class aliases.
        :return:
        """
        return Labels(self.class_aliases)

    @cached_property
    def n_classes(self) -> int:
        return len(self.classes.labels)

    @cached_property
    def attributes(self) -> Labels:
        _attributes = list(set(chain.from_iterable(self.category_to_attributes.values())))
        return Labels(_attributes)

    @cached_property
    def n_attributes(self) -> int:
        return len(self.attributes.labels)

    @cached_property
    def class_to_alias(self) -> dict[str, str]:
        return {value: key for key, value in self.alias_to_class.items()}

    @cached_property
    def category_to_alias(self) -> dict[str, str]:
        return {value: key for key, value in self.alias_to_category.items()}

    def retrieve_class(self, class_name_verbose: str) -> Optional[Enum]:
        for class_alias, class_name_abbreviated in self.alias_to_class.items():
            if class_name_verbose.startswith(class_name_abbreviated):
                return self.classes.from_name(class_alias)
        return None

    def resolve_attribute(self, attribute: str, class_alias: str) -> str:
        """
        Resolve the short-form attribute to the verbose form. Check first, if the predicted attribute is at all valid
        for the predicted class, if it is not, replace with the category default.
        :param attribute: The short-form attribute {moving, standing, sitting_lying_down, parked, stopped, void, ...}
        :param class_alias:
        :return: The resolved, verbose attribute name
        """
        category = self.alias_to_category[class_alias]
        attribute_choices = self.category_to_attributes[category]
        if attribute not in attribute_choices:
            attribute = attribute_choices[0]

        if attribute == "void":
            attribute = ""
        else:
            attribute = f"{category}.{attribute}"

        return attribute


def load_targets(nuscenes_db: NuScDB, eval_split: str):
    return load_gt(nuscenes_db, eval_split, DetectionBox)


def load_detections(results_path: str):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvalBoxes.deserialize(data["results"], DetectionBox)


def build_transformation_matrix(rotation: list, translation: list, inverse: bool = False) -> torch.Tensor:
    float_type = torch.get_default_dtype()
    rotation = torch.tensor(Quaternion(rotation).rotation_matrix).to(float_type)
    translation = torch.tensor(translation).to(float_type)
    matrix = torch.eye(4).to(float_type)
    if inverse:
        rotation = rotation.T
        translation = -rotation @ translation
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def transform_boxes(transformation: torch.Tensor, boxes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Apply the 4x4 transformation matrix to centers, orientations and velocities.
    :param transformation:
    :param boxes:
    :return:
    """
    n = boxes["class"].numel()
    center = torch.ones((n, 4))
    center[:, :3] = boxes["center"]
    orientation = torch.eye(4)[None, ...].repeat(n, 1, 1)
    orientation[:, :3, :3] = boxes["orientation"]
    velocity = boxes["velocity"]
    center = torch.einsum("ij,...j->...i", transformation, center)[..., :3]
    orientation = torch.einsum("ij,...jk->...ik", transformation, orientation)[..., :3, :3]
    velocity = torch.einsum("ij,...j->...i", transformation[:2, :2], velocity)
    boxes["center"] = center
    boxes["orientation"] = orientation
    boxes["velocity"] = velocity
    return boxes


def prepare_boxes(
    nusc: NuScDB, boxes: EvalBoxes, annotations_config: Annotations, transform: bool = False
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load boxes, transform if specified, retrieve integer labels and stack into tensors.
    :param nusc:
    :param boxes:
    :param annotations_config:
    :param transform:
    :return:
    """
    float_type = torch.get_default_dtype()
    return_boxes = {}
    for sample_token, sample_boxes in boxes.boxes.items():
        sample_boxes: list[DetectionBox]
        sample = nusc.get("sample", sample_token)
        data_token = sample["data"][annotations_config.coos]
        sample_data = nusc.get("sample_data", data_token)
        # Create the transformation from the global to the EGO frame
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        global_to_ego = build_transformation_matrix(ego_pose["rotation"], ego_pose["translation"], inverse=True)
        # Create the transformation from the EGO to the global frame
        sensor_pose = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        ego_to_sensor = build_transformation_matrix(sensor_pose["rotation"], sensor_pose["translation"], inverse=True)
        # Chain the transformations
        global_to_sensor = torch.einsum("ij,jk->ik", ego_to_sensor, global_to_ego)

        sample_boxes_dict = {
            "class": [],
            "center": [],
            "ego_translation": [],
            "wlh": [],
            "orientation": [],
            "velocity": [],
            "attribute": [],
            "score": [],
        }

        for b in sample_boxes:
            sample_boxes_dict["class"].append(annotations_config.classes.from_name(b.detection_name).value)
            attribute = b.attribute_name.split(".")[-1] or "void"
            sample_boxes_dict["attribute"].append(annotations_config.attributes.from_name(attribute).value)
            sample_boxes_dict["center"].append(b.translation)
            sample_boxes_dict["ego_translation"].append(b.ego_translation)
            sample_boxes_dict["wlh"].append(b.size)
            sample_boxes_dict["orientation"].append(Quaternion(b.rotation).rotation_matrix)
            sample_boxes_dict["velocity"].append(b.velocity)
            sample_boxes_dict["score"].append(b.detection_score)

        sample_boxes_dict["class"] = torch.tensor(sample_boxes_dict["class"]).int()
        sample_boxes_dict["attribute"] = torch.tensor(sample_boxes_dict["attribute"]).int()
        sample_boxes_dict["center"] = torch.tensor(sample_boxes_dict["center"]).to(float_type)
        sample_boxes_dict["ego_translation"] = torch.tensor(sample_boxes_dict["ego_translation"]).to(float_type)
        sample_boxes_dict["wlh"] = torch.tensor(sample_boxes_dict["wlh"]).to(float_type)
        sample_boxes_dict["orientation"] = torch.tensor(sample_boxes_dict["orientation"]).to(float_type)
        sample_boxes_dict["velocity"] = torch.tensor(sample_boxes_dict["velocity"]).to(float_type)
        sample_boxes_dict["score"] = torch.tensor(sample_boxes_dict["score"]).to(float_type)

        if transform:
            # Transform to the coordinate system of the specified sensor
            sample_boxes_dict = transform_boxes(global_to_sensor, sample_boxes_dict)

        return_boxes[sample_token] = sample_boxes_dict

    return return_boxes


@pytest.mark.parametrize("case", ["case_01", "case_02"])
class TestNuScenesDetectionScore:
    version = "v1.0-mini"
    eval_split = "mini_val"
    nuscenes_db = NuScDB(version, data_root_getter("NuScenes", version, ROOT_DIR), verbose=False)
    annotations_config = Annotations()
    nuscenes_evaluator_config = config_factory("detection_cvpr_2019")

    def _preprocess_boxes(self, boxes):
        boxes = add_center_dist(self.nuscenes_db, boxes)
        boxes = filter_eval_boxes(
            self.nuscenes_db,
            boxes,
            self.nuscenes_evaluator_config.class_range,
            verbose=False,
        )
        return boxes

    def _prepare_targets(self, coos: str = "global"):
        targets = load_targets(self.nuscenes_db, self.eval_split)
        targets = self._preprocess_boxes(targets)
        transform = False
        if coos == "lidar":
            transform = True
        return prepare_boxes(self.nuscenes_db, targets, self.annotations_config, transform=transform)

    def _prepare_detections(self, detections_file, coos: str = "global"):
        detections = load_detections(detections_file)
        detections = self._preprocess_boxes(detections)
        transform = False
        if coos == "lidar":
            transform = True
        return prepare_boxes(self.nuscenes_db, detections, self.annotations_config, transform=transform)

    @pytest.mark.parametrize("coos", ["global", "lidar"])
    def test_cases(self, case: str, coos: str):
        assert coos in ("global", "lidar")
        detections_file = f"{TEST_DIR}/nds_test_data/{case}_detections.json"
        targets = self._prepare_targets(coos=coos)
        detections = self._prepare_detections(detections_file, coos=coos)

        metric = NuScenesDetectionScore(self.annotations_config)
        assert set(detections.keys()) == set(targets.keys())
        for sample_token, sample_detections in detections.items():
            metric.update([sample_detections], [targets[sample_token]])
        metric_results = metric.compute(f"{TEST_DIR}/nds_test_output/metric_results_{case}_{coos}.json")

        with open(f"{TEST_DIR}/nds_test_data/{case}_results.json", "r", encoding="utf-8") as f:
            reference_results = json.load(f)

        abs_tol = 1e-8
        if coos == "lidar":
            # Evaluating in a COOS other than the global one is not really possible with high accuracy.
            # The NDS uses the 2D distance to match detections to targets and also evaluate TP metrics.
            abs_tol = 2e-3

        assert isclose(reference_results["mean_ap"], metric_results["mAP"], abs_tol=abs_tol)
        assert isclose(reference_results["tp_errors"]["trans_err"], metric_results["mATE"], abs_tol=abs_tol)
        assert isclose(reference_results["tp_errors"]["scale_err"], metric_results["mASE"], abs_tol=abs_tol)
        assert isclose(reference_results["tp_errors"]["orient_err"], metric_results["mAOE"], abs_tol=abs_tol)
        assert isclose(reference_results["tp_errors"]["vel_err"], metric_results["mAVE"], abs_tol=abs_tol)
        assert isclose(reference_results["tp_errors"]["attr_err"], metric_results["mAAE"], abs_tol=abs_tol)
        assert isclose(reference_results["nd_score"], metric_results["NDS"], abs_tol=abs_tol)
