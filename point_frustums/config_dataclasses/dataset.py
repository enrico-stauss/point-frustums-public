from collections.abc import Sequence
from dataclasses import dataclass, asdict, fields
from enum import Enum
from functools import cached_property
from itertools import chain
from typing import Literal, Optional


class Labels:
    def __init__(self, labels: Sequence):
        self.labels = Enum("labels", labels, start=0)

    def from_index(self, index: int) -> Enum:
        return self.labels(index)

    def from_name(self, name: str) -> Enum:
        return self.labels[name]


@dataclass(frozen=True)
class Annotations:
    coos: str
    visibility: float
    class_aliases: list[str]
    alias_to_class: dict[str, str]
    alias_to_category: dict[str, str]
    category_to_attributes: dict[str, list]

    def serialize(self):
        """
        Return the serial representation that was used to initialize the dataclass.
        :return:
        """
        return asdict(self)

    def __getstate__(self):
        # Only pickle the fields: the cached label enums are created dynamically and cannot be pickled (e.g. when the
        # dataset is sent to DataLoader worker processes); all cached properties are recreated lazily after unpickling.
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @cached_property
    def classes(self) -> Labels:
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


@dataclass(slots=True, frozen=True)
class Sensor:
    active: bool
    modality: Literal["lidar", "camera", "radar"]
    channels_out: list[str]
    angle_of_view_degrees: dict[Literal["polar", "azimuthal"], int]
    orientation_degrees: dict[Literal["polar", "azimuthal"], int]
    sweeps: Optional[int] = None
    resolution: Optional[dict[Literal["polar", "azimuthal"], int]] = None


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    name: str
    version: str
    sensors: dict[str, Sensor]
    annotations: Annotations
    load_velocity: bool = False
    load_can: bool = False
