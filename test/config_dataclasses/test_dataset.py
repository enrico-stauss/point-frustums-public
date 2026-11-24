import subprocess
import sys

CODE = """
from point_frustums.config_dataclasses.dataset import Annotations
annotations = Annotations(
    coos="EGO", visibility=0, class_aliases=["car"], alias_to_class={"car": "vehicle.car"},
    alias_to_category={"car": "vehicle"},
    category_to_attributes={
        "pedestrian": ["moving", "standing", "sitting_lying_down"], "vehicle": ["moving", "parked", "stopped"],
        "cycle": ["with_rider", "without_rider"], "other": ["void"],
    },
)
print(",".join(label.name for label in annotations.attributes.labels))
"""


def test_attribute_labels_are_identical_across_processes():
    # The integer attribute labels are stored in the optimized dataset by one process and read by others; they must not
    #   depend on the (per process randomized) hash seed
    orders = {
        subprocess.run(
            [sys.executable, "-c", CODE], env={"PYTHONHASHSEED": str(seed)}, capture_output=True, text=True, check=True
        ).stdout.strip()
        for seed in range(5)
    }
    assert orders == {"moving,standing,sitting_lying_down,parked,stopped,with_rider,without_rider,void"}
