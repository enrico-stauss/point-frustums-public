import os

import pytest
import torch
from litdata import optimize

from point_frustums.dataloaders.sampling import (
    SamplingConfig,
    class_weights_from_counts,
    count_targets_streaming,
    frame_weights,
    num_streaming_items,
)

CLASSES = ["car", "pedestrian", "bicycle"]
# 4 frames x 3 classes: car is frequent, bicycle is rare
COUNTS = torch.tensor(
    [
        [10, 0, 0],
        [10, 5, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
)


def test_class_weights_inverse_frequency_rarest_is_one():
    weights = class_weights_from_counts(COUNTS, CLASSES)
    # totals: car 20, pedestrian 5, bicycle 1
    torch.testing.assert_close(weights, torch.tensor([1 / 20, 1 / 5, 1.0], dtype=torch.float64))


def test_class_weights_override_and_validation():
    weights = class_weights_from_counts(COUNTS, CLASSES, overrides={"car": 0.2})
    assert weights[0].item() == pytest.approx(0.2)
    with pytest.raises(ValueError):
        class_weights_from_counts(COUNTS, CLASSES, overrides={"truck": 1.0})


def test_target_count_weights():
    weights = frame_weights(COUNTS, SamplingConfig(strategy="target_count"), CLASSES)
    # 1 + sum_c w_c * n_c with w = [1/20, 1/5, 1]
    torch.testing.assert_close(weights, torch.tensor([1.5, 2.5, 2.0, 1.0], dtype=torch.float64))
    capped = frame_weights(COUNTS, SamplingConfig(strategy="target_count", max_frame_weight=2.0), CLASSES)
    assert capped.max().item() == pytest.approx(2.0)


def test_cbgs_gives_every_class_the_same_share_of_draws():
    weights = frame_weights(COUNTS, SamplingConfig(strategy="cbgs"), CLASSES)
    probabilities = weights / weights.sum()
    contains = (COUNTS > 0).double()
    # Probability mass assigned on behalf of each class: frames containing it, divided by the number of such frames
    share_per_class = (contains / contains.sum(0)).T @ (torch.ones(4, dtype=torch.float64) / 3)
    torch.testing.assert_close(share_per_class, torch.full((3,), 1 / 3, dtype=torch.float64))
    # Frames without targets are never drawn
    assert probabilities[3] == 0


def test_uniform_weights():
    weights = frame_weights(COUNTS, SamplingConfig(strategy="uniform"), CLASSES)
    assert torch.equal(weights, torch.ones(4, dtype=torch.float64))


def _make_item(index: int):
    classes = torch.cat([torch.full((int(n),), c) for c, n in enumerate(COUNTS[index].tolist())]).long()
    return {"targets": {"class": classes}, "index": index}


def test_count_targets_streaming_matches_items_and_is_cached(tmp_path):
    output_dir = str(tmp_path / "train")
    optimize(fn=_make_item, inputs=list(range(len(COUNTS))), output_dir=output_dir, chunk_size=2, num_workers=1)
    assert num_streaming_items(output_dir) == len(COUNTS)

    counts = count_targets_streaming(output_dir, n_classes=len(CLASSES))
    # The counts are in the order of the litdata indices; compare as a multiset of rows
    assert sorted(counts.tolist()) == sorted(COUNTS.tolist())
    cache_file = os.path.join(output_dir, "target_counts.pt")
    cached = torch.load(cache_file, weights_only=True)
    assert torch.equal(cached["counts"], counts)

    # A valid cache is used as is
    torch.save({"fingerprint": cached["fingerprint"], "counts": counts + 1}, cache_file)
    assert torch.equal(count_targets_streaming(output_dir, n_classes=len(CLASSES)), counts + 1)
    # An outdated cache (the index.json changed) is recomputed
    torch.save({"fingerprint": "outdated", "counts": counts + 1}, cache_file)
    assert torch.equal(count_targets_streaming(output_dir, n_classes=len(CLASSES)), counts)
