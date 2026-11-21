"""
Weighted sampling of training frames based on the number of (relevant) targets per class.

Strategies:
    - uniform: Every frame is drawn with the same probability (equivalent to shuffling, but with replacement).
    - target_count: Frame weight `1 + sum_c w_c * n_c` where n_c is the number of targets of class c in the frame and
        w_c the worth of a single target of class c. Frames with many relevant targets are drawn more often and a
        target of a rare class is worth more than one of a frequent class. By default, w_c is derived from the class
        frequency in the training split (w_c = n_min / n_c, i.e. the rarest class gets 1); individual weights can be
        overridden.
    - cbgs: Class-balanced sampling (Zhu et al., 2019, "Class-balanced Grouping and Sampling for Point Cloud 3D Object
        Detection"): every class receives the same share of draws, which is spread uniformly over the frames that
        contain the class. Frames without any target are never drawn.
"""

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from litdata import StreamingDataset
from loguru import logger
from torch.utils.data import Dataset, WeightedRandomSampler

from point_frustums.utils import storage


@dataclass(frozen=True)
class SamplingConfig:
    strategy: Literal["uniform", "target_count", "cbgs"] = "target_count"
    # Overrides of the per-target worth for individual classes (by alias), only used by `target_count`
    class_weights: Optional[dict[str, float]] = None
    # Upper bound of the frame weight, only used by `target_count`; prevents few crowded frames from dominating
    max_frame_weight: Optional[float] = None
    # Number of frames drawn per epoch, defaults to the number of frames in the split
    num_samples: Optional[int] = None


def class_weights_from_counts(
    counts: torch.Tensor, class_aliases: Sequence[str], overrides: Optional[dict[str, float]] = None
) -> torch.Tensor:
    """
    Worth of a single target per class: inversely proportional to the number of targets of the class in the split,
    normalized such that the rarest (present) class has a weight of 1.
    :param counts: Number of targets per frame and class [N, C]
    :param class_aliases: Class names in the order of the label indices
    :param overrides: Fixed weights for individual classes
    :return: [C]
    """
    totals = counts.sum(dim=0).double()
    present = totals > 0
    weights = torch.zeros_like(totals)
    if present.any():
        weights[present] = totals[present].min() / totals[present]
    for alias, weight in (overrides or {}).items():
        if alias not in class_aliases:
            raise ValueError(f"Unknown class '{alias}' in the sampling class weights, valid are {list(class_aliases)}")
        weights[class_aliases.index(alias)] = weight
    return weights


def frame_weights(counts: torch.Tensor, config: SamplingConfig, class_aliases: Sequence[str]) -> torch.Tensor:
    """
    Evaluate the sampling weight of every frame.
    :param counts: Number of targets per frame and class [N, C]
    :param config:
    :param class_aliases: Class names in the order of the label indices
    :return: [N]
    """
    counts = counts.double()
    match config.strategy:
        case "uniform":
            weights = torch.ones(counts.shape[0], dtype=torch.float64)
        case "target_count":
            class_weights = class_weights_from_counts(counts, class_aliases, config.class_weights)
            weights = 1 + counts @ class_weights
            if config.max_frame_weight is not None:
                weights = weights.clamp(max=config.max_frame_weight)
        case "cbgs":
            contains = counts > 0
            frames_per_class = contains.sum(dim=0)
            present = frames_per_class > 0
            weights = (contains[:, present].double() / frames_per_class[present]).sum(dim=1) / present.sum()
            if weights.sum() == 0:
                weights = torch.ones_like(weights)
        case _:
            raise ValueError(f"Unknown sampling strategy '{config.strategy}'")
    return weights


def sampling_summary(weights: torch.Tensor, counts: torch.Tensor, class_aliases: Sequence[str]) -> str:
    """Describe the expected effect of the frame weights compared to uniform sampling."""
    counts = counts.double()
    probabilities = weights / weights.sum()
    expected = probabilities @ counts
    uniform = counts.mean(dim=0)
    expected_share, uniform_share = expected / expected.sum(), uniform / uniform.sum()
    n_top = max(1, len(weights) // 10)
    lines = [
        f"{len(weights)} frames, max/min frame weight {(weights.max() / weights[weights > 0].min()).item():.1f}, "
        f"the 10% highest weighted frames receive {100 * probabilities.sort(descending=True).values[:n_top].sum():.1f}%"
        " of the draws. Expected share of the targets per class (uniform -> weighted):"
    ]
    for alias, u, e in zip(class_aliases, uniform_share.tolist(), expected_share.tolist()):
        lines.append(f"    {alias:22s} {100 * u:5.1f}% -> {100 * e:5.1f}%")
    return "\n".join(lines)


def weighted_sampler(
    counts: torch.Tensor, config: SamplingConfig, class_aliases: Sequence[str]
) -> WeightedRandomSampler:
    weights = frame_weights(counts, config, class_aliases)
    logger.info(f"Weighted sampling ({config.strategy}): {sampling_summary(weights, counts, class_aliases)}")
    num_samples = config.num_samples if config.num_samples is not None else len(weights)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


TARGET_COUNTS_FILE = "target_counts.pt"


def _read_index(input_dir: str) -> bytes:
    return storage.read_bytes(storage.join(input_dir, "index.json"))


def _fingerprint(index: bytes) -> str:
    return hashlib.sha1(index).hexdigest()


def num_streaming_items(input_dir: str) -> int:
    """
    Total number of items in an optimized litdata directory (independent of workers and distributed setup). The
    directory can be local or remote (see point_frustums.utils.storage).
    """
    return sum(chunk["chunk_size"] for chunk in json.loads(_read_index(input_dir))["chunks"])


def write_target_counts(
    input_dir: str,
    counts: torch.Tensor,
    sample_tokens: Optional[Sequence[str]] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Store the targets per class of every item (in the order of the litdata indices) next to the index.json of an
    optimized litdata directory, tagged with the fingerprint of the current index.json.
    :param input_dir: Local or remote litdata directory
    :param counts: [N_items, N_classes]
    :param sample_tokens: Optionally, the sample token of every item (used to resume incremental builds)
    :param extra: Optionally, additional (picklable with weights_only) entries, e.g. the state of an incremental build
    """
    index = _read_index(input_dir)
    if (n_items := sum(chunk["chunk_size"] for chunk in json.loads(index)["chunks"])) != counts.shape[0]:
        raise ValueError(f"Got target counts for {counts.shape[0]} items but the dataset holds {n_items} items")
    content = {"fingerprint": _fingerprint(index), "counts": counts}
    if sample_tokens is not None:
        content["sample_tokens"] = list(sample_tokens)
    content.update(extra or {})
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, TARGET_COUNTS_FILE)
        torch.save(content, local)
        storage.write_file(local, storage.join(input_dir, TARGET_COUNTS_FILE))


def read_target_counts(input_dir: str) -> Optional[dict]:
    """
    Load the stored target counts if they exist and match the current index.json of the litdata directory, else None.
    :return: {"fingerprint": str, "counts": [N_items, N_classes], "sample_tokens": Optional[list[str]]} or None
    """
    path = storage.join(input_dir, TARGET_COUNTS_FILE)
    if not storage.exists(path):
        return None
    content = torch.load(io.BytesIO(storage.read_bytes(path)), weights_only=True)
    if content["fingerprint"] != _fingerprint(_read_index(input_dir)):
        return None
    return content


class IndexedStreamingDataset(Dataset):
    """
    Map-style view of a litdata StreamingDataset so that it can be used with a torch DataLoader and a sampler. The
    StreamingDataLoader does not support custom samplers but the StreamingDataset provides random access by index.
    """

    def __init__(self, dataset: StreamingDataset, length: int):
        self.dataset = dataset
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.dataset[int(index)]


def count_targets_streaming(input_dir: str, n_classes: int) -> torch.Tensor:
    """
    Count the targets per class for every item of an optimized litdata directory, in the order of the litdata indices.

    The counts are cached next to the index.json (see write_target_counts) and recomputed if the latter changes. For
    local directories, missing counts are taken from the stored targets, hence are consistent with the training targets
    by construction. Reading an item takes ~1ms (the chunks are memory-mapped), so a single process is faster than
    spawning workers. Remote directories have to provide the counts, as counting would download the whole dataset; the
    streaming build (point_frustums.dataloaders.build_streaming) writes them.
    :return: [N_items, N_classes]
    """
    cached = read_target_counts(input_dir)
    if cached is not None and cached["counts"].shape[1] == n_classes:
        return cached["counts"]
    if storage.is_remote(input_dir):
        raise FileNotFoundError(
            f"No up-to-date {TARGET_COUNTS_FILE} in {input_dir}; it is written when building the dataset with "
            "point_frustums.dataloaders.build_streaming."
        )

    n_items = num_streaming_items(input_dir)
    logger.info(f"Counting the targets per class of {n_items} items in {input_dir} (cached afterwards)")
    dataset = StreamingDataset(input_dir=input_dir)
    counts = torch.stack(
        [torch.bincount(dataset[i]["targets"]["class"].long(), minlength=n_classes) for i in range(n_items)]
    )
    try:
        write_target_counts(input_dir, counts)
    except OSError as err:
        logger.warning(f"Could not cache the target counts in {input_dir}: {err}")
    return counts
