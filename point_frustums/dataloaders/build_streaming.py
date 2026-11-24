"""
Incrementally build the optimized (litdata) NuScenes dataset, part by part, into a local directory or into
S3-compatible object storage such as Cloudflare R2.

The raw sensor data of NuScenes trainval is published in 10 parts. Each part is downloaded and extracted in a single
stream (the archive itself is never stored), all train/val samples whose files are then available are optimized and
appended to the dataset, and the raw files of the part are deleted again. Hence, the local disk only has to hold the
metadata and one part of raw data at a time (~15-20GB for the lidar of one part). With a remote output directory,
litdata uploads every chunk as soon as it is written and removes it locally.

Next to the index.json of each split, target_counts.pt holds the number of targets per class of every item (in the
order of the litdata indices) for weighted sampling and the sample tokens of the items. build_state.json in the output
directory lists the parts that are done. Both are rewritten after each part, so an interrupted build resumes with the
first part that is not done; samples that are already stored are skipped.

Usage (the dataset, sensor and annotation settings are read from the regular training configs):

    # Cloudflare R2: create an API token with object read/write permissions and export
    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
    export AWS_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com AWS_DEFAULT_REGION=auto

    python -m point_frustums.dataloaders.build_streaming --output s3://<bucket>/nuscenes-trainval --parts 1-10 \\
        --config configs/point_frustums.yaml --data.dataset.version=v1.0-trainval

    # Training then reads the dataset from the bucket:
    --data.streaming_data_root=s3://<bucket>/nuscenes-trainval

Without --parts, the samples whose files are already on disk are optimized once (e.g. v1.0-mini).
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from typing import Optional

import torch
from loguru import logger

from point_frustums.dataloaders.sampling import read_target_counts, write_target_counts
from point_frustums.utils import storage

# CloudFront mirror of the public nuScenes bucket (s3://motional-nuscenes, ap-northeast-1), much faster outside Asia
NUSCENES_URL = "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
N_TRAINVAL_PARTS = 10
SPLITS = ("train", "val")
STATE_FILE = "build_state.json"


def part_archive(version: str, part: int, modality: str = "lidar") -> str:
    return f"{version}{part:02d}_blobs_{modality}.tgz"


def parse_parts(spec: str) -> list[int]:
    """Parse e.g. "1-3,5" into [1, 2, 3, 5]."""
    parts = []
    for item in spec.split(","):
        start, _, end = item.partition("-")
        parts.extend(range(int(start), int(end or start) + 1))
    if invalid := [p for p in parts if not 1 <= p <= N_TRAINVAL_PARTS]:
        raise ValueError(f"Invalid parts {invalid}, NuScenes trainval has the parts 1-{N_TRAINVAL_PARTS}")
    return sorted(set(parts))


def download_and_extract(url: str, dest_dir: str, retries: int = 3) -> list[str]:
    """
    Download a .tgz archive and extract it on the fly into dest_dir (the archive is not stored). Files that already
    exist in dest_dir and were not extracted by this call are skipped and not returned, so that removing the returned
    files later does not delete data that was there before (e.g. v1.0-mini, whose scenes are part of v1.0-trainval).
    :return: The paths of the extracted files
    """
    extracted: set[str] = set()  # Across attempts: files of an interrupted attempt are overwritten and still returned
    for attempt in range(1, retries + 1):
        n_bytes, t_start, t_log = 0, time.time(), time.time()
        try:
            with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    path = os.path.join(dest_dir, member.name)
                    if member.isfile():
                        if path not in extracted and os.path.exists(path):
                            continue
                        extracted.add(path)
                        n_bytes += member.size
                    archive.extract(member, dest_dir, filter="data")
                    if time.time() - t_log > 60:
                        t_log = time.time()
                        rate = n_bytes / (t_log - t_start) / 2**20
                        logger.info(f"{os.path.basename(url)}: extracted {n_bytes / 2**30:.1f}GB ({rate:.1f}MB/s)")
            return sorted(extracted)
        except OSError as err:  # Includes connection errors and truncated streams
            logger.warning(f"Downloading {url} failed (attempt {attempt}/{retries}): {err}")
            if attempt == retries:
                raise
    raise AssertionError("unreachable")


def download_and_unzip(url: str, dest_dir: str) -> None:
    """Download a .zip archive (requires random access, hence a temporary file) and extract it into dest_dir."""
    with tempfile.TemporaryDirectory(dir=dest_dir) as tmp:
        archive = os.path.join(tmp, os.path.basename(url))
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if os.path.isabs(name) or ".." in name.split("/"):
                    raise ValueError(f"Unsafe path {name} in {url}")
            zf.extractall(dest_dir)


def ensure_metadata(data_root: str, version: str, load_can: bool, source_url: str = NUSCENES_URL) -> None:
    """Download the metadata (and CAN bus data) if not yet present in data_root."""
    os.makedirs(data_root, exist_ok=True)
    if not os.path.isdir(os.path.join(data_root, version)):
        logger.info(f"Downloading the {version} metadata into {data_root}")
        download_and_extract(f"{source_url}/{version}_meta.tgz", data_root)
    if load_can and not os.path.isdir(os.path.join(data_root, "can_bus")):
        logger.info(f"Downloading the CAN bus expansion into {data_root}")
        download_and_unzip(f"{source_url}/can_bus.zip", data_root)


def remove_files(files: Sequence[str], root: str) -> None:
    """Remove the files and all directories below root that are empty afterwards."""
    directories = set()
    for file in files:
        if os.path.lexists(file):
            os.remove(file)
        directories.add(os.path.dirname(file))
    root = os.path.abspath(root)
    for directory in sorted(directories, key=len, reverse=True):
        directory = os.path.abspath(directory)
        while directory.startswith(root + os.sep) and os.path.isdir(directory) and not os.listdir(directory):
            os.rmdir(directory)
            directory = os.path.dirname(directory)


class StreamingBuild:
    """
    The state of an incremental build of the train and val splits into output_dir (local or s3://).
    :param datamodule: A point_frustums.dataloaders.NuScenesDataModule
    """

    def __init__(
        self,
        datamodule,
        output_dir: str,
        num_workers: Optional[int] = None,
        chunk_bytes: Optional[int | str] = None,
    ):
        self.datamodule = datamodule
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.chunk_bytes = chunk_bytes
        self.datasets = {split: datamodule.nuscenes_split(split) for split in SPLITS}
        n_classes = datamodule.dataset.annotations.n_classes
        self.counts, self.tokens = {}, {}
        state_file = storage.join(output_dir, STATE_FILE)
        state = json.loads(storage.read_bytes(state_file)) if storage.exists(state_file) else {}
        self.parts_done = set(state.get("parts_done", []))
        for split in SPLITS:
            split_dir = storage.join(output_dir, split)
            stored = read_target_counts(split_dir) if storage.exists(storage.join(split_dir, "index.json")) else None
            if stored is None and storage.exists(storage.join(split_dir, "index.json")):
                raise RuntimeError(
                    f"{split_dir} holds an optimized dataset without matching target_counts.pt, it was not created by "
                    "an incremental build or the last part was interrupted while writing; remove it to start over."
                )
            stored = stored or {}
            self.counts[split] = stored.get("counts", torch.zeros((0, n_classes), dtype=torch.int64))
            self.tokens[split] = list(stored.get("sample_tokens", []))

    def is_done(self, part: int) -> bool:
        return part in self.parts_done

    def mark_done(self, part: int) -> None:
        self.parts_done.add(part)
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, STATE_FILE)
            with open(local, "w", encoding="utf-8") as f:
                json.dump({"parts_done": sorted(self.parts_done)}, f)
            storage.write_file(local, storage.join(self.output_dir, STATE_FILE))

    def add_available_samples(self) -> dict[str, int]:
        """
        Optimize and append all samples whose sensor data files are on disk and that are not yet in the dataset.
        :return: The number of added samples per split
        """
        added = {}
        for split in SPLITS:
            dataset, split_dir = self.datasets[split], storage.join(self.output_dir, split)
            done = set(self.tokens[split])
            indices, incomplete = [], 0
            for i, token in enumerate(dataset.sample_tokens):
                if token in done:
                    continue
                available = [os.path.exists(f) for f in dataset.required_files(i)]
                if all(available):
                    indices.append(i)
                elif any(available):
                    incomplete += 1
            if incomplete:
                logger.warning(f"{incomplete} {split} samples have only part of their sensor data files on disk")
            if indices:
                tokens = self.datamodule.optimize_split(
                    output_dir=split_dir,
                    split=split,
                    indices=indices,
                    append=len(self.tokens[split]) > 0,
                    num_workers=self.num_workers,
                    chunk_bytes=self.chunk_bytes,
                )
                self.tokens[split].extend(tokens)
                self.counts[split] = torch.cat([self.counts[split], self.all_counts(split)[indices]])
                write_target_counts(split_dir, self.counts[split], self.tokens[split])
            added[split] = len(indices)
        return added

    def all_counts(self, split: str) -> torch.Tensor:
        """The targets per class of all samples of the split (from the metadata, no sensor data required)."""
        if not hasattr(self, "_all_counts"):
            self._all_counts = {}
        if split not in self._all_counts:
            self._all_counts[split] = self.datasets[split].count_targets()
        return self._all_counts[split]


def build(
    datamodule,
    output_dir: str,
    parts: Optional[Sequence[int]] = None,
    num_workers: Optional[int] = None,
    chunk_bytes: Optional[int | str] = None,
    keep_raw: bool = False,
    source_url: str = NUSCENES_URL,
    fetch_part: Optional[Callable[[int], list[str]]] = None,
) -> None:
    """
    :param datamodule: A point_frustums.dataloaders.NuScenesDataModule
    :param output_dir: Local directory or remote URL (s3://...) of the optimized dataset
    :param parts: The trainval parts to process; None processes the samples that are already on disk
    :param num_workers: The number of workers to optimize with; each holds a copy of the NuScenes metadata in memory
    :param chunk_bytes: The size of the litdata chunks
    :param keep_raw: Do not delete the raw files of a part after processing it
    :param source_url: Base URL of the nuScenes archives
    :param fetch_part: Override how the raw files of a part are obtained (returns the paths of the files to delete)
    """
    dataset_config = datamodule.dataset
    data_root = datamodule.data_root
    if parts is not None:
        ensure_metadata(data_root, dataset_config.version, dataset_config.load_can, source_url)
    state = StreamingBuild(datamodule, output_dir, num_workers=num_workers, chunk_bytes=chunk_bytes)

    if parts is None:
        added = state.add_available_samples()
        logger.info(f"Added {added} samples to {output_dir}")
        return

    if fetch_part is None:

        def fetch_part(part: int) -> list[str]:
            url = f"{source_url}/{part_archive(dataset_config.version, part)}"
            logger.info(f"Downloading and extracting {url} into {data_root}")
            return download_and_extract(url, data_root)

    for part in parts:
        if state.is_done(part):
            logger.info(f"Part {part} is already done, skipping")
            continue
        t_start = time.time()
        files = fetch_part(part)
        added = state.add_available_samples()
        state.mark_done(part)
        if not keep_raw:
            remove_files(files, data_root)
        logger.info(
            f"Part {part}: added {added} samples to {output_dir} in {(time.time() - t_start) / 60:.1f}min, in total "
            f"{ {split: len(state.tokens[split]) for split in SPLITS} }"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="All other arguments are passed to the LightningCLI (e.g. --config configs/point_frustums.yaml) to "
        "configure the datamodule.",
    )
    parser.add_argument("--output", required=True, help="Local directory or s3://bucket/prefix of the dataset")
    parser.add_argument("--parts", default=None, help="Trainval parts to process, e.g. 1-10 or 1,4; default: on disk")
    parser.add_argument("--num-workers", type=int, default=4, help="Optimization workers (each holds the metadata)")
    parser.add_argument("--chunk-bytes", default="64MB", help="Size of the litdata chunks")
    parser.add_argument("--keep-raw", action="store_true", help="Keep the raw files of the parts")
    parser.add_argument("--source-url", default=NUSCENES_URL, help="Base URL of the nuScenes archives")
    args, cli_args = parser.parse_known_args(argv)

    from pytorch_lightning.cli import LightningCLI  # pylint: disable=import-outside-toplevel

    cli = LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        run=False,
        args=[*cli_args, "--trainer.logger=false", "--trainer.accelerator=cpu"],
    )
    build(
        cli.datamodule,
        output_dir=args.output if storage.is_remote(args.output) else os.path.abspath(args.output),
        parts=parse_parts(args.parts) if args.parts else None,
        num_workers=args.num_workers,
        chunk_bytes=args.chunk_bytes,
        keep_raw=args.keep_raw,
        source_url=args.source_url,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
