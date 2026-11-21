import json
import os
import tarfile

import pytest
import torch
import yaml
from litdata import StreamingDataset

from point_frustums import ROOT_DIR
from point_frustums.dataloaders import build_streaming
from point_frustums.dataloaders.sampling import count_targets_streaming, num_streaming_items, read_target_counts

CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "point_frustums.yaml")


def _mini_root():
    # Read the configured data root without data_root_getter, which prompts for missing roots and rewrites the file
    path = os.path.join(ROOT_DIR, "..", ".data-root.yaml")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        root = (yaml.safe_load(f) or {}).get("NuScenes", {}).get("v1.0-mini")
    return root if root is not None and os.path.isdir(os.path.join(root, "v1.0-mini")) else None


def test_parse_parts():
    assert build_streaming.parse_parts("1-3,5") == [1, 2, 3, 5]
    assert build_streaming.parse_parts("10") == [10]
    with pytest.raises(ValueError):
        build_streaming.parse_parts("0-2")


def test_download_and_extract_streams_tgz(tmp_path):
    source = tmp_path / "source" / "samples" / "LIDAR_TOP"
    source.mkdir(parents=True)
    (source / "a.pcd.bin").write_bytes(b"a" * 100)
    (source / "b.pcd.bin").write_bytes(b"b" * 50)
    archive = tmp_path / "part.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "source" / "samples", arcname="samples")

    dest = tmp_path / "root"
    files = build_streaming.download_and_extract(archive.as_uri(), str(dest))
    assert sorted(os.path.relpath(f, dest) for f in files) == [
        "samples/LIDAR_TOP/a.pcd.bin",
        "samples/LIDAR_TOP/b.pcd.bin",
    ]
    assert (dest / "samples" / "LIDAR_TOP" / "a.pcd.bin").read_bytes() == b"a" * 100

    build_streaming.remove_files(files, str(dest))
    assert not (dest / "samples").exists()
    assert dest.exists()


@pytest.mark.skipif(_mini_root() is None, reason="NuScenes v1.0-mini is not available")
def test_incremental_build_on_mini(tmp_path, monkeypatch):
    from pytorch_lightning.cli import LightningCLI  # pylint: disable=import-outside-toplevel

    mini_root = _mini_root()
    # A data root that only holds the metadata; the lidar files are "downloaded" (symlinked) part by part
    data_root = tmp_path / "nuscenes"
    data_root.mkdir()
    for entry in ("v1.0-mini", "maps", "can_bus"):
        if os.path.exists(os.path.join(mini_root, entry)):
            os.symlink(os.path.join(mini_root, entry), data_root / entry)

    # Use the temporary data root without persisting it in the .data-root.yaml (as data_root_getter would)
    monkeypatch.setattr("point_frustums.dataloaders.nuscenes.data_root_getter", lambda *args, **kwargs: str(data_root))
    cli = LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        run=False,
        args=[
            "--config", CONFIG,
            "--data.streaming_data_root=null",
            "--trainer.logger=false",
            "--trainer.accelerator=cpu",
        ],
    )  # fmt: skip
    datamodule = cli.datamodule
    db = datamodule.nuscenes_database

    # Partition the scenes into two parts and collect the lidar files of each
    scenes = sorted(scene["token"] for scene in db.scene)
    part_of_scene = {token: 1 + i % 2 for i, token in enumerate(scenes)}
    part_files = {1: [], 2: []}
    for record in db.sample_data:
        if record["channel"] == "LIDAR_TOP":
            part = part_of_scene[db.get("sample", record["sample_token"])["scene_token"]]
            part_files[part].append(record["filename"])

    def fetch_part(part: int) -> list[str]:
        files = []
        for filename in part_files[part]:
            os.makedirs(data_root / os.path.dirname(filename), exist_ok=True)
            os.symlink(os.path.join(mini_root, filename), data_root / filename)
            files.append(str(data_root / filename))
        return files

    output = str(tmp_path / "optimized")
    build_streaming.build(datamodule, output, parts=[1], num_workers=2, chunk_bytes="16MB", fetch_part=fetch_part)
    # The raw files of the part are removed again
    assert not (data_root / "samples").exists()
    n_part_1 = {
        split: num_streaming_items(os.path.join(output, split)) if os.path.exists(os.path.join(output, split)) else 0
        for split in ("train", "val")
    }

    assert n_part_1["train"] > 0
    # A restarted build skips the parts that are done
    fetched = []
    build_streaming.build(
        datamodule, output, parts=[1, 2], num_workers=2, chunk_bytes="16MB", fetch_part=lambda p: fetched.append(p) or fetch_part(p)
    )  # fmt: skip
    assert fetched == [2]
    with open(os.path.join(output, build_streaming.STATE_FILE), encoding="utf-8") as f:
        assert json.load(f) == {"parts_done": [1, 2]}

    for split in ("train", "val"):
        split_dir = os.path.join(output, split)
        dataset = datamodule.nuscenes_split(split)
        stored = read_target_counts(split_dir)
        # Every sample of the split is stored exactly once
        assert sorted(stored["sample_tokens"]) == sorted(dataset.sample_tokens)
        assert n_part_1[split] < len(dataset.sample_tokens)

        # The stored counts and tokens correspond to the items in the order of the litdata indices
        items = StreamingDataset(input_dir=split_dir)
        assert len(items) == len(dataset.sample_tokens)
        counts = count_targets_streaming(split_dir, n_classes=datamodule.dataset.annotations.n_classes)
        for i in range(len(items)):
            item = items[i]
            assert item["metadata"]["sample_token"] == stored["sample_tokens"][i]
            expected = torch.bincount(item["targets"]["class"].long(), minlength=counts.shape[1])
            assert torch.equal(counts[i], expected)
