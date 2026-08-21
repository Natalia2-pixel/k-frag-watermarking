import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from kfrag.data import CocoImageDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_IMAGE_DIRECTORY = REPOSITORY_ROOT / "data" / "raw" / "coco_val2017_100"


def _assert_dataset_contents(dataset: CocoImageDataset) -> None:
    # Reading every item verifies that every discovered image can be decoded.
    for item in dataset:
        assert item["image"].shape == (3, 256, 256)

    sample = dataset[0]
    assert sample["image"].dtype == torch.float32
    assert torch.isfinite(sample["image"]).all()
    assert sample["image"].min().item() >= 0.0
    assert sample["image"].max().item() <= 1.0
    assert sample["filename"]
    assert isinstance(sample["original_width"], int)
    assert isinstance(sample["original_height"], int)
    assert sample["original_width"] > 0
    assert sample["original_height"] > 0


def test_local_coco_images_load_end_to_end() -> None:
    configured_directory = os.environ.get("KFRAG_COCO_TEST_DIR")
    image_directory = (
        Path(configured_directory).expanduser()
        if configured_directory
        else OPTIONAL_IMAGE_DIRECTORY
    )
    if not image_directory.is_dir():
        pytest.skip(
            "COCO integration dataset unavailable; set KFRAG_COCO_TEST_DIR or "
            f"provide the optional fixture at {OPTIONAL_IMAGE_DIRECTORY}"
        )

    dataset = CocoImageDataset(image_directory)
    assert len(dataset) > 0
    _assert_dataset_contents(dataset)


def test_coco_image_loader_with_temporary_fixture(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    nested_directory = image_directory / "nested"
    nested_directory.mkdir(parents=True)
    Image.new("RGB", (17, 11), color=(255, 0, 127)).save(image_directory / "a.png")
    Image.new("L", (9, 13), color=64).save(nested_directory / "b.jpg")

    dataset = CocoImageDataset(image_directory)
    assert len(dataset) == 2
    assert dataset[0]["relative_path"] == "a.png"
    assert dataset[0]["original_width"] == 17
    assert dataset[0]["original_height"] == 11
    assert dataset[1]["relative_path"] == "nested/b.jpg"
    _assert_dataset_contents(dataset)

    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["image"].shape == (2, 3, 256, 256)
