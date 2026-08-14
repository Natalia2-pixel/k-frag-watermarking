from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kfrag.data import CocoImageDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRECTORY = REPOSITORY_ROOT / "data" / "raw" / "coco_val2017_100"


def test_local_coco_images_load_end_to_end() -> None:
    assert IMAGE_DIRECTORY.is_dir(), f"Missing local dataset: {IMAGE_DIRECTORY}"

    dataset = CocoImageDataset(IMAGE_DIRECTORY)
    assert len(dataset) == 100

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

    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
    assert batch["image"].shape == (4, 3, 256, 256)
