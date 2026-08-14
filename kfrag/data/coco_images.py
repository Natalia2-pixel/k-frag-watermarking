"""Image-only dataset for a local subset of COCO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class CocoImageDataset(Dataset[dict[str, Any]]):
    """Load supported images below a directory as 256x256 RGB tensors."""

    image_size = (256, 256)
    supported_suffixes = {".jpg", ".jpeg", ".png"}

    def __init__(self, image_directory: str | Path) -> None:
        self.image_directory = Path(image_directory)
        if not self.image_directory.is_dir():
            raise FileNotFoundError(
                f"COCO image directory does not exist: {self.image_directory}"
            )

        self.image_paths = sorted(
            (
                path
                for path in self.image_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in self.supported_suffixes
            ),
            key=lambda path: path.relative_to(self.image_directory).as_posix(),
        )
        if not self.image_paths:
            raise RuntimeError(
                f"No JPG, JPEG, or PNG images found in: {self.image_directory}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as source:
            original_width, original_height = source.size
            image = source.convert("RGB")
            image = TF.resize(image, self.image_size, antialias=True)
            tensor = TF.pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)

        return {
            "image": tensor,
            "filename": image_path.name,
            "relative_path": image_path.relative_to(self.image_directory).as_posix(),
            "original_width": original_width,
            "original_height": original_height,
        }
