from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, random_split


LANDCOVERNET_CLASSES = [
    "Water",
    "Artificial Bare Ground",
    "Natural Bare Ground",
    "Permanent Snow and Ice",
    "Woody Vegetation",
    "Cultivated Vegetation",
    "Natural Grassland",
]

ORIGINAL_TO_TRAIN_LABEL = {
    0: 255,  # ignore / no data
    1: 0,    # Water
    2: 1,    # Artificial Bare Ground
    3: 2,    # Natural Bare Ground
    4: 3,    # Permanent Snow and Ice
    5: 4,    # Woody Vegetation
    6: 5,    # Cultivated Vegetation
    7: 6,    # Natural Grassland
}

NORMALIZATION_MODES = ["none", "reflectance", "torchgeo_s2"]

# Sentinel-2 13-band order used by TorchGeo SENTINEL2_ALL_* ResNet50 weights:
# B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12.
# Values are in raw reflectance-like digital numbers, not divided by 10000.
TORCHGEO_S2_MEAN = np.array(
    [
        1612.9052, 1397.6073, 1322.2919, 1373.3869, 1561.2764,
        2108.4822, 2390.7429, 2318.7560, 2581.6467, 837.2778,
        22.2936, 2195.5127, 1537.6821,
    ],
    dtype=np.float32,
)

TORCHGEO_S2_STD = np.array(
    [
        791.9904, 854.8830, 878.2342, 1144.6632, 1127.9778,
        1164.8842, 1276.8750, 1249.5778, 1345.5264, 577.3161,
        47.9234, 1340.4779, 1142.0638,
    ],
    dtype=np.float32,
)


class LandCoverNetDataset(Dataset):
    """
    Dataset for processed LandCoverNet Sentinel-2 median composites.

    Expected structure:

    root_dir/
    ├── images/
    │   ├── chip_id.tif
    │   └── ...
    └── masks/
        ├── chip_id.tif
        └── ...

    Image:
        GeoTIFF with shape [C, H, W].

    Mask:
        GeoTIFF with shape [H, W].
        original labels: 0 = ignore / no data, 1..7 = valid land cover classes
        training labels: 255 = ignore_index, 0..6 = valid classes
    """

    def __init__(
        self,
        root_dir: str | Path,
        normalize: bool = True,
        normalization_mode: str = "reflectance",
        reflectance_scale: float = 10000.0,
        clip_max: float = 1.5,
        augment: bool = False,
        random_crop_size: int | None = None,
        ignore_index: int = 255,
        expected_channels: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"

        if normalization_mode not in NORMALIZATION_MODES:
            raise ValueError(
                f"Unsupported normalization_mode={normalization_mode}. "
                f"Supported values: {NORMALIZATION_MODES}"
            )

        if not normalize:
            normalization_mode = "none"

        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.reflectance_scale = reflectance_scale
        self.clip_max = clip_max
        self.augment = augment
        self.random_crop_size = random_crop_size
        self.ignore_index = ignore_index
        self.expected_channels = expected_channels

        self.image_paths = sorted(self.images_dir.glob("*.tif"))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .tif images found in {self.images_dir}")

        self.samples: list[tuple[Path, Path]] = []

        for image_path in self.image_paths:
            mask_path = self.masks_dir / image_path.name

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image {image_path.name}: {mask_path}"
                )

            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]

        image = self._read_image(image_path)  # [C, H, W], float32
        mask = self._read_mask(mask_path)     # [H, W], int64

        mask = self._remap_mask(mask)

        if self.normalize:
            image = self._normalize_image(image)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor

    def _read_image(self, image_path: Path) -> np.ndarray:
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)  # [C, H, W]

        if image.ndim != 3:
            raise ValueError(f"Expected image shape [C, H, W], got {image.shape}")

        if self.expected_channels is not None and image.shape[0] != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels}-channel Sentinel-2 image, "
                f"got shape {image.shape}"
            )

        return image

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # [H, W]

        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape [H, W], got {mask.shape}")

        return mask

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Original LandCoverNet labels:
            0 = ignore / no data
            1 = Water
            2 = Artificial Bare Ground
            3 = Natural Bare Ground
            4 = Permanent Snow and Ice
            5 = Woody Vegetation
            6 = Cultivated Vegetation
            7 = Natural Grassland

        Training labels:
            255 = ignore_index
            0..6 = valid classes
        """
        remapped = np.full_like(mask, fill_value=self.ignore_index, dtype=np.int64)

        for original_label, train_label in ORIGINAL_TO_TRAIN_LABEL.items():
            if original_label == 0:
                continue

            remapped[mask == original_label] = train_label

        return remapped

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image according to the selected mode."""
        if self.normalization_mode == "none":
            return image.astype(np.float32)

        if self.normalization_mode == "reflectance":
            image = image / self.reflectance_scale
            image = np.clip(image, 0.0, self.clip_max)
            return image.astype(np.float32)

        if self.normalization_mode == "torchgeo_s2":
            if image.shape[0] != 13:
                raise ValueError(
                    "normalization_mode='torchgeo_s2' expects 13 Sentinel-2 channels "
                    f"in TorchGeo order, got image shape {image.shape}."
                )

            mean = TORCHGEO_S2_MEAN[:, None, None]
            std = TORCHGEO_S2_STD[:, None, None]
            image = (image - mean) / std
            return image.astype(np.float32)

        raise ValueError(f"Unsupported normalization_mode: {self.normalization_mode}")

    def _augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply simple joint augmentations to image and mask.

        image: [C, H, W]
        mask:  [H, W]
        """

        # Horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, axis=2)
            mask = np.flip(mask, axis=1)

        # Vertical flip
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=0)

        # Rotate by 0, 90, 180, or 270 degrees
        if random.random() < 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k=k, axes=(1, 2))
            mask = np.rot90(mask, k=k, axes=(0, 1))

        # Random crop
        if self.random_crop_size is not None:
            image, mask = self._random_crop(image, mask, self.random_crop_size)

        # Brightness / radiometric scaling
        # Applied only to image, not to mask.
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            image = image * scale

        # Small additive shift
        if random.random() < 0.3:
            shift = random.uniform(-0.03, 0.03)
            image = image + shift

            image = np.clip(image, 0.0, self.clip_max)

        # Important: np.flip / np.rot90 may create arrays with negative strides
        # torch.from_numpy does not support negative strides
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        return image, mask

    def _random_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        crop_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, height, width = image.shape

        if crop_size > height or crop_size > width:
            raise ValueError(
                f"crop_size={crop_size} is larger than image size {(height, width)}"
            )

        if crop_size == height and crop_size == width:
            return image, mask

        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)

        image = image[:, top : top + crop_size, left : left + crop_size]
        mask = mask[top : top + crop_size, left : left + crop_size]

        return image, mask


def create_train_val_test_datasets(
    root_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    normalize: bool = True,
    normalization_mode: str = "reflectance",
    augment_train: bool = True,
    random_crop_size: int | None = None,
    ignore_index: int = 255,
    expected_channels: int | None = None,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Create train / validation / test splits.

    Train dataset may use augmentations.
    Validation and test datasets should not use augmentations.
    """

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1.0, got {total_ratio}"
        )

    base_dataset = LandCoverNetDataset(
        root_dir=root_dir,
        normalize=normalize,
        normalization_mode=normalization_mode,
        augment=False,
        random_crop_size=None,
        ignore_index=ignore_index,
        expected_channels=expected_channels,
    )

    dataset_size = len(base_dataset)

    train_size = int(np.round(dataset_size * train_ratio))
    val_size = int(np.round(dataset_size * val_ratio))
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset, test_subset = random_split(
        base_dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator,
    )

    # Create separate datasets with the same root but different augmentation settings
    train_dataset = LandCoverNetDataset(
        root_dir=root_dir,
        normalize=normalize,
        normalization_mode=normalization_mode,
        augment=augment_train,
        random_crop_size=random_crop_size,
        ignore_index=ignore_index,
        expected_channels=expected_channels,
    )

    val_dataset = LandCoverNetDataset(
        root_dir=root_dir,
        normalize=normalize,
        normalization_mode=normalization_mode,
        augment=False,
        random_crop_size=None,
        ignore_index=ignore_index,
        expected_channels=expected_channels,
    )

    test_dataset = LandCoverNetDataset(
        root_dir=root_dir,
        normalize=normalize,
        normalization_mode=normalization_mode,
        augment=False,
        random_crop_size=None,
        ignore_index=ignore_index,
        expected_channels=expected_channels,
    )

    # Reuse the same indices from random_split
    train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_subset.indices)

    return train_dataset, val_dataset, test_dataset
