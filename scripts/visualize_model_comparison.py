from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import random_split

from src.dataset import LandCoverNetDataset, LANDCOVERNET_CLASSES
from src.model_factory import create_model


CLASS_COLORS = np.array(
    [
        [0, 90, 200],      # 0 Water
        [190, 120, 70],    # 1 Artificial Bare Ground
        [220, 190, 120],   # 2 Natural Bare Ground
        [240, 240, 255],   # 3 Permanent Snow and Ice
        [20, 120, 40],     # 4 Woody Vegetation
        [180, 210, 80],    # 5 Cultivated Vegetation
        [120, 180, 90],    # 6 Natural Grassland
    ],
    dtype=np.float32,
) / 255.0

MASK_CMAP = ListedColormap(CLASS_COLORS)


def percentile_stretch(rgb: np.ndarray, lower: float = 2, upper: float = 98) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)

    for c in range(rgb.shape[-1]):
        band = rgb[..., c]
        p_low, p_high = np.percentile(band, [lower, upper])

        if p_high <= p_low:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)

    return out


def normalize_reflectance(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 10000.0
    image = np.clip(image, 0.0, 1.5)
    return image.astype(np.float32)


def read_image(path: Path, expected_channels: int) -> np.ndarray:
    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    if image.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, got {image.shape} for {path}"
        )

    return image


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.int64)

    # Remap original LandCoverNet labels:
    # original 0 -> 255 ignore
    # original 1..7 -> 0..6
    remapped = np.full_like(mask, fill_value=255, dtype=np.int64)

    for original_label in range(1, 8):
        remapped[mask == original_label] = original_label - 1

    return remapped


def make_rgb_from_13band(image13_raw: np.ndarray) -> np.ndarray:
    # torchgeo13 order:
    # B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
    rgb = np.stack(
        [
            image13_raw[3],  # B04 Red
            image13_raw[2],  # B03 Green
            image13_raw[1],  # B02 Blue
        ],
        axis=-1,
    )
    return percentile_stretch(rgb)


def load_checkpoint_model(
    checkpoint_path: Path,
    model_name: str,
    in_channels: int,
    num_classes: int = 7,
    base_features: int = 64,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    torchgeo_weights: str | None = None,
    device: torch.device | str = "cpu",
):
    model = create_model(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        base_features=base_features,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        torchgeo_weights=torchgeo_weights,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict(model, image_raw: np.ndarray, device: torch.device) -> np.ndarray:
    image = normalize_reflectance(image_raw)
    tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)

    logits = model(tensor)
    pred = torch.argmax(logits, dim=1).squeeze(0)

    return pred.detach().cpu().numpy().astype(np.int64)


def get_test_chip_ids(data_root_13: Path, seed: int, num_samples: int) -> list[str]:
    base_dataset = LandCoverNetDataset(
        root_dir=data_root_13,
        normalize=True,
        augment=False,
        random_crop_size=None,
        ignore_index=255,
        expected_channels=13,
    )

    dataset_size = len(base_dataset)

    train_size = int(np.round(dataset_size * 0.70))
    val_size = int(np.round(dataset_size * 0.15))
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    _, _, test_subset = random_split(
        base_dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator,
    )

    all_samples = base_dataset.samples
    test_chip_ids = [
        all_samples[idx][0].stem
        for idx in test_subset.indices
    ]

    rng = random.Random(seed)
    selected = rng.sample(test_chip_ids, k=min(num_samples, len(test_chip_ids)))

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root-10", type=Path, required=True)
    parser.add_argument("--data-root-13", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=8)

    parser.add_argument("--exp19-checkpoint", type=Path, required=True)
    parser.add_argument("--exp20-checkpoint", type=Path, required=True)
    parser.add_argument("--exp21-checkpoint", type=Path, required=True)
    parser.add_argument("--exp22-checkpoint", type=Path, required=True)
    parser.add_argument("--exp27-checkpoint", type=Path, required=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chip_ids = get_test_chip_ids(
        data_root_13=args.data_root_13,
        seed=args.seed,
        num_samples=args.num_samples,
    )

    print("Selected chip IDs:")
    for chip_id in chip_ids:
        print(" ", chip_id)

    models = {
        "ResUNet 13b\nexp22": load_checkpoint_model(
            checkpoint_path=args.exp22_checkpoint,
            model_name="resunet",
            in_channels=13,
            base_features=64,
            device=device,
        ),
        "DeepLabV3+ R50\nexp21": load_checkpoint_model(
            checkpoint_path=args.exp21_checkpoint,
            model_name="deeplabv3plus",
            in_channels=10,
            encoder_name="resnet50",
            encoder_weights="imagenet",
            device=device,
        ),
        "Unet++ EffB3\nexp19": load_checkpoint_model(
            checkpoint_path=args.exp19_checkpoint,
            model_name="unetplusplus",
            in_channels=10,
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            device=device,
        ),
        "FPN EffB3\nexp20": load_checkpoint_model(
            checkpoint_path=args.exp20_checkpoint,
            model_name="fpn",
            in_channels=10,
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            device=device,
        ),
        "TorchGeo DINO α\nexp27": load_checkpoint_model(
            checkpoint_path=args.exp27_checkpoint,
            model_name="torchgeo_resnet50_unet",
            in_channels=13,
            torchgeo_weights="sentinel2_all_dino",
            device=device,
        ),
    }

    column_titles = [
        "RGB",
        "Ground truth",
        *models.keys(),
    ]

    n_rows = len(chip_ids)
    n_cols = len(column_titles)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for row_idx, chip_id in enumerate(chip_ids):
        image10_path = args.data_root_10 / "images" / f"{chip_id}.tif"
        image13_path = args.data_root_13 / "images" / f"{chip_id}.tif"
        mask_path = args.data_root_13 / "masks" / f"{chip_id}.tif"

        image10_raw = read_image(image10_path, expected_channels=10)
        image13_raw = read_image(image13_path, expected_channels=13)
        mask = read_mask(mask_path)

        rgb = make_rgb_from_13band(image13_raw)

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_ylabel(chip_id, fontsize=9)

        axes[row_idx, 1].imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=6, interpolation="nearest")

        predictions = {
            "ResUNet 13b\nexp22": predict(models["ResUNet 13b\nexp22"], image13_raw, device),
            "DeepLabV3+ R50\nexp21": predict(models["DeepLabV3+ R50\nexp21"], image10_raw, device),
            "Unet++ EffB3\nexp19": predict(models["Unet++ EffB3\nexp19"], image10_raw, device),
            "FPN EffB3\nexp20": predict(models["FPN EffB3\nexp20"], image10_raw, device),
            "TorchGeo DINO α\nexp27": predict(models["TorchGeo DINO α\nexp27"], image13_raw, device),
        }

        for col_offset, model_name in enumerate(models.keys(), start=2):
            axes[row_idx, col_offset].imshow(
                predictions[model_name],
                cmap=MASK_CMAP,
                vmin=0,
                vmax=6,
                interpolation="nearest",
            )

        for col_idx in range(n_cols):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=10)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=CLASS_COLORS[i],
            markeredgecolor=CLASS_COLORS[i],
            markersize=8,
            label=f"{i}: {name}",
        )
        for i, name in enumerate(LANDCOVERNET_CLASSES)
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=200)
    print(f"Saved figure to: {args.output_path}")


if __name__ == "__main__":
    main()
