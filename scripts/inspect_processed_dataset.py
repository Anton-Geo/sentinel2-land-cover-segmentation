import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def percentile_stretch(image: np.ndarray, lower: float = 2, upper: float = 98) -> np.ndarray:
    image = image.astype(np.float32)
    out = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[-1]):
        band = image[..., c]
        p_low, p_high = np.percentile(band, [lower, upper])

        if p_high <= p_low:
            out[..., c] = 0
        else:
            out[..., c] = np.clip((band - p_low) / (p_high - p_low), 0, 1)

    return out


def get_band_index(descriptions: tuple[str | None, ...], band_name: str, fallback: int | None = None) -> int:
    """Return zero-based band index from GeoTIFF band descriptions."""
    normalized = [desc.strip() if desc is not None else None for desc in descriptions]

    if band_name in normalized:
        return normalized.index(band_name)

    if fallback is not None:
        return fallback

    raise ValueError(
        f"Band {band_name} not found in descriptions: {descriptions}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--chip-id", type=str, default=None)
    args = parser.parse_args()

    image_dir = args.data_dir / "images"
    mask_dir = args.data_dir / "masks"

    image_paths = sorted(image_dir.glob("*.tif"))

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    if args.chip_id is None:
        image_path = image_paths[0]
        chip_id = image_path.stem
    else:
        chip_id = args.chip_id
        image_path = image_dir / f"{chip_id}.tif"

    mask_path = mask_dir / f"{chip_id}.tif"

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)
        descriptions = src.descriptions
        print(f"Image path: {image_path}")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Image CRS: {src.crs}")
        print(f"Image transform: {src.transform}")
        print(f"Band descriptions: {descriptions}")

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        print(f"Mask path: {mask_path}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask unique values: {np.unique(mask)}")

    # RGB visualization: B04, B03, B02. Use band descriptions so this works
    # for both the old 10-band order and the new TorchGeo 13-band order.
    red_idx = get_band_index(descriptions, "B04", fallback=2)
    green_idx = get_band_index(descriptions, "B03", fallback=1)
    blue_idx = get_band_index(descriptions, "B02", fallback=0)

    rgb = np.stack([image[red_idx], image[green_idx], image[blue_idx]], axis=-1)
    rgb_vis = percentile_stretch(rgb)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_vis)
    axes[0].set_title(f"Summer median RGB: {chip_id}")
    axes[0].axis("off")

    im = axes[1].imshow(mask, interpolation="nearest")
    axes[1].set_title("Land cover mask")
    axes[1].axis("off")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
