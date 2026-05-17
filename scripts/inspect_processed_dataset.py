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
        print(f"Image path: {image_path}")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Image CRS: {src.crs}")
        print(f"Image transform: {src.transform}")
        print(f"Band descriptions: {src.descriptions}")

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        print(f"Mask path: {mask_path}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask unique values: {np.unique(mask)}")

    # image bands: B02, B03, B04, B08
    # RGB visualization: B04, B03, B02
    rgb = np.stack([image[2], image[1], image[0]], axis=-1)
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
