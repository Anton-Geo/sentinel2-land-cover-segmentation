import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    image_dir = args.data_dir / "images"
    mask_dir = args.data_dir / "masks"

    image_paths = sorted(image_dir.glob("*.tif"))

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    rows = []
    global_class_counts = {}
    global_band_min = None
    global_band_max = None
    nan_images = 0
    inf_images = 0
    missing_masks = 0

    for image_path in tqdm(image_paths):
        chip_id = image_path.stem
        mask_path = mask_dir / f"{chip_id}.tif"

        if not mask_path.exists():
            missing_masks += 1
            continue

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        has_nan = bool(np.isnan(image).any())
        has_inf = bool(np.isinf(image).any())

        if has_nan:
            nan_images += 1

        if has_inf:
            inf_images += 1

        band_min = image.reshape(image.shape[0], -1).min(axis=1)
        band_max = image.reshape(image.shape[0], -1).max(axis=1)

        if global_band_min is None:
            global_band_min = band_min.copy()
            global_band_max = band_max.copy()
        else:
            global_band_min = np.minimum(global_band_min, band_min)
            global_band_max = np.maximum(global_band_max, band_max)

        values, counts = np.unique(mask, return_counts=True)

        class_counts = dict(zip(values.tolist(), counts.tolist()))

        for value, count in class_counts.items():
            global_class_counts[value] = global_class_counts.get(value, 0) + count

        row = {
            "chip_id": chip_id,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "image_min": float(image.min()),
            "image_max": float(image.max()),
            "mask_values": " ".join(map(str, values.tolist())),
        }

        for value, count in class_counts.items():
            row[f"class_{value}_pixels"] = count

        rows.append(row)

    df = pd.DataFrame(rows)

    print()
    print(f"Images found: {len(image_paths)}")
    print(f"Missing masks: {missing_masks}")
    print(f"Images with NaN: {nan_images}")
    print(f"Images with Inf: {inf_images}")

    print()
    print("Global band min:")
    print(global_band_min)

    print("Global band max:")
    print(global_band_max)

    print()
    print("Global class counts:")
    for value in sorted(global_class_counts):
        print(f"  class {value}: {global_class_counts[value]}")

    total_pixels = sum(global_class_counts.values())

    print()
    print("Global class frequencies:")
    for value in sorted(global_class_counts):
        freq = global_class_counts[value] / total_pixels
        print(f"  class {value}: {freq:.6f}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved per-chip report to: {args.output_csv}")


if __name__ == "__main__":
    main()
