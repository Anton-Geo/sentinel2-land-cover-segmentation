from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


DEFAULT_BANDS = [
    "B02",  # Blue
    "B03",  # Green
    "B04",  # Red
    "B08",  # NIR
    "B05",  # Red Edge 1
    "B06",  # Red Edge 2
    "B07",  # Red Edge 3
    "B8A",  # Narrow NIR
    "B11",  # SWIR 1
    "B12",  # SWIR 2
]

# TorchGeo Sentinel-2 ALL ResNet50 weights expect 13 input channels.
# LandCoverNet v1.0 raw data does not include B10, because it is a cirrus band
# and is not useful for land surface analysis. For TorchGeo compatibility, the
# torchgeo13 preset inserts B10 as a zero-filled channel in the correct position.
TORCHGEO13_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",  # zero-filled synthetic band
    "B11",
    "B12",
]

BAND_PRESETS = {
    "10bands": DEFAULT_BANDS,
    "torchgeo13": TORCHGEO13_BANDS,
}

DEFAULT_MONTHS = [6, 7, 8]

# Sentinel-2 Scene Classification Layer classes to remove:
# 0  = No data
# 1  = Saturated / defective
# 3  = Cloud shadows
# 8  = Cloud medium probability
# 9  = Cloud high probability
# 10 = Thin cirrus
# 11 = Snow / ice
BAD_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare multi-band processed LandCoverNet dataset from raw 2018 data. "
            "For each chip, the script reads summer Sentinel-2 scenes, applies SCL "
            "cloud/shadow masking, computes per-pixel median composite, and saves "
            "a multi-band GeoTIFF image plus the corresponding LC mask."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Path to raw LandCoverNet 2018 directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed dataset.",
    )

    parser.add_argument(
        "--band-preset",
        type=str,
        default="10bands",
        choices=sorted(BAND_PRESETS.keys()),
        help=(
            "Band preset to use when --bands is not provided. "
            "'10bands' reproduces the previous 10-channel dataset. "
            "'torchgeo13' creates Sentinel-2 13-channel order with zero-filled B10."
        ),
    )

    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help=(
            "Custom Sentinel-2 bands to include. Overrides --band-preset. "
            "Use B10 only if you want a zero-filled synthetic B10 channel."
        ),
    )

    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        default=DEFAULT_MONTHS,
        help="Months to include in median composite, e.g. 6 7 8.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output image/mask files.",
    )

    parser.add_argument(
        "--debug-limit",
        type=int,
        default=None,
        help="Process only first N chips for debugging.",
    )

    return parser.parse_args()


def iter_chip_dirs(raw_dir: Path) -> list[Path]:
    """
    Expected raw structure:

        raw_dir/
        ├── 30SWH/
        │   ├── 00/
        │   ├── 01/
        │   └── ...
        ├── 30TYN/
        │   ├── 00/
        │   └── ...
        └── ...

    Returns chip directories like:

        raw_dir / "30SWH" / "00"
        raw_dir / "30SWH" / "01"
    """
    chip_dirs: list[Path] = []

    for tile_dir in sorted(raw_dir.iterdir()):
        if not tile_dir.is_dir():
            continue

        for chip_dir in sorted(tile_dir.iterdir()):
            if chip_dir.is_dir():
                chip_dirs.append(chip_dir)

    return chip_dirs


def get_chip_id(chip_dir: Path) -> str:
    """
    Convert:

        .../30SWH/00

    to:

        30SWH_00
    """
    tile_id = chip_dir.parent.name
    chip_number = chip_dir.name

    return f"{tile_id}_{chip_number}"


def extract_month_from_scene_dir(scene_dir: Path) -> int | None:
    """
    Expected scene folder pattern:

        30SWH_00_20180104
        30SWH_00_20180623
        30SWH_00_20180812

    Extract month from YYYYMMDD part.
    """
    parts = scene_dir.name.split("_")

    if not parts:
        return None

    date_part = parts[-1]

    if len(date_part) != 8 or not date_part.isdigit():
        return None

    year = int(date_part[:4])
    month = int(date_part[4:6])

    if year != 2018:
        return None

    if not 1 <= month <= 12:
        return None

    return month


def list_summer_scene_dirs(
    chip_dir: Path,
    months: set[int],
) -> list[Path]:
    s2_dir = chip_dir / "S2"

    if not s2_dir.exists():
        return []

    selected_scene_dirs: list[Path] = []

    for scene_dir in sorted(s2_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        month = extract_month_from_scene_dir(scene_dir)

        if month in months:
            selected_scene_dirs.append(scene_dir)

    return selected_scene_dirs


def find_file_without_extension(path: Path) -> Path | None:
    """
    LandCoverNet files can be opened by rasterio even if the file extension
    is omitted. Still, we also check common GeoTIFF suffixes.
    """
    if path.exists():
        return path

    for suffix in [".tif", ".tiff"]:
        candidate = path.with_suffix(suffix)

        if candidate.exists():
            return candidate

    return None


def find_band_file(
    scene_dir: Path,
    chip_id: str,
    band: str,
) -> Path | None:
    """
    Expected file pattern:

        <chip_id>_<YYYYMMDD>_<band>_10m

    Example:

        30SWH_00_20180104_B02_10m
        30SWH_00_20180104_B8A_10m
    """
    date_part = scene_dir.name.split("_")[-1]

    expected_path = scene_dir / f"{chip_id}_{date_part}_{band}_10m"

    found = find_file_without_extension(expected_path)

    if found is not None:
        return found

    matches = sorted(scene_dir.glob(f"*_{band}_10m*"))

    if matches:
        return matches[0]

    return None


def find_scl_file(
    scene_dir: Path,
    chip_id: str,
) -> Path | None:
    """
    Expected SCL file pattern:

        <chip_id>_<YYYYMMDD>_SCL_10m
    """
    date_part = scene_dir.name.split("_")[-1]

    expected_path = scene_dir / f"{chip_id}_{date_part}_SCL_10m"

    found = find_file_without_extension(expected_path)

    if found is not None:
        return found

    matches = sorted(scene_dir.glob("*_SCL_10m*"))

    if matches:
        return matches[0]

    return None


def find_mask_file(
    chip_dir: Path,
    chip_id: str,
) -> Path | None:
    """
    Expected LC mask file patterns:

        <chip_id>_2018_LC_10m
        <chip_id>_LC_10m

    The first band of this file is the label mask.
    """
    expected_names = [
        f"{chip_id}_2018_LC_10m",
        f"{chip_id}_LC_10m",
    ]

    for expected_name in expected_names:
        expected_path = chip_dir / expected_name

        found = find_file_without_extension(expected_path)

        if found is not None:
            return found

    matches = sorted(chip_dir.glob("*LC_10m*"))

    if matches:
        return matches[0]

    return None


def read_raster_band(
    source_path: Path,
    output_dtype: np.dtype,
) -> np.ndarray:
    with rasterio.open(source_path) as src:
        array = src.read(1).astype(output_dtype)

    return array


def read_reference_profile(reference_band_path: Path) -> dict:
    with rasterio.open(reference_band_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=256,
        width=256,
        count=1,
        dtype="float32",
        nodata=None,
        compress="deflate",
    )

    return profile


def read_scene_stack(
    scene_dir: Path,
    chip_id: str,
    bands: list[str],
) -> tuple[np.ndarray, dict] | None:
    """
    Read one Sentinel-2 scene.

    Returns:
        stack: [C, 256, 256]
        reference_profile

    If any required band or SCL is missing, returns None.
    """
    reference_band_path = find_band_file(
        scene_dir=scene_dir,
        chip_id=chip_id,
        band="B02",
    )

    scl_path = find_scl_file(
        scene_dir=scene_dir,
        chip_id=chip_id,
    )

    if reference_band_path is None or scl_path is None:
        return None

    reference_profile = read_reference_profile(reference_band_path)

    scl = read_raster_band(
        source_path=scl_path,
        output_dtype=np.uint8,
    )

    if scl.shape != (256, 256):
        raise ValueError(
            f"SCL must be 256x256, got {scl.shape}. "
            f"File: {scl_path}"
        )

    valid_mask = ~np.isin(scl, list(BAD_SCL_CLASSES))

    scene_bands: list[np.ndarray] = []

    for band in bands:
        if band == "B10":
            # LandCoverNet raw data does not contain B10. Keep the channel in the
            # correct Sentinel-2 position for TorchGeo pretrained weights by
            # inserting a zero-filled synthetic band.
            scene_bands.append(np.zeros_like(scl, dtype=np.float32))
            continue

        band_path = find_band_file(
            scene_dir=scene_dir,
            chip_id=chip_id,
            band=band,
        )

        if band_path is None:
            return None

        band_array = read_raster_band(
            source_path=band_path,
            output_dtype=np.float32,
        )

        if band_array.shape != (256, 256):
            raise ValueError(
                f"Band {band} must be 256x256, got {band_array.shape}. "
                f"File: {band_path}"
            )

        if band_array.shape != scl.shape:
            raise ValueError(
                f"Shape mismatch in scene {scene_dir}. "
                f"Band {band} shape: {band_array.shape}, "
                f"SCL shape: {scl.shape}"
            )

        band_array[~valid_mask] = np.nan
        scene_bands.append(band_array)

    stack = np.stack(scene_bands, axis=0).astype(np.float32)

    if stack.shape != (len(bands), 256, 256):
        raise ValueError(
            f"Invalid scene stack shape for {scene_dir}: {stack.shape}"
        )

    return stack, reference_profile


def compute_median_composite(
    scene_dirs: list[Path],
    chip_id: str,
    bands: list[str],
) -> tuple[np.ndarray, dict, int] | None:
    """
    Compute summer median composite.

    Input:
        scene_dirs: list of selected Sentinel-2 scene folders

    Output:
        median image: [C, 256, 256]
        reference profile
        number of scenes used
    """
    scene_stacks: list[np.ndarray] = []
    output_profile: dict | None = None

    for scene_dir in scene_dirs:
        scene_result = read_scene_stack(
            scene_dir=scene_dir,
            chip_id=chip_id,
            bands=bands,
        )

        if scene_result is None:
            continue

        stack, reference_profile = scene_result

        scene_stacks.append(stack)

        if output_profile is None:
            output_profile = reference_profile

    if not scene_stacks or output_profile is None:
        return None

    time_stack = np.stack(scene_stacks, axis=0)  # [T, C, H, W]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        with np.errstate(all="ignore"):
            median = np.nanmedian(time_stack, axis=0)  # [C, H, W]

    median = np.nan_to_num(
        median,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if median.shape != (len(bands), 256, 256):
        raise ValueError(
            f"Invalid median shape for {chip_id}: {median.shape}"
        )

    return median, output_profile, len(scene_stacks)


def save_image(
    output_path: Path,
    image: np.ndarray,
    reference_profile: dict,
    bands: list[str],
) -> None:
    """
    Save multi-band image GeoTIFF.

    Image shape:
        [C, 256, 256]
    """
    if image.shape != (len(bands), 256, 256):
        raise ValueError(
            f"Invalid image shape before saving: {image.shape}"
        )

    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        height=256,
        width=256,
        count=image.shape[0],
        dtype="float32",
        nodata=None,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(image)

        for band_index, band_name in enumerate(bands, start=1):
            dst.set_band_description(band_index, band_name)


def save_mask(
    source_mask_path: Path,
    output_mask_path: Path,
    reference_profile: dict,
) -> None:
    """
    Save LC mask as single-band GeoTIFF.

    The source LC file may contain:
        band 1 = label
        band 2 = consensus score

    We save only band 1 for segmentation training.
    """
    mask = read_raster_band(
        source_path=source_mask_path,
        output_dtype=np.uint16,
    )

    if mask.shape != (256, 256):
        raise ValueError(
            f"Mask must be 256x256, got {mask.shape}. "
            f"File: {source_mask_path}"
        )

    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        height=256,
        width=256,
        count=1,
        dtype="uint16",
        nodata=None,
        compress="deflate",
    )

    with rasterio.open(output_mask_path, "w", **profile) as dst:
        dst.write(mask, 1)


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"

    if args.bands is not None:
        bands = list(args.bands)
        band_preset = "custom"
    else:
        bands = list(BAND_PRESETS[args.band_preset])
        band_preset = args.band_preset

    months = set(args.months)

    print(f"Raw dir: {raw_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Band preset: {band_preset}")
    print(f"Bands: {bands}")
    print(f"Months: {sorted(months)}")

    chip_dirs = iter_chip_dirs(raw_dir)

    if args.debug_limit is not None:
        chip_dirs = chip_dirs[: args.debug_limit]

    print(f"Found chip dirs: {len(chip_dirs)}")

    rows: list[dict[str, str | int]] = []

    processed = 0
    skipped = 0

    skip_no_s2 = 0
    skip_no_summer_scenes = 0
    skip_no_mask = 0
    skip_no_composite = 0

    for chip_dir in tqdm(chip_dirs):
        chip_id = get_chip_id(chip_dir)

        output_image_path = images_dir / f"{chip_id}.tif"
        output_mask_path = masks_dir / f"{chip_id}.tif"

        if (
            output_image_path.exists()
            and output_mask_path.exists()
            and not args.overwrite
        ):
            processed += 1

            rows.append(
                {
                    "chip_id": chip_id,
                    "image_path": str(output_image_path),
                    "mask_path": str(output_mask_path),
                    "num_summer_scenes_total": -1,
                    "num_summer_scenes_used": -1,
                    "band_preset": band_preset,
                    "bands": ",".join(bands),
                    "status": "already_exists",
                }
            )

            continue

        s2_dir = chip_dir / "S2"

        if not s2_dir.exists():
            skipped += 1
            skip_no_s2 += 1
            continue

        scene_dirs = list_summer_scene_dirs(
            chip_dir=chip_dir,
            months=months,
        )

        if not scene_dirs:
            skipped += 1
            skip_no_summer_scenes += 1
            continue

        mask_path = find_mask_file(
            chip_dir=chip_dir,
            chip_id=chip_id,
        )

        if mask_path is None:
            skipped += 1
            skip_no_mask += 1
            continue

        composite_result = compute_median_composite(
            scene_dirs=scene_dirs,
            chip_id=chip_id,
            bands=bands,
        )

        if composite_result is None:
            skipped += 1
            skip_no_composite += 1
            continue

        image, reference_profile, used_scene_count = composite_result

        save_image(
            output_path=output_image_path,
            image=image,
            reference_profile=reference_profile,
            bands=bands,
        )

        save_mask(
            source_mask_path=mask_path,
            output_mask_path=output_mask_path,
            reference_profile=reference_profile,
        )

        rows.append(
            {
                "chip_id": chip_id,
                "image_path": str(output_image_path),
                "mask_path": str(output_mask_path),
                "num_summer_scenes_total": len(scene_dirs),
                "num_summer_scenes_used": used_scene_count,
                "band_preset": band_preset,
                "bands": ",".join(bands),
                "status": "processed",
            }
        )

        processed += 1

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "chip_id",
            "image_path",
            "mask_path",
            "num_summer_scenes_total",
            "num_summer_scenes_used",
            "band_preset",
            "bands",
            "status",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print("Done.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")

    print("")
    print("Skip reasons:")
    print(f"  no S2 directory:       {skip_no_s2}")
    print(f"  no summer scenes:      {skip_no_summer_scenes}")
    print(f"  no mask:               {skip_no_mask}")
    print(f"  no valid composite:    {skip_no_composite}")

    print("")
    print(f"Metadata:  {metadata_path}")


if __name__ == "__main__":
    main()
