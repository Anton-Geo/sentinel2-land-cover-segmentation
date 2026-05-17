import argparse
import csv
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio


BANDS = [
    "B02",  # Blue
    "B03",  # Green
    "B04",  # Red
    "B08"   # Near infrared
]

# Sentinel-2 SCL invalid classes:
# 0  = no data
# 1  = saturated / defective
# 3  = cloud shadow
# 8  = cloud medium probability
# 9  = cloud high probability
# 10 = thin cirrus
# 11 = snow / ice
BAD_SCL_VALUES = {0, 1, 3, 8, 9, 10, 11}


def find_chip_dirs(raw_dir: Path) -> list[Path]:
    """
    Finds chip directories in already downloaded LandCoverNet raw dataset.

    Expected:
    raw_dir / tile / chip
    e.g.
    D:/datasets/landcovernet_raw_2018/30SWH/00
    """
    chip_dirs = []

    for tile_dir in sorted(raw_dir.iterdir()):
        if not tile_dir.is_dir():
            continue

        for chip_dir in sorted(tile_dir.iterdir()):
            if not chip_dir.is_dir():
                continue

            if (chip_dir / "S2").exists():
                chip_dirs.append(chip_dir)

    return chip_dirs


def get_chip_id(chip_dir: Path) -> str:
    return f"{chip_dir.parent.name}_{chip_dir.name}"


def find_label_path(chip_dir: Path) -> Path | None:
    chip_id = get_chip_id(chip_dir)

    exact = chip_dir / f"{chip_id}_2018_LC_10m.tif"
    if exact.exists():
        return exact

    matches = sorted(chip_dir.glob("*_LC_10m.tif"))
    if matches:
        return matches[0]

    return None


def find_scene_dirs(chip_dir: Path, months: set[str]) -> list[Path]:
    s2_dir = chip_dir / "S2"

    if not s2_dir.exists():
        return []

    scene_dirs = []

    for scene_dir in sorted(s2_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        # Example scene name: 30SWH_00_20180713
        date_part = scene_dir.name.split("_")[-1]

        if len(date_part) != 8 or not date_part.isdigit():
            continue

        month = date_part[4:6]

        if month in months:
            scene_dirs.append(scene_dir)

    return scene_dirs


def find_band_path(scene_dir: Path, band: str) -> Path | None:
    matches = sorted(scene_dir.glob(f"*_{band}_10m.tif"))
    return matches[0] if matches else None


def read_single_band(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    return arr, profile


def read_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def build_valid_mask(
    scene_dir: Path,
    use_scl: bool,
    cld_threshold: float | None,
) -> tuple[np.ndarray | None, str]:
    """
    Returns valid_mask HxW.
    True means valid pixel.

    If neither SCL nor CLD is available/used, returns None.
    """
    valid_mask = None
    mask_source = []

    if use_scl:
        scl_path = find_band_path(scene_dir, "SCL")

        if scl_path is not None:
            scl = read_mask(scl_path)
            scl_valid = ~np.isin(scl, list(BAD_SCL_VALUES))
            valid_mask = scl_valid
            mask_source.append("SCL")

    if cld_threshold is not None:
        cld_path = find_band_path(scene_dir, "CLD")

        if cld_path is not None:
            cld = read_mask(cld_path).astype(np.float32)
            cld_valid = cld <= cld_threshold

            if valid_mask is None:
                valid_mask = cld_valid
            else:
                valid_mask = valid_mask & cld_valid

            mask_source.append(f"CLD<={cld_threshold}")

    if not mask_source:
        return None, "none"

    return valid_mask, "+".join(mask_source)


def compute_nanmedian_with_fallback(scene_stacks: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """
    scene_stacks: list of arrays C x H x W, with NaNs for invalid/cloud pixels.

    Returns:
    composite: C x H x W
    valid_fraction: fraction of pixels valid in all 4 bands after median.
    """
    time_stack = np.stack(scene_stacks, axis=0)  # T x C x H x W

    with np.errstate(all="ignore"):
        composite = np.nanmedian(time_stack, axis=0)

    valid_pixels = np.isfinite(composite).all(axis=0)
    valid_fraction = float(valid_pixels.mean())

    if np.isnan(composite).any():
        # Fallback: ordinary median over the original scene stack, ignoring masks only if all were NaN.
        # Since invalid pixels were replaced by NaN, this fallback fills remaining NaNs with 0 later.
        composite = np.nan_to_num(composite, nan=0.0)

    return composite.astype(np.float32), valid_fraction


def write_image(output_path: Path, image: np.ndarray, reference_profile: dict) -> None:
    """
    image: C x H x W
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = reference_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": image.shape[0],
            "dtype": "float32",
            "compress": "deflate",
            "predictor": 2,
            "nodata": None,
        }
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(image)

        for idx, band_name in enumerate(BANDS, start=1):
            dst.set_band_description(idx, band_name)


def copy_mask(label_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy is enough; keeps original dtype, CRS, transform, etc.
    shutil.copy2(label_path, output_path)


def process_chip(
    chip_dir: Path,
    output_dir: Path,
    months: set[str],
    min_scenes: int,
    use_scl: bool,
    cld_threshold: float | None,
    min_valid_fraction: float,
    overwrite: bool,
) -> dict:
    chip_id = get_chip_id(chip_dir)

    image_output_path = output_dir / "images" / f"{chip_id}.tif"
    mask_output_path = output_dir / "masks" / f"{chip_id}.tif"

    result = {
        "chip_id": chip_id,
        "status": "unknown",
        "num_scenes_found": 0,
        "num_scenes_used": 0,
        "valid_fraction": 0.0,
        "mask_source": "",
        "image_path": str(image_output_path),
        "mask_path": str(mask_output_path),
        "message": "",
    }

    if image_output_path.exists() and mask_output_path.exists() and not overwrite:
        result["status"] = "exists"
        result["message"] = "already_processed"
        return result

    label_path = find_label_path(chip_dir)

    if label_path is None:
        result["status"] = "skipped"
        result["message"] = "label_not_found"
        return result

    scene_dirs = find_scene_dirs(chip_dir, months)
    result["num_scenes_found"] = len(scene_dirs)

    if not scene_dirs:
        result["status"] = "skipped"
        result["message"] = "no_summer_scenes"
        return result

    scene_stacks = []
    reference_profile = None
    mask_sources = set()

    for scene_dir in scene_dirs:
        band_paths = [find_band_path(scene_dir, band) for band in BANDS]

        if any(path is None for path in band_paths):
            continue

        band_arrays = []
        scene_profile = None

        for band_path in band_paths:
            arr, profile = read_single_band(band_path)
            band_arrays.append(arr)
            scene_profile = profile

        stack = np.stack(band_arrays, axis=0).astype(np.float32)

        valid_mask, mask_source = build_valid_mask(
            scene_dir=scene_dir,
            use_scl=use_scl,
            cld_threshold=cld_threshold,
        )

        mask_sources.add(mask_source)

        if valid_mask is not None:
            stack[:, ~valid_mask] = np.nan

        scene_stacks.append(stack)

        if reference_profile is None:
            reference_profile = scene_profile

    result["num_scenes_used"] = len(scene_stacks)
    result["mask_source"] = ";".join(sorted(mask_sources))

    if len(scene_stacks) < min_scenes:
        result["status"] = "skipped"
        result["message"] = f"not_enough_complete_scenes_min_{min_scenes}"
        return result

    assert reference_profile is not None

    composite, valid_fraction = compute_nanmedian_with_fallback(scene_stacks)
    result["valid_fraction"] = valid_fraction

    if valid_fraction < min_valid_fraction:
        result["status"] = "skipped"
        result["message"] = f"valid_fraction_too_low_{valid_fraction:.3f}"
        return result

    write_image(image_output_path, composite, reference_profile)
    copy_mask(label_path, mask_output_path)

    result["status"] = "ok"
    result["message"] = "ok"

    return result


def write_metadata(output_dir: Path, results: list[dict]) -> None:
    metadata_path = output_dir / "metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "chip_id",
        "status",
        "num_scenes_found",
        "num_scenes_used",
        "valid_fraction",
        "mask_source",
        "image_path",
        "mask_path",
        "message",
    ]

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nMetadata saved to: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local LandCoverNet Sentinel-2 summer median composites."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Already downloaded raw LandCoverNet directory, e.g. D:/datasets/landcovernet_raw_2018",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output processed dataset directory.",
    )

    parser.add_argument(
        "--months",
        nargs="+",
        default=["06", "07", "08"],
        help="Months used for median composite. Default: 06 07 08.",
    )

    parser.add_argument(
        "--min-scenes",
        type=int,
        default=2,
        help="Minimum number of complete S2 scenes required per chip.",
    )

    parser.add_argument(
        "--no-scl-mask",
        action="store_true",
        help="Disable SCL cloud/shadow/snow masking.",
    )

    parser.add_argument(
        "--cld-threshold",
        type=float,
        default=None,
        help="Optional CLD threshold. Example: 60. If omitted, CLD is not used.",
    )

    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.50,
        help="Minimum fraction of pixels valid after masking. Default: 0.50.",
    )

    parser.add_argument(
        "--max-chips",
        type=int,
        default=None,
        help="Process only first N chips for testing.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel processes.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already processed chips.",
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir

    chip_dirs = find_chip_dirs(raw_dir)

    if args.max_chips is not None:
        chip_dirs = chip_dirs[: args.max_chips]

    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found chip directories: {len(chip_dirs)}")
    print(f"Months: {args.months}")
    print(f"Min scenes: {args.min_scenes}")
    print(f"Use SCL mask: {not args.no_scl_mask}")
    print(f"CLD threshold: {args.cld_threshold}")
    print(f"Min valid fraction: {args.min_valid_fraction}")
    print(f"Workers: {args.num_workers}")
    print(f"Overwrite: {args.overwrite}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_chip,
                chip_dir,
                output_dir,
                set(args.months),
                args.min_scenes,
                not args.no_scl_mask,
                args.cld_threshold,
                args.min_valid_fraction,
                args.overwrite,
            )
            for chip_dir in chip_dirs
        ]

        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)

            print(
                f"[{idx}/{len(futures)}] "
                f"{result['chip_id']} | "
                f"{result['status']} | "
                f"found: {result['num_scenes_found']} | "
                f"used: {result['num_scenes_used']} | "
                f"valid: {result['valid_fraction']:.3f} | "
                f"{result['message']}"
            )

    results = sorted(results, key=lambda x: x["chip_id"])
    write_metadata(output_dir, results)

    ok_count = sum(1 for r in results if r["status"] == "ok")
    exists_count = sum(1 for r in results if r["status"] == "exists")
    skipped_count = len(results) - ok_count - exists_count

    print()
    print("Done.")
    print(f"OK: {ok_count}")
    print(f"Already existed: {exists_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
