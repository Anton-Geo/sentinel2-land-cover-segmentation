from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.model_factory import create_model


NUM_CLASSES = 7
PRED_NODATA = 255
UNCERTAINTY_NODATA = -9999.0
EPS = 1e-8


@dataclass(frozen=True)
class EnsembleModelConfig:
    key: str
    title: str
    checkpoint_arg: str
    model_name: str
    in_channels: int
    data_key: str  # "10" or "13"
    base_features: int = 32
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    torchgeo_weights: str | None = "sentinel2_all_dino"
    torchgeo_decoder_dropout: float = 0.1
    model_weight: float = 1.0


@dataclass(frozen=True)
class EnsembleInferenceOutput:
    """Batch-level outputs from weighted soft-probability ensemble inference."""

    preds: torch.Tensor  # [B,H,W], uint8 class ids
    predictive_entropy: torch.Tensor | None = None  # [B,H,W], float32, normalized 0..1
    expected_entropy: torch.Tensor | None = None  # [B,H,W], float32, normalized 0..1
    mutual_information: torch.Tensor | None = None  # [B,H,W], float32, normalized 0..1
    confidence: torch.Tensor | None = None  # [B,H,W], float32, max averaged probability


# Model-level weights are test mIoU values from the evaluation stage.
# The best test-set strategy was weighted softmax probability averaging.
MODEL_CONFIGS: dict[str, EnsembleModelConfig] = {
    "exp22": EnsembleModelConfig(
        key="exp22",
        title="ResUNet 13-band exp22",
        checkpoint_arg="exp22_checkpoint",
        model_name="resunet",
        in_channels=13,
        data_key="13",
        base_features=64,
        encoder_weights=None,
        torchgeo_weights=None,
        model_weight=0.5429,
    ),
    "exp27": EnsembleModelConfig(
        key="exp27",
        title="TorchGeo DINO alpha exp27",
        checkpoint_arg="exp27_checkpoint",
        model_name="torchgeo_resnet50_unet",
        in_channels=13,
        data_key="13",
        encoder_weights=None,
        torchgeo_weights="sentinel2_all_dino",
        torchgeo_decoder_dropout=0.1,
        model_weight=0.5583,
    ),
    "exp19": EnsembleModelConfig(
        key="exp19",
        title="Unet++ EfficientNet-B3 exp19",
        checkpoint_arg="exp19_checkpoint",
        model_name="unetplusplus",
        in_channels=10,
        data_key="10",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        torchgeo_weights=None,
        model_weight=0.5365,
    ),
    "exp21": EnsembleModelConfig(
        key="exp21",
        title="DeepLabV3+ ResNet50 exp21",
        checkpoint_arg="exp21_checkpoint",
        model_name="deeplabv3plus",
        in_channels=10,
        data_key="10",
        encoder_name="resnet50",
        encoder_weights="imagenet",
        torchgeo_weights=None,
        model_weight=0.5237,
    ),
    "exp20": EnsembleModelConfig(
        key="exp20",
        title="FPN EfficientNet-B3 exp20",
        checkpoint_arg="exp20_checkpoint",
        model_name="fpn",
        in_channels=10,
        data_key="10",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        torchgeo_weights=None,
        model_weight=0.5104,
    ),
}


class PairedRealChipsDataset(Dataset):
    """
    Dataset for real Sentinel-2 chips without ground-truth masks.

    Expected structure:
        root_10/images/*.tif
        root_13/images/*.tif

    The same chip filenames must exist in both directories.
    Values are expected to be raw Sentinel-2 reflectance-like digital numbers.
    This dataset applies the same reflectance normalization used during training:
        image = clip(image / 10000, 0, 1.5)
    """

    def __init__(
        self,
        root_10: str | Path,
        root_13: str | Path,
        reflectance_scale: float = 10000.0,
        clip_max: float = 1.5,
    ) -> None:
        self.root_10 = Path(root_10)
        self.root_13 = Path(root_13)
        self.images_10_dir = self.root_10 / "images"
        self.images_13_dir = self.root_13 / "images"
        self.reflectance_scale = reflectance_scale
        self.clip_max = clip_max

        if not self.images_10_dir.exists():
            raise FileNotFoundError(f"10-band images directory not found: {self.images_10_dir}")
        if not self.images_13_dir.exists():
            raise FileNotFoundError(f"13-band images directory not found: {self.images_13_dir}")

        paths_10 = {path.name: path for path in sorted(self.images_10_dir.glob("*.tif"))}
        paths_13 = {path.name: path for path in sorted(self.images_13_dir.glob("*.tif"))}

        common_names = sorted(set(paths_10) & set(paths_13))
        missing_13 = sorted(set(paths_10) - set(paths_13))
        missing_10 = sorted(set(paths_13) - set(paths_10))

        if not common_names:
            raise RuntimeError(
                f"No paired .tif chips found in {self.images_10_dir} and {self.images_13_dir}"
            )

        if missing_13:
            print(f"Warning: {len(missing_13)} 10-band chips have no 13-band pair.")
        if missing_10:
            print(f"Warning: {len(missing_10)} 13-band chips have no 10-band pair.")

        self.samples = [(name, paths_10[name], paths_13[name]) for name in common_names]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name, path_10, path_13 = self.samples[idx]
        image_10 = self._read_and_normalize(path_10, expected_channels=10)
        image_13 = self._read_and_normalize(path_13, expected_channels=13)
        return torch.from_numpy(image_10).float(), torch.from_numpy(image_13).float(), name

    def _read_and_normalize(self, path: Path, expected_channels: int) -> np.ndarray:
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)

        if image.ndim != 3 or image.shape[0] != expected_channels:
            raise ValueError(
                f"Expected image with {expected_channels} channels [C,H,W], got {image.shape}: {path}"
            )

        image = image / self.reflectance_scale
        image = np.clip(image, 0.0, self.clip_max)
        return image.astype(np.float32)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def load_model(config: EnsembleModelConfig, checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {config.key}: {checkpoint_path}")

    model = create_model(
        model_name=config.model_name,
        in_channels=config.in_channels,
        num_classes=NUM_CLASSES,
        base_features=config.base_features,
        encoder_name=config.encoder_name,
        encoder_weights=normalize_optional_string(config.encoder_weights),
        torchgeo_weights=normalize_optional_string(config.torchgeo_weights),
        torchgeo_decoder_dropout=config.torchgeo_decoder_dropout,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _validate_excluded_classes(exclude_classes: list[int] | None) -> list[int]:
    excluded = sorted(set(exclude_classes or []))
    for class_id in excluded:
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(
                f"Invalid excluded class id: {class_id}. Valid class ids are 0..{NUM_CLASSES - 1}."
            )
    if len(excluded) >= NUM_CLASSES:
        raise ValueError("Cannot exclude all classes.")
    return excluded


def suppress_classes_and_renormalize(probs: torch.Tensor, exclude_classes: list[int] | None) -> torch.Tensor:
    """
    Suppress unwanted classes and renormalize probabilities over remaining classes.

    This is useful when, for example, class 3 = Permanent Snow and Ice is treated as
    impossible for summer Lithuania inference. Entropy is then computed over the
    remaining valid class space rather than over a distribution whose probabilities
    no longer sum to 1.
    """
    excluded = _validate_excluded_classes(exclude_classes)
    if not excluded:
        return probs

    probs = probs.clone()
    probs[:, excluded, :, :] = 0.0
    probs_sum = probs.sum(dim=1, keepdim=True).clamp_min(EPS)
    return probs / probs_sum


def normalized_entropy(probs: torch.Tensor, num_active_classes: int) -> torch.Tensor:
    """
    Compute normalized categorical entropy for [B,C,H,W] probabilities.

    Returns [B,H,W] in approximately [0,1], where 0 means a near-certain class and
    1 means probability mass is close to uniformly spread over active classes.
    """
    entropy = -(probs.clamp_min(EPS) * probs.clamp_min(EPS).log()).sum(dim=1)
    norm = math.log(float(num_active_classes))
    return entropy / max(norm, EPS)


@torch.no_grad()
def predict_soft_average(
    models: dict[str, nn.Module],
    images_10: torch.Tensor,
    images_13: torch.Tensor,
    selected_model_keys: list[str],
    device: torch.device,
    exclude_classes: list[int] | None = None,
    compute_uncertainty: bool = False,
) -> EnsembleInferenceOutput:
    """
    Run weighted soft-probability ensemble inference.

    If compute_uncertainty=True, also compute three normalized uncertainty maps:
        predictive_entropy = H[weighted-average probabilities]
        expected_entropy   = weighted average of H[individual model probabilities]
        mutual_information = predictive_entropy - expected_entropy

    With weighted ensembles, expected entropy is weighted with the same model weights
    used for soft probability averaging.
    """
    images_10 = images_10.to(device, non_blocking=True)
    images_13 = images_13.to(device, non_blocking=True)

    excluded = _validate_excluded_classes(exclude_classes)
    num_active_classes = NUM_CLASSES - len(excluded)

    avg_probs: torch.Tensor | None = None
    expected_entropy_sum: torch.Tensor | None = None
    total_weight = 0.0

    for key in selected_model_keys:
        config = MODEL_CONFIGS[key]
        model = models[key]
        images = images_10 if config.data_key == "10" else images_13

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        probs = suppress_classes_and_renormalize(probs, excluded)
        weight = float(config.model_weight)

        if avg_probs is None:
            avg_probs = probs * weight
        else:
            avg_probs += probs * weight

        if compute_uncertainty:
            model_entropy = normalized_entropy(probs, num_active_classes)
            if expected_entropy_sum is None:
                expected_entropy_sum = model_entropy * weight
            else:
                expected_entropy_sum += model_entropy * weight

        total_weight += weight

    if avg_probs is None or total_weight <= 0:
        raise RuntimeError("No ensemble probabilities were computed.")

    avg_probs = avg_probs / total_weight
    preds = avg_probs.argmax(dim=1).to(torch.uint8)  # [B,H,W]

    if not compute_uncertainty:
        return EnsembleInferenceOutput(preds=preds.detach().cpu())

    if expected_entropy_sum is None:
        raise RuntimeError("Expected entropy was not computed.")

    predictive_entropy = normalized_entropy(avg_probs, num_active_classes)
    expected_entropy = expected_entropy_sum / total_weight
    mutual_information = (predictive_entropy - expected_entropy).clamp_min(0.0)
    confidence = avg_probs.max(dim=1).values

    return EnsembleInferenceOutput(
        preds=preds.detach().cpu(),
        predictive_entropy=predictive_entropy.detach().cpu().float(),
        expected_entropy=expected_entropy.detach().cpu().float(),
        mutual_information=mutual_information.detach().cpu().float(),
        confidence=confidence.detach().cpu().float(),
    )


def _single_band_profile(
    reference_path: Path,
    array: np.ndarray,
    dtype: str,
    nodata: int | float,
) -> dict:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    # Avoid copying incompatible multi-band/chip tiling metadata blindly.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    profile.update(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    return profile


def save_prediction_chip(pred: np.ndarray, reference_path: Path, output_path: Path) -> None:
    """
    Save one predicted mask as single-band GeoTIFF using the 13-band chip as reference.
    Prediction labels are 0..6, nodata is 255.
    """
    profile = _single_band_profile(reference_path, pred, dtype="uint8", nodata=PRED_NODATA)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)
        dst.set_band_description(1, "land_cover_prediction")


def save_float32_chip(array: np.ndarray, reference_path: Path, output_path: Path, band_description: str) -> None:
    """Save one float32 uncertainty/confidence map as a single-band GeoTIFF."""
    array = array.astype(np.float32)
    profile = _single_band_profile(reference_path, array, dtype="float32", nodata=UNCERTAINTY_NODATA)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, band_description)


def create_single_band_mosaic(
    input_dir: Path,
    output_path: Path,
    band_description: str,
    dtype: str,
    nodata: int | float,
) -> None:
    """Merge single-band chip GeoTIFFs into one mosaic GeoTIFF."""
    from rasterio.merge import merge

    paths = sorted(input_dir.glob("*.tif"))
    if not paths:
        raise RuntimeError(f"No chips found in {input_dir}")

    srcs = [rasterio.open(path) for path in paths]
    try:
        mosaic, transform = merge(srcs, method="first", nodata=nodata)
        profile = srcs[0].profile.copy()
    finally:
        for src in srcs:
            src.close()

    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        count=1,
        dtype=dtype,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic.astype(dtype))
        dst.set_band_description(1, band_description)

    print(f"Saved mosaic: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run weighted soft-average ensemble on real Sentinel-2 10-band/13-band chips."
    )

    parser.add_argument(
        "--chips-root-10",
        type=str,
        required=True,
        help="Root directory with 10-band chips, containing images/*.tif.",
    )
    parser.add_argument(
        "--chips-root-13",
        type=str,
        required=True,
        help="Root directory with 13-band chips, containing images/*.tif.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for prediction chips, uncertainty chips, and optional mosaics.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["exp22", "exp27", "exp19", "exp21", "exp20"],
        choices=sorted(MODEL_CONFIGS.keys()),
        help="Models to include in the soft-average ensemble.",
    )

    parser.add_argument("--exp19-checkpoint", type=str, default=None)
    parser.add_argument("--exp20-checkpoint", type=str, default=None)
    parser.add_argument("--exp21-checkpoint", type=str, default=None)
    parser.add_argument("--exp22-checkpoint", type=str, default=None)
    parser.add_argument("--exp27-checkpoint", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--reflectance-scale", type=float, default=10000.0)
    parser.add_argument("--clip-max", type=float, default=1.5)
    parser.add_argument(
        "--exclude-classes",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Class IDs to suppress before argmax and uncertainty calculation. "
            "For summer Sentinel-2 inference in Lithuania, use '--exclude-classes 3' "
            "to remove snow/ice from the active class space."
        ),
    )

    parser.add_argument(
        "--save-uncertainty",
        action="store_true",
        help=(
            "Save normalized predictive entropy, expected entropy, mutual information, "
            "and confidence chips in addition to class predictions."
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prediction chips that already exist.",
    )
    parser.add_argument(
        "--make-mosaic",
        action="store_true",
        help="Merge prediction chips into a single GeoTIFF mosaic after inference.",
    )
    parser.add_argument(
        "--make-uncertainty-mosaics",
        action="store_true",
        help="Merge uncertainty/confidence chips into single GeoTIFF mosaics after inference.",
    )
    parser.add_argument(
        "--mosaic-output",
        type=str,
        default=None,
        help="Optional explicit mosaic output path. If omitted, saved inside output-dir.",
    )

    return parser.parse_args()


def required_output_paths(
    name: str,
    predictions_dir: Path,
    uncertainty_dirs: dict[str, Path],
    save_uncertainty: bool,
) -> list[Path]:
    paths = [predictions_dir / name]
    if save_uncertainty:
        paths.extend(directory / name for directory in uncertainty_dirs.values())
    return paths


@torch.no_grad()
def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "prediction_chips"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    uncertainty_dirs = {
        "predictive_entropy": output_dir / "predictive_entropy_chips",
        "expected_entropy": output_dir / "expected_entropy_chips",
        "mutual_information": output_dir / "mutual_information_chips",
        "confidence": output_dir / "confidence_chips",
    }
    if args.save_uncertainty:
        for directory in uncertainty_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"Device: {device}")
    print(f"Models: {', '.join(args.models)}")
    print(f"10-band chips root: {args.chips_root_10}")
    print(f"13-band chips root: {args.chips_root_13}")
    print(f"Output dir: {output_dir}")
    print(f"Prediction chips dir: {predictions_dir}")
    print(f"Save uncertainty: {args.save_uncertainty}")
    print(f"Excluded classes: {args.exclude_classes}")

    dataset = PairedRealChipsDataset(
        root_10=args.chips_root_10,
        root_13=args.chips_root_13,
        reflectance_scale=args.reflectance_scale,
        clip_max=args.clip_max,
    )
    print(f"Paired chips: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    models: dict[str, nn.Module] = {}
    for key in args.models:
        config = MODEL_CONFIGS[key]
        checkpoint_value = getattr(args, config.checkpoint_arg)
        if checkpoint_value is None:
            raise ValueError(
                f"Checkpoint argument is required for selected model {key}: "
                f"--{config.checkpoint_arg.replace('_', '-')}"
            )

        checkpoint_path = Path(checkpoint_value)
        print(f"Loading {key}: {checkpoint_path}")
        models[key] = load_model(config, checkpoint_path, device)

    # Used for georeferencing output predictions.
    images_13_dir = Path(args.chips_root_13) / "images"

    for images_10, images_13, names in tqdm(loader, desc="Real ensemble inference"):
        # Avoid unnecessary compute if all outputs already exist.
        if args.skip_existing:
            existing_flags = [
                all(
                    path.exists()
                    for path in required_output_paths(
                        name, predictions_dir, uncertainty_dirs, args.save_uncertainty
                    )
                )
                for name in names
            ]
            if all(existing_flags):
                continue

        outputs = predict_soft_average(
            models=models,
            images_10=images_10,
            images_13=images_13,
            selected_model_keys=list(args.models),
            device=device,
            exclude_classes=list(args.exclude_classes),
            compute_uncertainty=args.save_uncertainty,
        )

        for batch_idx, name in enumerate(names):
            output_path = predictions_dir / name
            reference_path = images_13_dir / name

            if not (args.skip_existing and output_path.exists()):
                pred_np = outputs.preds[batch_idx].numpy().astype(np.uint8)
                save_prediction_chip(pred_np, reference_path, output_path)

            if args.save_uncertainty:
                metric_tensors = {
                    "predictive_entropy": outputs.predictive_entropy,
                    "expected_entropy": outputs.expected_entropy,
                    "mutual_information": outputs.mutual_information,
                    "confidence": outputs.confidence,
                }
                band_descriptions = {
                    "predictive_entropy": "normalized_predictive_entropy_total_uncertainty",
                    "expected_entropy": "normalized_expected_entropy_aleatoric_proxy",
                    "mutual_information": "normalized_mutual_information_epistemic_proxy",
                    "confidence": "max_ensemble_probability_confidence",
                }

                for metric_name, tensor in metric_tensors.items():
                    if tensor is None:
                        raise RuntimeError(f"Missing tensor for metric: {metric_name}")

                    metric_output_path = uncertainty_dirs[metric_name] / name
                    if args.skip_existing and metric_output_path.exists():
                        continue

                    metric_np = tensor[batch_idx].numpy().astype(np.float32)
                    save_float32_chip(
                        metric_np,
                        reference_path,
                        metric_output_path,
                        band_description=band_descriptions[metric_name],
                    )

    print(f"Saved prediction chips: {len(list(predictions_dir.glob('*.tif')))}")
    if args.save_uncertainty:
        for metric_name, directory in uncertainty_dirs.items():
            print(f"Saved {metric_name} chips: {len(list(directory.glob('*.tif')))}")

    if args.make_mosaic:
        if args.mosaic_output is not None:
            mosaic_output = Path(args.mosaic_output)
        else:
            mosaic_output = output_dir / "landcover_ensemble_softavg_mosaic.tif"

        create_single_band_mosaic(
            predictions_dir,
            mosaic_output,
            band_description="land_cover_prediction",
            dtype="uint8",
            nodata=PRED_NODATA,
        )

    if args.save_uncertainty and args.make_uncertainty_mosaics:
        create_single_band_mosaic(
            uncertainty_dirs["predictive_entropy"],
            output_dir / "predictive_entropy_mosaic.tif",
            band_description="normalized_predictive_entropy_total_uncertainty",
            dtype="float32",
            nodata=UNCERTAINTY_NODATA,
        )
        create_single_band_mosaic(
            uncertainty_dirs["expected_entropy"],
            output_dir / "expected_entropy_mosaic.tif",
            band_description="normalized_expected_entropy_aleatoric_proxy",
            dtype="float32",
            nodata=UNCERTAINTY_NODATA,
        )
        create_single_band_mosaic(
            uncertainty_dirs["mutual_information"],
            output_dir / "mutual_information_mosaic.tif",
            band_description="normalized_mutual_information_epistemic_proxy",
            dtype="float32",
            nodata=UNCERTAINTY_NODATA,
        )
        create_single_band_mosaic(
            uncertainty_dirs["confidence"],
            output_dir / "confidence_mosaic.tif",
            band_description="max_ensemble_probability_confidence",
            dtype="float32",
            nodata=UNCERTAINTY_NODATA,
        )


if __name__ == "__main__":
    main()
