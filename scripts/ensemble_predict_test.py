from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.dataset import LandCoverNetDataset, LANDCOVERNET_CLASSES
from src.metrics import SegmentationMetricTracker, format_metrics
from src.model_factory import create_model


NUM_CLASSES = 7
IGNORE_INDEX = 255


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
    class_weights: tuple[float, ...] | None = None


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
        class_weights=(
            0.8702,  # Water
            0.6957,  # Artificial Bare Ground
            0.1782,  # Natural Bare Ground
            0.2617,  # Permanent Snow and Ice
            0.6713,  # Woody Vegetation
            0.6834,  # Cultivated Vegetation
            0.4396,  # Natural Grassland
        ),
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
        class_weights=(
            0.8681,  # Water
            0.6559,  # Artificial Bare Ground
            0.0798,  # Natural Bare Ground
            0.3019,  # Permanent Snow and Ice
            0.6565,  # Woody Vegetation
            0.6853,  # Cultivated Vegetation
            0.4186,  # Natural Grassland
        ),
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
        class_weights=(
            0.8337,  # Water
            0.6865,  # Artificial Bare Ground
            0.0613,  # Natural Bare Ground
            0.3374,  # Permanent Snow and Ice
            0.6717,  # Woody Vegetation
            0.7207,  # Cultivated Vegetation
            0.4441,  # Natural Grassland
        ),
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
        class_weights=(
            0.7665,  # Water
            0.6666,  # Artificial Bare Ground
            0.1958,  # Natural Bare Ground
            0.0965,  # Permanent Snow and Ice
            0.6517,  # Woody Vegetation
            0.7462,  # Cultivated Vegetation
            0.4493,  # Natural Grassland
        ),
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
        class_weights=(
            0.8699,  # Water
            0.6838,  # Artificial Bare Ground
            0.1283,  # Natural Bare Ground
            0.3229,  # Permanent Snow and Ice
            0.6809,  # Woody Vegetation
            0.7353,  # Cultivated Vegetation
            0.4868,  # Natural Grassland
        ),
    ),
}


class PairedLandCoverNetTestDataset(Dataset):
    """
    Test subset that returns aligned 10-band and 13-band tensors for the same chip_id.

    It reproduces the same 70/15/15 split logic used in train.py/evaluate.py,
    using the 13-band dataset as the reference list of chip IDs.
    """

    def __init__(
        self,
        data_root_10: str | Path,
        data_root_13: str | Path,
        seed: int = 42,
        normalization_mode_10: str = "reflectance",
        normalization_mode_13: str = "reflectance",
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()

        self.dataset10 = LandCoverNetDataset(
            root_dir=data_root_10,
            normalize=normalization_mode_10 != "none",
            normalization_mode=normalization_mode_10,
            augment=False,
            random_crop_size=None,
            ignore_index=ignore_index,
            expected_channels=10,
        )

        self.dataset13 = LandCoverNetDataset(
            root_dir=data_root_13,
            normalize=normalization_mode_13 != "none",
            normalization_mode=normalization_mode_13,
            augment=False,
            random_crop_size=None,
            ignore_index=ignore_index,
            expected_channels=13,
        )

        self.index10_by_chip_id = {
            image_path.stem: idx
            for idx, (image_path, _mask_path) in enumerate(self.dataset10.samples)
        }

        self.index13_by_chip_id = {
            image_path.stem: idx
            for idx, (image_path, _mask_path) in enumerate(self.dataset13.samples)
        }

        dataset_size = len(self.dataset13)
        train_size = int(np.round(dataset_size * 0.70))
        val_size = int(np.round(dataset_size * 0.15))
        test_size = dataset_size - train_size - val_size

        generator = torch.Generator().manual_seed(seed)
        _train_subset, _val_subset, test_subset = random_split(
            self.dataset13,
            lengths=[train_size, val_size, test_size],
            generator=generator,
        )

        chip_ids: list[str] = []
        for idx in test_subset.indices:
            image13_path, _mask13_path = self.dataset13.samples[idx]
            chip_id = image13_path.stem

            if chip_id not in self.index10_by_chip_id:
                raise FileNotFoundError(
                    f"Chip {chip_id} exists in 13-band dataset but not in 10-band dataset."
                )

            chip_ids.append(chip_id)

        self.chip_ids = chip_ids

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, idx: int) -> dict[str, object]:
        chip_id = self.chip_ids[idx]

        image10, mask10 = self.dataset10[self.index10_by_chip_id[chip_id]]
        image13, mask13 = self.dataset13[self.index13_by_chip_id[chip_id]]

        # Masks should be the same because both processed datasets are derived
        # from the same LandCoverNet LC mask. Keep this check to catch data mismatch.
        if not torch.equal(mask10, mask13):
            raise ValueError(f"Mask mismatch between 10-band and 13-band data for {chip_id}")

        image13_path, mask13_path = self.dataset13.samples[self.index13_by_chip_id[chip_id]]

        return {
            "chip_id": chip_id,
            "image10": image10,
            "image13": image13,
            "mask": mask13,
            "reference_path": str(image13_path),
            "mask_path": str(mask13_path),
        }


def collate_paired_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "chip_id": [item["chip_id"] for item in batch],
        "image10": torch.stack([item["image10"] for item in batch]),
        "image13": torch.stack([item["image13"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "reference_path": [item["reference_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_encoder_weights(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).lower() in {"", "none", "null"}:
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
        encoder_weights=resolve_encoder_weights(config.encoder_weights),
        torchgeo_weights=config.torchgeo_weights,
        torchgeo_decoder_dropout=config.torchgeo_decoder_dropout,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def forward_model(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    logits = model(images)

    if isinstance(logits, dict):
        logits = logits["out"]
    elif isinstance(logits, tuple):
        logits = logits[0]

    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B, C, H, W], got {logits.shape}")

    return logits


def hard_voting(
    preds: dict[str, torch.Tensor],
    configs: dict[str, EnsembleModelConfig],
    mode: str,
) -> torch.Tensor:
    """
    mode:
        majority    - each model vote has weight 1. Ties are softly broken by model mIoU.
        weighted    - each model vote is weighted by model-level mIoU.
        class_aware - each model vote for class c is weighted by per-class IoU[c].
    """
    first_pred = next(iter(preds.values()))
    batch_size, height, width = first_pred.shape

    scores = torch.zeros(
        (batch_size, NUM_CLASSES, height, width),
        dtype=torch.float32,
        device=first_pred.device,
    )

    tie_break_scores = torch.zeros_like(scores)

    for model_key, pred in preds.items():
        config = configs[model_key]

        if mode == "majority":
            weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=pred.device)
        elif mode == "weighted":
            weights = torch.full(
                (NUM_CLASSES,),
                fill_value=float(config.model_weight),
                dtype=torch.float32,
                device=pred.device,
            )
        elif mode == "class_aware":
            if config.class_weights is None:
                raise ValueError(f"class_weights are missing for {model_key}")
            weights = torch.tensor(config.class_weights, dtype=torch.float32, device=pred.device)
        else:
            raise ValueError(f"Unknown voting mode: {mode}")

        # Used only to make majority-vote ties deterministic and quality-aware.
        model_weight = float(config.model_weight)

        for class_id in range(NUM_CLASSES):
            class_mask = (pred == class_id).float()
            scores[:, class_id] += class_mask * weights[class_id]
            tie_break_scores[:, class_id] += class_mask * model_weight

    if mode == "majority":
        scores = scores + tie_break_scores * 1e-6

    return scores.argmax(dim=1)


def soft_probability_averaging(
    logits_by_model: dict[str, torch.Tensor],
    configs: dict[str, EnsembleModelConfig],
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    avg_probs: torch.Tensor | None = None
    total_weight = 0.0

    for model_key, logits in logits_by_model.items():
        weight = float(configs[model_key].model_weight)
        probs = F.softmax(logits / temperature, dim=1)

        if avg_probs is None:
            avg_probs = probs * weight
        else:
            avg_probs += probs * weight

        total_weight += weight

    if avg_probs is None:
        raise RuntimeError("No logits were provided for soft_probability_averaging")

    avg_probs = avg_probs / total_weight
    return avg_probs.argmax(dim=1)


def train_ids_to_original_labels(pred: np.ndarray) -> np.ndarray:
    """Convert train IDs 0..6 back to original LandCoverNet labels 1..7."""
    out = np.zeros_like(pred, dtype=np.uint16)
    valid = (pred >= 0) & (pred < NUM_CLASSES)
    out[valid] = pred[valid].astype(np.uint16) + 1
    return out


def save_prediction_geotiff(
    pred: np.ndarray,
    reference_path: str | Path,
    output_path: Path,
    save_original_labels: bool = False,
) -> None:
    reference_path = Path(reference_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    if save_original_labels:
        pred_to_save = train_ids_to_original_labels(pred)
        dtype = "uint16"
        nodata = 0
    else:
        pred_to_save = pred.astype(np.uint8)
        dtype = "uint8"
        nodata = 255

    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_to_save, 1)


def save_confusion_matrix(
    confusion_matrix: torch.Tensor,
    output_path: Path,
    class_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target/pred", *class_names])

        for idx, class_name in enumerate(class_names):
            writer.writerow([class_name, *confusion_matrix[idx].tolist()])


def save_summary_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "pixel_accuracy",
        "mean_iou",
        "mean_dice",
        *[f"iou_{idx}_{name}" for idx, name in enumerate(LANDCOVERNET_CLASSES)],
        *[f"dice_{idx}_{name}" for idx, name in enumerate(LANDCOVERNET_CLASSES)],
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metrics_to_row(name: str, tracker: SegmentationMetricTracker) -> dict[str, object]:
    metrics = tracker.compute()

    row: dict[str, object] = {
        "name": name,
        "pixel_accuracy": metrics.pixel_accuracy,
        "mean_iou": metrics.mean_iou,
        "mean_dice": metrics.mean_dice,
    }

    for idx, class_name in enumerate(LANDCOVERNET_CLASSES):
        row[f"iou_{idx}_{class_name}"] = metrics.per_class_iou[idx]
        row[f"dice_{idx}_{class_name}"] = metrics.per_class_dice[idx]

    return row


@torch.no_grad()
def run_ensemble(args: argparse.Namespace) -> None:
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_configs: dict[str, EnsembleModelConfig] = {
        key: MODEL_CONFIGS[key]
        for key in args.models
    }

    print(f"Device: {device}")
    print(f"Models: {', '.join(selected_configs.keys())}")
    print(f"Output dir: {output_dir}")

    dataset = PairedLandCoverNetTestDataset(
        data_root_10=args.data_root_10,
        data_root_13=args.data_root_13,
        seed=args.seed,
        normalization_mode_10=args.normalization_mode_10,
        normalization_mode_13=args.normalization_mode_13,
        ignore_index=args.ignore_index,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_paired_batch,
    )

    print(f"Test samples: {len(dataset)}")

    models: dict[str, nn.Module] = {}

    for model_key, config in selected_configs.items():
        checkpoint_path = Path(getattr(args, config.checkpoint_arg))
        print(f"Loading {model_key}: {checkpoint_path}")
        models[model_key] = load_model(config, checkpoint_path, device)

    trackers: dict[str, SegmentationMetricTracker] = {}

    for model_key in selected_configs:
        trackers[model_key] = SegmentationMetricTracker(
            num_classes=NUM_CLASSES,
            ignore_index=args.ignore_index,
            device=device,
        )

    ensemble_names = ["majority", "weighted", "class_aware", "soft_avg"]
    for name in ensemble_names:
        trackers[name] = SegmentationMetricTracker(
            num_classes=NUM_CLASSES,
            ignore_index=args.ignore_index,
            device=device,
        )

    for batch in tqdm(loader, desc="Ensemble inference"):
        image10 = batch["image10"].to(device, non_blocking=True).float()
        image13 = batch["image13"].to(device, non_blocking=True).float()
        masks = batch["mask"].to(device, non_blocking=True).long()

        logits_by_model: dict[str, torch.Tensor] = {}
        preds_by_model: dict[str, torch.Tensor] = {}

        for model_key, model in models.items():
            config = selected_configs[model_key]
            images = image10 if config.data_key == "10" else image13

            logits = forward_model(model, images)
            pred = logits.argmax(dim=1)

            logits_by_model[model_key] = logits
            preds_by_model[model_key] = pred
            trackers[model_key].update_from_predictions(pred, masks)

        ensemble_preds = {
            "majority": hard_voting(preds_by_model, selected_configs, mode="majority"),
            "weighted": hard_voting(preds_by_model, selected_configs, mode="weighted"),
            "class_aware": hard_voting(preds_by_model, selected_configs, mode="class_aware"),
            "soft_avg": soft_probability_averaging(
                logits_by_model,
                selected_configs,
                temperature=args.temperature,
            ),
        }

        for name, pred in ensemble_preds.items():
            trackers[name].update_from_predictions(pred, masks)

        if args.save_predictions:
            chip_ids = batch["chip_id"]
            reference_paths = batch["reference_path"]

            # Save only ensemble outputs by default, not every individual model.
            for name, pred_batch in ensemble_preds.items():
                pred_np_batch = pred_batch.detach().cpu().numpy()

                for i, chip_id in enumerate(chip_ids):
                    output_path = output_dir / "predictions" / name / f"{chip_id}.tif"
                    save_prediction_geotiff(
                        pred=pred_np_batch[i],
                        reference_path=reference_paths[i],
                        output_path=output_path,
                        save_original_labels=args.save_original_labels,
                    )

    summary_rows: list[dict[str, object]] = []
    metrics_txt_path = output_dir / "ensemble_test_metrics.txt"

    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("Ensemble test results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Models: {', '.join(selected_configs.keys())}\n")
        f.write(f"Test samples: {len(dataset)}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Normalization 10-band: {args.normalization_mode_10}\n")
        f.write(f"Normalization 13-band: {args.normalization_mode_13}\n")
        f.write("\n")

        for name, tracker in trackers.items():
            metrics = tracker.compute()
            text = format_metrics(metrics, class_names=LANDCOVERNET_CLASSES)

            print("")
            print("=" * 80)
            print(name)
            print("=" * 80)
            print(text)

            f.write("=" * 80 + "\n")
            f.write(f"{name}\n")
            f.write("=" * 80 + "\n")
            f.write(text + "\n\n")

            summary_rows.append(metrics_to_row(name, tracker))

            save_confusion_matrix(
                confusion_matrix=tracker.get_confusion_matrix(),
                output_path=output_dir / "confusion_matrices" / f"{name}.csv",
                class_names=LANDCOVERNET_CLASSES,
            )

    save_summary_csv(summary_rows, output_dir / "ensemble_test_summary.csv")

    print("")
    print(f"Saved metrics: {metrics_txt_path}")
    print(f"Saved summary CSV: {output_dir / 'ensemble_test_summary.csv'}")
    print(f"Saved confusion matrices: {output_dir / 'confusion_matrices'}")

    if args.save_predictions:
        print(f"Saved prediction GeoTIFFs: {output_dir / 'predictions'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble predictions on aligned 10-band and 13-band LandCoverNet test split."
    )

    parser.add_argument("--data-root-10", type=Path, required=True)
    parser.add_argument("--data-root-13", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ensemble_test"))

    parser.add_argument(
        "--models",
        nargs="+",
        default=["exp22", "exp27"],
        choices=sorted(MODEL_CONFIGS.keys()),
        help="Models to include in the ensemble.",
    )

    parser.add_argument("--exp19-checkpoint", type=Path, default=None)
    parser.add_argument("--exp20-checkpoint", type=Path, default=None)
    parser.add_argument("--exp21-checkpoint", type=Path, default=None)
    parser.add_argument("--exp22-checkpoint", type=Path, default=None)
    parser.add_argument("--exp27-checkpoint", type=Path, default=None)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-index", type=int, default=IGNORE_INDEX)

    parser.add_argument(
        "--normalization-mode-10",
        type=str,
        default="reflectance",
        choices=["none", "reflectance", "torchgeo_s2"],
        help="Must match training normalization for 10-band models.",
    )
    parser.add_argument(
        "--normalization-mode-13",
        type=str,
        default="reflectance",
        choices=["none", "reflectance", "torchgeo_s2"],
        help="Must match training normalization for 13-band models.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for soft probability averaging. 1.0 means no scaling.",
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save ensemble prediction GeoTIFFs for each test chip.",
    )
    parser.add_argument(
        "--save-original-labels",
        action="store_true",
        help="When saving GeoTIFFs, save original LandCoverNet labels 1..7 instead of train IDs 0..6.",
    )

    args = parser.parse_args()

    for model_key in args.models:
        config = MODEL_CONFIGS[model_key]
        checkpoint_path = getattr(args, config.checkpoint_arg)
        if checkpoint_path is None:
            parser.error(f"--{config.checkpoint_arg.replace('_', '-')} is required when using {model_key}")

    return args


def main() -> None:
    args = parse_args()
    run_ensemble(args)


if __name__ == "__main__":
    main()
