from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import create_train_val_test_datasets, LANDCOVERNET_CLASSES
from src.losses import ComboLoss
from src.metrics import SegmentationMetricTracker, format_metrics
from src.model import count_trainable_parameters
from src.model_factory import SUPPORTED_MODELS, create_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> tuple[float, object, torch.Tensor]:
    model.eval()

    total_loss = 0.0
    total_batches = 0

    metric_tracker = SegmentationMetricTracker(
        num_classes=num_classes,
        ignore_index=ignore_index,
        device=device,
    )

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_batches += 1

        metric_tracker.update_from_logits(logits, masks)

    avg_loss = total_loss / max(total_batches, 1)
    metrics = metric_tracker.compute()
    confusion_matrix = metric_tracker.get_confusion_matrix()

    return avg_loss, metrics, confusion_matrix


def save_confusion_matrix(
    confusion_matrix: torch.Tensor,
    output_path: Path,
    class_names: list[str] | None = None,
) -> None:
    num_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = LANDCOVERNET_CLASSES

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["target/pred"] + class_names)

        for idx in range(num_classes):
            row = [class_names[idx]] + confusion_matrix[idx].tolist()
            writer.writerow(row)


def save_test_summary(
    output_path: Path,
    test_loss: float,
    pixel_accuracy: float,
    mean_iou: float,
    mean_dice: float,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_loss",
                "test_pixel_accuracy",
                "test_mean_iou",
                "test_mean_dice",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "test_loss": test_loss,
                "test_pixel_accuracy": pixel_accuracy,
                "test_mean_iou": mean_iou,
                "test_mean_dice": mean_dice,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on processed LandCoverNet test set."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to processed dataset root directory with images/ and masks/.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, for example best_model.pth.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resunet",
        choices=SUPPORTED_MODELS,
        help="Segmentation model architecture. Must match the training run.",
    )

    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet34",
        help="Encoder name for segmentation_models_pytorch models.",
    )

    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Encoder weights for SMP models. Use 'imagenet' or 'none'.",
    )

    parser.add_argument(
        "--torchgeo-weights",
        type=str,
        default="sentinel2_all_dino",
        help=(
            "TorchGeo ResNet50 pretrained weights. Examples: "
            "sentinel2_all_dino, sentinel2_all_moco, sentinel2_all_seco_eco, none."
        ),
    )

    parser.add_argument(
        "--torchgeo-decoder-dropout",
        type=float,
        default=0.1,
        help="Dropout used in the TorchGeo ResNet50 U-Net decoder.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Must match the seed used during training split.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--in-channels",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--normalization-mode",
        type=str,
        default="reflectance",
        choices=["none", "reflectance", "torchgeo_s2"],
        help=(
            "Input normalization. Must match the training run. "
            "Use 'torchgeo_s2' for TorchGeo Sentinel-2 stats normalization."
        ),
    )

    parser.add_argument(
        "--base-features",
        type=int,
        default=32,
        help="Base number of features for the first U-Net layer. Default: 32.",
    )

    parser.add_argument(
        "--ignore-index",
        type=int,
        default=255,
    )

    parser.add_argument(
        "--focal-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--use-class-alpha",
        action="store_true",
        help="Use the same class-frequency alpha weights as in train.py.",
    )

    return parser.parse_args()


def build_class_alpha(device: torch.device) -> torch.Tensor:
    """
    Same alpha construction as in train.py.

    Original classes:
        1..7

    Training classes:
        0..6
    """
    class_frequencies = torch.tensor(
        [
            0.046550,  # original class 1 -> train class 0
            0.054580,  # original class 2 -> train class 1
            0.010020,  # original class 3 -> train class 2
            0.014138,  # original class 4 -> train class 3
            0.265378,  # original class 5 -> train class 4
            0.371394,  # original class 6 -> train class 5
            0.237867,  # original class 7 -> train class 6
        ],
        dtype=torch.float32,
        device=device,
    )

    alpha = 1.0 / torch.sqrt(class_frequencies)
    alpha = alpha / alpha.mean()

    return alpha


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
        root_dir=data_root,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        normalize=args.normalization_mode != "none",
        normalization_mode=args.normalization_mode,
        augment_train=False,
        random_crop_size=None,
        ignore_index=args.ignore_index,
        expected_channels=args.in_channels,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    encoder_weights = args.encoder_weights

    if encoder_weights.lower() in {"none", "null", ""}:
        encoder_weights = None

    model = create_model(
        model_name=args.model,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_features=args.base_features,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        torchgeo_weights=args.torchgeo_weights,
        torchgeo_decoder_dropout=args.torchgeo_decoder_dropout,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Model: {args.model}")
    print(f"Normalization mode: {args.normalization_mode}")

    if args.model in {"deeplabv3plus", "unetplusplus", "fpn"}:
        print(f"Encoder: {args.encoder_name}")
        print(f"Encoder weights: {encoder_weights}")

    if args.model == "torchgeo_resnet50_unet":
        print(f"TorchGeo weights: {args.torchgeo_weights}")
        print(f"TorchGeo decoder dropout: {args.torchgeo_decoder_dropout}")

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    checkpoint_best_miou = checkpoint.get("best_val_miou", "unknown")

    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"Checkpoint best val mIoU: {checkpoint_best_miou}")

    if args.use_class_alpha:
        alpha = build_class_alpha(device=device)
        print(f"Using class alpha: {alpha.detach().cpu().tolist()}")
    else:
        alpha = None
        print("Using class alpha: None")

    criterion = ComboLoss(
        alpha=alpha,
        gamma=args.gamma,
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        ignore_index=args.ignore_index,
    )

    test_loss, test_metrics, confusion_matrix = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )

    class_names = LANDCOVERNET_CLASSES

    metrics_text = format_metrics(
        metrics=test_metrics,
        class_names=class_names,
    )

    print("")
    print("=" * 80)
    print("Test results")
    print("=" * 80)
    print(f"Test loss: {test_loss:.4f}")
    print(metrics_text)

    metrics_txt_path = output_dir / "test_metrics.txt"
    summary_csv_path = output_dir / "test_summary.csv"
    confusion_csv_path = output_dir / "confusion_matrix.csv"

    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("Test results\n")
        f.write("=" * 80)
        f.write("\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Checkpoint epoch: {checkpoint_epoch}\n")
        f.write(f"Checkpoint best val mIoU: {checkpoint_best_miou}\n")
        f.write(f"Test loss: {test_loss:.6f}\n\n")
        f.write(metrics_text)
        f.write("\n")

    save_test_summary(
        output_path=summary_csv_path,
        test_loss=test_loss,
        pixel_accuracy=test_metrics.pixel_accuracy,
        mean_iou=test_metrics.mean_iou,
        mean_dice=test_metrics.mean_dice,
    )

    save_confusion_matrix(
        confusion_matrix=confusion_matrix,
        output_path=confusion_csv_path,
        class_names=class_names,
    )

    print("")
    print(f"Saved metrics:          {metrics_txt_path}")
    print(f"Saved summary CSV:      {summary_csv_path}")
    print(f"Saved confusion matrix: {confusion_csv_path}")


if __name__ == "__main__":
    main()
