from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import create_train_val_test_datasets, LANDCOVERNET_CLASSES
from src.losses import ComboLoss
from src.metrics import SegmentationMetricTracker, format_metrics
from src.model import count_trainable_parameters
from src.model_factory import SUPPORTED_MODELS, create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More reproducible, but may be slightly slower
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    keep_encoder_frozen: bool = False,
) -> tuple[float, float, float, float]:
    model.train()

    # If the encoder is frozen, keep its BatchNorm layers in eval mode as well.
    # This avoids updating pretrained running statistics during decoder warm-up.
    if keep_encoder_frozen and hasattr(model, "encoder"):
        model.encoder.eval()

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

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        metric_tracker.update_from_logits(logits.detach(), masks)

    avg_loss = total_loss / max(total_batches, 1)
    metrics = metric_tracker.compute()

    return (
        avg_loss,
        metrics.pixel_accuracy,
        metrics.mean_iou,
        metrics.mean_dice,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    class_names: list[str] | None = None,
) -> tuple[float, float, float, float, str]:
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

    metrics_text = format_metrics(metrics, class_names=class_names)

    return (
        avg_loss,
        metrics.pixel_accuracy,
        metrics.mean_iou,
        metrics.mean_dice,
        metrics_text,
    )


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_miou: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_miou": best_val_miou,
        "args": vars(args),
    }

    torch.save(checkpoint, output_path)


def append_history_row(
    history_path: Path,
    row: dict[str, float | int | str],
) -> None:
    file_exists = history_path.exists()

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_pixel_acc",
                "train_miou",
                "train_mdice",
                "val_loss",
                "val_pixel_acc",
                "val_miou",
                "val_mdice",
                "lr",
                "encoder_lr",
                "decoder_lr",
                "encoder_frozen",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train segmentation models on processed LandCoverNet dataset."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to processed dataset root directory with images/ and masks/.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/residual_unet",
        help="Directory for checkpoints and logs.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resunet",
        choices=SUPPORTED_MODELS,
        help="Segmentation model architecture.",
    )

    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet34",
        help=(
            "Encoder name for segmentation_models_pytorch models, "
            "for example resnet34, resnet50, efficientnet-b3."
        ),
    )

    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help=(
            "Encoder weights for segmentation_models_pytorch models. "
            "Use 'imagenet' for pretrained weights or 'none' for random init."
        ),
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
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Main learning rate. For TorchGeo models this is used for decoder/head parameters.",
    )

    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=None,
        help=(
            "Optional encoder learning rate for TorchGeo pretrained models. "
            "If omitted, the encoder uses --learning-rate. Recommended: 1e-4."
        ),
    )

    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help=(
            "For TorchGeo pretrained models, freeze encoder parameters for the first N epochs. "
            "Use 0 to disable."
        ),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
            "Input normalization. 'reflectance' divides by 10000 and clips to [0, 1.5]. "
            "'torchgeo_s2' applies 13-band Sentinel-2 mean/std normalization for TorchGeo weights."
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
        "--no-augment",
        action="store_true",
        help="Disable train augmentations.",
    )

    parser.add_argument(
        "--random-crop-size",
        type=int,
        default=None,
        help="Optional random crop size for train dataset, e.g. 224.",
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
        help="Use class-frequency-based alpha weights for FocalLoss.",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop training if val mIoU does not improve for N epochs.",
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
    )

    return parser.parse_args()


def build_class_alpha(device: torch.device) -> torch.Tensor:
    """
    Class frequencies from processed dataset statistics.

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



def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    if not hasattr(model, "encoder"):
        return

    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def get_optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result: dict[str, float] = {}

    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{idx}"))
        result[name] = float(group["lr"])

    return result


def create_optimizer(
    model: nn.Module,
    model_name: str,
    learning_rate: float,
    encoder_learning_rate: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if model_name == "torchgeo_resnet50_unet" and hasattr(model, "encoder"):
        encoder_lr = learning_rate if encoder_learning_rate is None else encoder_learning_rate

        encoder_params = []
        decoder_params = []

        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(parameter)
            else:
                decoder_params.append(parameter)

        return torch.optim.AdamW(
            [
                {
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                    "name": "encoder",
                },
                {
                    "params": decoder_params,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                    "name": "decoder",
                },
            ]
        )

    return torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "all",
            }
        ]
    )

def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.csv"
    best_checkpoint_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_model.pth"

    device = get_device()

    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")

    train_dataset, val_dataset, test_dataset = create_train_val_test_datasets(
        root_dir=data_root,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        normalize=args.normalization_mode != "none",
        normalization_mode=args.normalization_mode,
        augment_train=not args.no_augment,
        random_crop_size=args.random_crop_size,
        ignore_index=args.ignore_index,
        expected_channels=args.in_channels,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
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

    model = model.to(device)

    print(f"Model: {args.model}")
    print(f"Normalization mode: {args.normalization_mode}")

    if args.model == "resunet":
        f1 = args.base_features
        features_tuple = (f1, f1 * 2, f1 * 4, f1 * 8)
        print(f"Features: {features_tuple}")

    if args.model in {"deeplabv3plus", "unetplusplus", "fpn"}:
        print(f"Encoder: {args.encoder_name}")
        print(f"Encoder weights: {encoder_weights}")

    if args.model == "torchgeo_resnet50_unet":
        print(f"TorchGeo weights: {args.torchgeo_weights}")
        print(f"TorchGeo decoder dropout: {args.torchgeo_decoder_dropout}")
        print(f"Encoder learning rate: {args.encoder_learning_rate}")
        print(f"Freeze encoder epochs: {args.freeze_encoder_epochs}")

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

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

    optimizer = create_optimizer(
        model=model,
        model_name=args.model,
        learning_rate=args.learning_rate,
        encoder_learning_rate=args.encoder_learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_val_miou = -1.0
    epochs_without_improvement = 0

    encoder_was_frozen: bool | None = None

    for epoch in range(1, args.epochs + 1):
        encoder_frozen = (
            args.model == "torchgeo_resnet50_unet"
            and args.freeze_encoder_epochs > 0
            and epoch <= args.freeze_encoder_epochs
        )

        if args.model == "torchgeo_resnet50_unet" and encoder_frozen != encoder_was_frozen:
            set_encoder_trainable(model, trainable=not encoder_frozen)
            encoder_was_frozen = encoder_frozen

            if encoder_frozen:
                print(f"Freezing TorchGeo encoder for epoch {epoch}.")
            else:
                print(f"Unfreezing TorchGeo encoder at epoch {epoch}.")

        lrs = get_optimizer_lrs(optimizer)
        current_lr = max(lrs.values())
        encoder_lr = lrs.get("encoder", current_lr)
        decoder_lr = lrs.get("decoder", current_lr)

        print("")
        print("=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")

        if args.model == "torchgeo_resnet50_unet":
            print(
                f"Learning rate: encoder={encoder_lr:.6g}, "
                f"decoder={decoder_lr:.6g}"
            )
            print(f"Encoder frozen: {encoder_frozen}")
        else:
            print(f"Learning rate: {current_lr:.6g}")

        train_loss, train_acc, train_miou, train_mdice = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            keep_encoder_frozen=encoder_frozen,
        )

        val_loss, val_acc, val_miou, val_mdice, val_metrics_text = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            class_names=LANDCOVERNET_CLASSES,
        )

        scheduler.step(val_miou)

        print(
            f"Train: "
            f"loss={train_loss:.4f}, "
            f"acc={train_acc:.4f}, "
            f"mIoU={train_miou:.4f}, "
            f"mDice={train_mdice:.4f}"
        )

        print(
            f"Val:   "
            f"loss={val_loss:.4f}, "
            f"acc={val_acc:.4f}, "
            f"mIoU={val_miou:.4f}, "
            f"mDice={val_mdice:.4f}"
        )

        print("")
        print("Validation details:")
        print(val_metrics_text)

        append_history_row(
            history_path=history_path,
            row={
                "epoch": epoch,
                "train_loss": train_loss,
                "train_pixel_acc": train_acc,
                "train_miou": train_miou,
                "train_mdice": train_mdice,
                "val_loss": val_loss,
                "val_pixel_acc": val_acc,
                "val_miou": val_miou,
                "val_mdice": val_mdice,
                "lr": current_lr,
                "encoder_lr": encoder_lr,
                "decoder_lr": decoder_lr,
                "encoder_frozen": int(encoder_frozen),
            },
        )

        save_checkpoint(
            output_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_miou=best_val_miou,
            args=args,
        )

        improved = val_miou > best_val_miou + args.early_stopping_min_delta

        if improved:
            best_val_miou = val_miou
            epochs_without_improvement = 0

            save_checkpoint(
                output_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_miou=best_val_miou,
                args=args,
            )

            print(f"Saved new best model with val mIoU={best_val_miou:.4f}")
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience is not None:
            print(
                f"Early stopping counter: "
                f"{epochs_without_improvement}/{args.early_stopping_patience}"
            )

            if epochs_without_improvement >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

    config_path = output_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("")
    print("=" * 80)
    print("Training finished.")
    print(f"Best val mIoU: {best_val_miou:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
