from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SegmentationMetrics:
    pixel_accuracy: float
    mean_iou: float
    mean_dice: float
    per_class_iou: list[float]
    per_class_dice: list[float]


class SegmentationMetricTracker:
    """
    Metric tracker for multi-class semantic segmentation.

    Expected shapes:
        logits:  [B, C, H, W]
        preds:   [B, H, W]
        targets: [B, H, W]

    Labels:
        valid classes: 0..num_classes-1
        ignore_index: ignored pixels, for example 255
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = 255,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device(device)

        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.long,
            device=self.device,
        )

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update_from_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Convert logits to predictions and update confusion matrix.

        logits:  [B, C, H, W]
        targets: [B, H, W]
        """
        if logits.ndim != 4:
            raise ValueError(f"Expected logits shape [B, C, H, W], got {logits.shape}")

        preds = torch.argmax(logits, dim=1)  # [B, H, W]
        self.update_from_predictions(preds, targets)

    @torch.no_grad()
    def update_from_predictions(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update confusion matrix from predicted class ids.

        preds:   [B, H, W]
        targets: [B, H, W]
        """
        if preds.shape != targets.shape:
            raise ValueError(
                f"preds and targets must have the same shape, "
                f"got preds={preds.shape}, targets={targets.shape}"
            )

        preds = preds.to(self.device)
        targets = targets.to(self.device)

        preds = preds.view(-1)
        targets = targets.view(-1)

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            preds = preds[valid_mask]
            targets = targets[valid_mask]

        valid_mask = (
            (targets >= 0)
            & (targets < self.num_classes)
            & (preds >= 0)
            & (preds < self.num_classes)
        )

        preds = preds[valid_mask]
        targets = targets[valid_mask]

        if targets.numel() == 0:
            return

        indices = targets * self.num_classes + preds

        confusion = torch.bincount(
            indices,
            minlength=self.num_classes * self.num_classes,
        )

        confusion = confusion.reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += confusion

    def compute(self) -> SegmentationMetrics:
        """
        Compute metrics from accumulated confusion matrix.

        confusion_matrix rows:    ground truth classes
        confusion_matrix columns: predicted classes
        """
        cm = self.confusion_matrix.float()

        true_positive = torch.diag(cm)

        support = cm.sum(dim=1)      # ground truth pixels per class
        predicted = cm.sum(dim=0)    # predicted pixels per class

        total = cm.sum()
        correct = true_positive.sum()

        if total > 0:
            pixel_accuracy = (correct / total).item()
        else:
            pixel_accuracy = 0.0

        union = support + predicted - true_positive
        iou = true_positive / union.clamp(min=1.0)

        dice_denominator = support + predicted
        dice = (2.0 * true_positive) / dice_denominator.clamp(min=1.0)

        # Classes that are absent in both target and prediction should not
        # affect mean IoU / mean Dice.
        valid_iou_classes = union > 0
        valid_dice_classes = dice_denominator > 0

        if valid_iou_classes.any():
            mean_iou = iou[valid_iou_classes].mean().item()
        else:
            mean_iou = 0.0

        if valid_dice_classes.any():
            mean_dice = dice[valid_dice_classes].mean().item()
        else:
            mean_dice = 0.0

        per_class_iou = iou.cpu().tolist()
        per_class_dice = dice.cpu().tolist()

        return SegmentationMetrics(
            pixel_accuracy=pixel_accuracy,
            mean_iou=mean_iou,
            mean_dice=mean_dice,
            per_class_iou=per_class_iou,
            per_class_dice=per_class_dice,
        )

    def get_confusion_matrix(self) -> torch.Tensor:
        return self.confusion_matrix.detach().cpu().clone()


def format_metrics(
    metrics: SegmentationMetrics,
    class_names: list[str] | None = None,
) -> str:
    """
    Create a readable string with global and per-class metrics.
    Useful for logging.
    """
    lines = [
        f"Pixel Acc: {metrics.pixel_accuracy:.4f}",
        f"Mean IoU:  {metrics.mean_iou:.4f}",
        f"Mean Dice: {metrics.mean_dice:.4f}",
    ]

    if class_names is None:
        class_names = [f"class_{idx}" for idx in range(len(metrics.per_class_iou))]

    lines.append("")
    lines.append("Per-class metrics:")

    for idx, class_name in enumerate(class_names):
        iou = metrics.per_class_iou[idx]
        dice = metrics.per_class_dice[idx]

        lines.append(
            f"  {idx:02d} {class_name:<20} IoU={iou:.4f} Dice={dice:.4f}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    batch_size = 2
    num_classes = 7
    height = 256
    width = 256

    logits = torch.randn(batch_size, num_classes, height, width)

    targets = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size, height, width),
    )

    # Add ignored area
    targets[:, :20, :20] = 255

    tracker = SegmentationMetricTracker(
        num_classes=num_classes,
        ignore_index=255,
    )

    tracker.update_from_logits(logits, targets)

    metrics = tracker.compute()

    print(format_metrics(metrics))
