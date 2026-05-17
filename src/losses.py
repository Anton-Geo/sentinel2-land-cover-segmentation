from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for semantic segmentation.

    Expected inputs:
        logits:  [B, C, H, W]
        targets: [B, H, W]

    targets should contain class ids from 0 to C-1.
    ignore_index pixels are ignored.
    """

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        if alpha is None:
            self.alpha = None
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor)

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]
        """
        if logits.ndim != 4:
            raise ValueError(f"Expected logits shape [B, C, H, W], got {logits.shape}")

        if targets.ndim != 3:
            raise ValueError(f"Expected targets shape [B, H, W], got {targets.shape}")

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits batch={logits.shape[0]}, "
                f"targets batch={targets.shape[0]}"
            )

        if logits.shape[2:] != targets.shape[1:]:
            raise ValueError(
                f"Spatial size mismatch: logits spatial={logits.shape[2:]}, "
                f"targets spatial={targets.shape[1:]}"
            )

        num_classes = logits.shape[1]

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index

            safe_targets = targets.clone()
            safe_targets[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            safe_targets = targets

        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        if safe_targets.min() < 0 or safe_targets.max() >= num_classes:
            raise ValueError(
                f"Target values must be in [0, {num_classes - 1}] "
                f"or equal to ignore_index={self.ignore_index}. "
                f"Got min={safe_targets.min().item()}, max={safe_targets.max().item()}."
            )

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets_unsqueezed = safe_targets.unsqueeze(1)

        log_pt = torch.gather(
            log_probs,
            dim=1,
            index=targets_unsqueezed,
        ).squeeze(1)

        pt = torch.gather(
            probs,
            dim=1,
            index=targets_unsqueezed,
        ).squeeze(1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)

            if alpha.numel() != num_classes:
                raise ValueError(
                    f"alpha must have {num_classes} values, got {alpha.numel()}"
                )

            alpha_t = alpha[safe_targets]
            loss = alpha_t * loss

        loss = loss[valid_mask]

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class DiceLoss(nn.Module):
    """
    Multi-class soft Dice Loss for semantic segmentation.

    Expected inputs:
        logits:  [B, C, H, W]
        targets: [B, H, W]

    targets should contain class ids from 0 to C-1.
    ignore_index pixels are ignored.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C, H, W]
        targets: [B, H, W]
        """
        if logits.ndim != 4:
            raise ValueError(f"Expected logits shape [B, C, H, W], got {logits.shape}")

        if targets.ndim != 3:
            raise ValueError(f"Expected targets shape [B, H, W], got {targets.shape}")

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits batch={logits.shape[0]}, "
                f"targets batch={targets.shape[0]}"
            )

        if logits.shape[2:] != targets.shape[1:]:
            raise ValueError(
                f"Spatial size mismatch: logits spatial={logits.shape[2:]}, "
                f"targets spatial={targets.shape[1:]}"
            )

        num_classes = logits.shape[1]

        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index

            safe_targets = targets.clone()
            safe_targets[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)
            safe_targets = targets

        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        if safe_targets.min() < 0 or safe_targets.max() >= num_classes:
            raise ValueError(
                f"Target values must be in [0, {num_classes - 1}] "
                f"or equal to ignore_index={self.ignore_index}. "
                f"Got min={safe_targets.min().item()}, max={safe_targets.max().item()}."
            )

        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(
            safe_targets,
            num_classes=num_classes,
        )  # [B, H, W, C]

        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class ComboLoss(nn.Module):
    """
    Combined loss:

        total_loss = focal_weight * FocalLoss + dice_weight * DiceLoss

    By default:

        total_loss = 0.5 * FocalLoss + 0.5 * DiceLoss
    """

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        total_weight = focal_weight + dice_weight

        if total_weight <= 0:
            raise ValueError("focal_weight + dice_weight must be > 0")

        self.focal_weight = focal_weight / total_weight
        self.dice_weight = dice_weight / total_weight

        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
            ignore_index=ignore_index,
        )

        self.dice = DiceLoss(
            smooth=1e-6,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.focal.ignore_index is not None:
            if not (targets != self.focal.ignore_index).any():
                return logits.sum() * 0.0

        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss

        return loss


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

    # Add some ignored pixels
    targets[:, :10, :10] = 255

    criterion = ComboLoss(
        alpha=None,
        gamma=2.0,
        focal_weight=0.5,
        dice_weight=0.5,
        ignore_index=255,
    )

    loss = criterion(logits, targets)

    print(f"Loss: {loss.item():.6f}")
