from __future__ import annotations

from typing import Optional

import torch.nn as nn

from src.model import ResidualUNet


def create_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    base_features: int = 32,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    """
    Create segmentation model by name.

    Supported models:
    - resunet: custom Residual U-Net from src.model
    - deeplabv3plus: DeepLabV3+ from segmentation_models_pytorch
    """

    model_name = model_name.lower()

    if model_name == "resunet":
        features = (
            base_features,
            base_features * 2,
            base_features * 4,
            base_features * 8,
        )

        return ResidualUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
        )

    if model_name == "deeplabv3plus":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is not installed. "
                "Install it with: pip install segmentation-models-pytorch"
            ) from exc

        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    if model_name == "unetplusplus":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is not installed. "
                "Install it with: pip install segmentation-models-pytorch"
            ) from exc

        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    if model_name == "fpn":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is not installed. "
                "Install it with: pip install segmentation-models-pytorch"
            ) from exc

        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        "Supported models: resunet, deeplabv3plus"
    )
