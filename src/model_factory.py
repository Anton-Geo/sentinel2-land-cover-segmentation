from __future__ import annotations

from typing import Optional

import torch.nn as nn

from src.model import ResidualUNet
from src.model_torchgeo import TorchGeoResNet50UNet


SUPPORTED_MODELS = [
    "resunet",
    "deeplabv3plus",
    "unetplusplus",
    "fpn",
    "torchgeo_resnet50_unet",
]


def create_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    base_features: int = 32,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    torchgeo_weights: str | None = "sentinel2_all_dino",
    torchgeo_decoder_dropout: float = 0.1,
) -> nn.Module:
    """
    Create segmentation model by name.

    Supported models:
    - resunet: custom Residual U-Net from src.model
    - deeplabv3plus: DeepLabV3+ from segmentation_models_pytorch
    - unetplusplus: U-Net++ from segmentation_models_pytorch
    - fpn: FPN from segmentation_models_pytorch
    - torchgeo_resnet50_unet: TorchGeo pretrained ResNet50 encoder + U-Net decoder
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

    if model_name in {"deeplabv3plus", "unetplusplus", "fpn"}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is not installed. "
                "Install it with: pip install segmentation-models-pytorch"
            ) from exc

        common_kwargs = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": num_classes,
        }

        if model_name == "deeplabv3plus":
            return smp.DeepLabV3Plus(**common_kwargs)

        if model_name == "unetplusplus":
            return smp.UnetPlusPlus(**common_kwargs)

        if model_name == "fpn":
            return smp.FPN(**common_kwargs)

    if model_name == "torchgeo_resnet50_unet":
        return TorchGeoResNet50UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            weights_name=torchgeo_weights,
            dropout=torchgeo_decoder_dropout,
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        f"Supported models: {', '.join(SUPPORTED_MODELS)}"
    )
