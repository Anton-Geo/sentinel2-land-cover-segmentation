from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_TORCHGEO_RESNET50_WEIGHTS = {
    "none": None,
    "sentinel2_all_closp": "SENTINEL2_ALL_CLOSP",
    "sentinel2_all_decur": "SENTINEL2_ALL_DECUR",
    "sentinel2_all_dino": "SENTINEL2_ALL_DINO",
    "sentinel2_all_geoclosp": "SENTINEL2_ALL_GEOCLOSP",
    "sentinel2_all_moco": "SENTINEL2_ALL_MOCO",
    "sentinel2_all_softcon": "SENTINEL2_ALL_SOFTCON",
    "sentinel2_all_seco_eco": "SENTINEL2_ALL_SECO_ECO",
    "sentinel2_all_ndvi_seco_eco": "SENTINEL2_ALL_NDVI_SECO_ECO",
    "sentinel2_mi_ms_satlas": "SENTINEL2_MI_MS_SATLAS",
    "sentinel2_mi_rgb_satlas": "SENTINEL2_MI_RGB_SATLAS",
    "sentinel2_rgb_moco": "SENTINEL2_RGB_MOCO",
    "sentinel2_rgb_seco": "SENTINEL2_RGB_SECO",
    "sentinel2_si_ms_satlas": "SENTINEL2_SI_MS_SATLAS",
    "sentinel2_si_rgb_satlas": "SENTINEL2_SI_RGB_SATLAS",
}


class DecoderBlock(nn.Module):
    """
    Simple U-Net-like decoder block.

    If skip is provided:
        upsample x to skip resolution -> concatenate -> two conv layers.

    If skip is None:
        upsample x by 2 -> two conv layers.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )

        return self.block(x)


def get_resnet50_weights(weights_name: str | None) -> Any:
    """
    Convert a CLI-friendly string into a TorchGeo ResNet50_Weights enum.

    TorchGeo is imported lazily so the rest of the project can still run
    without torchgeo when using non-TorchGeo models.
    """
    if weights_name is None:
        return None

    normalized = weights_name.strip().lower()

    if normalized in {"", "none", "null"}:
        return None

    if normalized not in SUPPORTED_TORCHGEO_RESNET50_WEIGHTS:
        supported = ", ".join(sorted(SUPPORTED_TORCHGEO_RESNET50_WEIGHTS))
        raise ValueError(
            f"Unknown TorchGeo ResNet50 weights: {weights_name}. "
            f"Supported values: {supported}"
        )

    enum_name = SUPPORTED_TORCHGEO_RESNET50_WEIGHTS[normalized]

    if enum_name is None:
        return None

    try:
        from torchgeo.models import ResNet50_Weights
    except ImportError as exc:
        raise ImportError(
            "torchgeo is not installed. Install it with: pip install torchgeo"
        ) from exc

    return getattr(ResNet50_Weights, enum_name)


class TorchGeoResNet50UNet(nn.Module):
    """
    Semantic segmentation model with a TorchGeo pretrained ResNet50 encoder
    and a lightweight U-Net-like decoder.

    Expected output:
        logits: [B, num_classes, H, W]

    For Sentinel-2 all-band weights, use a 13-channel input ordered as:
        B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
    """

    def __init__(
        self,
        in_channels: int = 13,
        num_classes: int = 7,
        weights_name: str | None = "sentinel2_all_dino",
        decoder_channels: tuple[int, int, int, int, int] = (512, 256, 128, 64, 32),
        dropout: float = 0.1,
        use_input_adapter: bool = True,
    ) -> None:
        super().__init__()

        try:
            from torchgeo.models import resnet50
        except ImportError as exc:
            raise ImportError(
                "torchgeo is not installed. Install it with: pip install torchgeo"
            ) from exc

        weights = get_resnet50_weights(weights_name)
        self.encoder = resnet50(weights=weights)

        encoder_in_channels = self.encoder.conv1.in_channels

        if in_channels != encoder_in_channels:
            if not use_input_adapter:
                raise ValueError(
                    f"Input has {in_channels} channels, but TorchGeo encoder expects "
                    f"{encoder_in_channels}. Either prepare matching data or enable "
                    "use_input_adapter=True."
                )

            self.input_adapter = nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_in_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.input_adapter = nn.Identity()

        d1, d2, d3, d4, d5 = decoder_channels

        # ResNet50 feature channels:
        # x0 after stem: 64, H/2
        # x1 layer1:    256, H/4
        # x2 layer2:    512, H/8
        # x3 layer3:    1024, H/16
        # x4 layer4:    2048, H/32
        self.dec4 = DecoderBlock(2048, 1024, d1, dropout=dropout)
        self.dec3 = DecoderBlock(d1, 512, d2, dropout=dropout)
        self.dec2 = DecoderBlock(d2, 256, d3, dropout=dropout)
        self.dec1 = DecoderBlock(d3, 64, d4, dropout=dropout)
        self.dec0 = DecoderBlock(d4, 0, d5, dropout=dropout)

        self.final_conv = nn.Conv2d(d5, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x = self.input_adapter(x)

        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)

        # TorchGeo 0.9 uses timm-style ResNet models where the first
        # activation is usually called act1, while torchvision-style
        # ResNets use relu. Support both to keep the wrapper robust.
        if hasattr(self.encoder, "act1"):
            x0 = self.encoder.act1(x0)
        elif hasattr(self.encoder, "relu"):
            x0 = self.encoder.relu(x0)
        else:
            x0 = F.relu(x0, inplace=True)

        x = self.encoder.maxpool(x0)

        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        x = self.dec4(x4, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        x = self.dec0(x, None)

        logits = self.final_conv(x)
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits


if __name__ == "__main__":
    model = TorchGeoResNet50UNet(
        in_channels=13,
        num_classes=7,
        weights_name="sentinel2_all_dino",
    )

    x = torch.randn(2, 13, 192, 192)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Encoder input channels: {model.encoder.conv1.in_channels}")
