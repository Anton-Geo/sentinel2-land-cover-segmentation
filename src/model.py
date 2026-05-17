from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Standard U-Net block:
    Conv2d -> BatchNorm -> ReLU -> Dropout -> Conv2d -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class ResidualDoubleConv(nn.Module):
    """
    Residual version of DoubleConv.

    Main path:
        Conv2d -> BatchNorm -> ReLU -> Dropout -> Conv2d -> BatchNorm

    Shortcut path:
        Identity, if in_channels == out_channels
        1x1 Conv + BatchNorm, otherwise

    Output:
        ReLU(main path + shortcut)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + residual
        out = self.relu(out)

        return out


class ResidualUNet(nn.Module):
    """
    Residual U-Net for Sentinel-2 land cover segmentation.

    Input:
        x: [B, 4, H, W]

    Output:
        logits: [B, num_classes, H, W]

    Default setup:
        in_channels = 4
        num_classes = 7
        features = (32, 64, 128, 256)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 7,
        features: tuple[int, int, int, int] = (32, 64, 128, 256),
        dropout_encoder: tuple[float, float, float, float] = (0.0, 0.0, 0.05, 0.1),
        dropout_bottleneck: float = 0.2,
        dropout_decoder: tuple[float, float, float, float] = (0.1, 0.1, 0.05, 0.0),
    ) -> None:
        super().__init__()

        if len(features) != 4:
            raise ValueError("This implementation expects exactly 4 encoder levels.")

        if len(dropout_encoder) != 4:
            raise ValueError("dropout_encoder must contain exactly 4 values.")

        if len(dropout_decoder) != 4:
            raise ValueError("dropout_decoder must contain exactly 4 values.")

        f1, f2, f3, f4 = features

        self.enc1 = ResidualDoubleConv(
            in_channels=in_channels,
            out_channels=f1,
            dropout=dropout_encoder[0],
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResidualDoubleConv(
            in_channels=f1,
            out_channels=f2,
            dropout=dropout_encoder[1],
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ResidualDoubleConv(
            in_channels=f2,
            out_channels=f3,
            dropout=dropout_encoder[2],
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ResidualDoubleConv(
            in_channels=f3,
            out_channels=f4,
            dropout=dropout_encoder[3],
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ResidualDoubleConv(
            in_channels=f4,
            out_channels=f4 * 2,
            dropout=dropout_bottleneck,
        )

        self.up4 = nn.ConvTranspose2d(
            in_channels=f4 * 2,
            out_channels=f4,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = DoubleConv(
            in_channels=f4 * 2,
            out_channels=f4,
            dropout=dropout_decoder[0],
        )

        self.up3 = nn.ConvTranspose2d(
            in_channels=f4,
            out_channels=f3,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = DoubleConv(
            in_channels=f3 * 2,
            out_channels=f3,
            dropout=dropout_decoder[1],
        )

        self.up2 = nn.ConvTranspose2d(
            in_channels=f3,
            out_channels=f2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = DoubleConv(
            in_channels=f2 * 2,
            out_channels=f2,
            dropout=dropout_decoder[2],
        )

        self.up1 = nn.ConvTranspose2d(
            in_channels=f2,
            out_channels=f1,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = DoubleConv(
            in_channels=f1 * 2,
            out_channels=f1,
            dropout=dropout_decoder[3],
        )

        self.final_conv = nn.Conv2d(
            in_channels=f1,
            out_channels=num_classes,
            kernel_size=1,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)                  # [B, f1, H, W]
        enc2 = self.enc2(self.pool1(enc1))   # [B, f2, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))   # [B, f3, H/4, W/4]
        enc4 = self.enc4(self.pool3(enc3))   # [B, f4, H/8, W/8]

        bottleneck = self.bottleneck(
            self.pool4(enc4)
        )                                    # [B, 2*f4, H/16, W/16]

        dec4 = self.up4(bottleneck)          # [B, f4, H/8, W/8]
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)                # [B, f3, H/4, W/4]
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)                # [B, f2, H/2, W/2]
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)                # [B, f1, H, W]
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        logits = self.final_conv(dec1)       # [B, num_classes, H, W]

        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = ResidualUNet(
        in_channels=4,
        num_classes=7,
        features=(32, 64, 128, 256),
    )

    x = torch.randn(2, 4, 256, 256)

    with torch.no_grad():
        logits = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
