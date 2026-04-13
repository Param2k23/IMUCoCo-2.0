"""
model.py
========
1D-CNN / ResNet-1D classifier for IMU-based sensor location identification.

Architecture choices
--------------------
  ResNet1D  : 4 residual blocks → Global Average Pooling → 24-class head (default).
  CNN1D     : 3 plain conv blocks (lightweight baseline).

Input  : (B, 6, 120)  — 3-axis Accel + 3-axis Gyro, channels-first
Output : (B, 24)      — raw logits (apply softmax externally / in loss)

Usage
-----
    from model import ResNet1D
    net = ResNet1D(n_classes=24, in_channels=6)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    """Conv1d → BatchNorm1d → ReLU."""
    def __init__(self, in_c: int, out_c: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__(
            nn.Conv1d(in_c, out_c, kernel, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )


class ResBlock1D(nn.Module):
    """
    Residual block: two ConvBnRelu layers + skip connection.
    Optional stride on the first conv to downsample the sequence length.
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBnRelu(in_c, out_c, kernel=3, stride=stride,
                                padding=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_c, out_c, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm1d(out_c),
        )
        # Skip connection: 1×1 conv if channels or stride change
        self.skip = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


# ---------------------------------------------------------------------------
# ResNet-1D  (primary model)
# ---------------------------------------------------------------------------

class ResNet1D(nn.Module):
    """
    4-block ResNet for 1-D time-series classification.

    Block channel progression: 64 → 128 → 256 → 256
    Sequence downsampling (stride=2) at blocks 2 and 3.
    """

    def __init__(self, n_classes: int = 24, in_channels: int = 6,
                 base_filters: int = 64, dropout: float = 0.3):
        super().__init__()

        # Stem
        self.stem = ConvBnRelu(in_channels, base_filters, kernel=7,
                               stride=1, padding=3)

        # Residual blocks
        self.layer1 = ResBlock1D(base_filters,     base_filters,     stride=1)
        self.layer2 = ResBlock1D(base_filters,     base_filters * 2, stride=2)
        self.layer3 = ResBlock1D(base_filters * 2, base_filters * 4, stride=2)
        self.layer4 = ResBlock1D(base_filters * 4, base_filters * 4, stride=1)

        # Classification head
        self.gap     = nn.AdaptiveAvgPool1d(1)    # Global Average Pooling
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(base_filters * 4, n_classes)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)  where C=6, T=120 by default
        returns logits : (B, n_classes)
        """
        x = self.stem(x)      # (B, 64, T)
        x = self.layer1(x)    # (B, 64, T)
        x = self.layer2(x)    # (B, 128, T/2)
        x = self.layer3(x)    # (B, 256, T/4)
        x = self.layer4(x)    # (B, 256, T/4)
        x = self.gap(x)       # (B, 256, 1)
        x = x.squeeze(-1)     # (B, 256)
        x = self.dropout(x)
        return self.fc(x)     # (B, 24)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax applied)."""
        return F.softmax(self.forward(x), dim=-1)


# ---------------------------------------------------------------------------
# CNN-1D  (lightweight baseline)
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """
    3-block plain 1-D CNN baseline (no residual connections).
    Easier to train on small datasets; useful for ablation.
    """

    def __init__(self, n_classes: int = 24, in_channels: int = 6,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(in_channels, 64,  kernel=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            ConvBnRelu(64,  128, kernel=5, stride=1, padding=2),
            nn.MaxPool1d(2),
            ConvBnRelu(128, 256, kernel=3, stride=1, padding=1),
        )
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(arch: str = 'resnet', **kwargs) -> nn.Module:
    """
    Factory function.

    Parameters
    ----------
    arch : 'resnet' (default) or 'cnn'
    kwargs : forwarded to the model constructor
    """
    if arch == 'resnet':
        return ResNet1D(**kwargs)
    elif arch == 'cnn':
        return CNN1D(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch!r}. Choose 'resnet' or 'cnn'.")


if __name__ == '__main__':
    # Quick sanity check
    device = torch.device('cpu')
    for arch in ('resnet', 'cnn'):
        net = build_model(arch).to(device)
        x   = torch.randn(8, 6, 120)
        out = net(x)
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"{arch:8s}  in={tuple(x.shape)}  out={tuple(out.shape)}"
              f"  params={n_params:,}")
