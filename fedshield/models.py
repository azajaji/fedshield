"""Model zoo: classifiers + lightweight autoencoder for FEDShield's edge-side
local validation stage."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


# --------------------------------------------------------------------------- #
#                              Classifiers
# --------------------------------------------------------------------------- #
class ECG1DCNN(nn.Module):
    """1-D CNN suitable for MIT-BIH-style heart-beat classification."""

    def __init__(self, num_classes: int = 5, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:        # (B, T) -> (B, 1, T)
            x = x.unsqueeze(1)
        f = self.features(x).flatten(1)
        return self.classifier(f)


class WESADCNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 6) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> conv -> (B, 64, T') -> (B, T', 64)
        f = self.conv(x).transpose(1, 2)
        out, _ = self.lstm(f)
        return self.fc(out[:, -1, :])


class IoMTMLP(nn.Module):
    def __init__(self, num_classes: int = 6, input_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#                              Lightweight Autoencoder
# --------------------------------------------------------------------------- #
class TinyAE(nn.Module):
    """Under-complete autoencoder used for edge-side anomaly gating.

    The bottleneck is the *only* knob exposed; the rest of the topology is
    auto-derived from the input shape so the AE remains dataset-agnostic.
    Footprint stays under 10k parameters by construction.
    """

    def __init__(self, input_numel: int, bottleneck: int = 16) -> None:
        super().__init__()
        h = max(32, bottleneck * 4)
        self.input_numel = input_numel
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_numel, h), nn.ReLU(),
            nn.Linear(h, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h), nn.ReLU(),
            nn.Linear(h, input_numel),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        z = self.encoder(flat)
        out = self.decoder(z)
        return out.view_as(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
        # per-sample MSE
        return ((recon - x) ** 2).reshape(x.size(0), -1).mean(dim=1)


# --------------------------------------------------------------------------- #
#                              Factory
# --------------------------------------------------------------------------- #
def build_classifier(cfg: ModelConfig, input_shape: Tuple[int, ...]) -> nn.Module:
    arch = cfg.arch.lower()
    if arch == "ecg_cnn":
        in_ch = input_shape[0] if len(input_shape) >= 2 else 1
        return ECG1DCNN(num_classes=cfg.num_classes, in_channels=in_ch)
    if arch == "wesad_cnnlstm":
        in_ch = input_shape[0] if len(input_shape) >= 2 else 1
        return WESADCNNLSTM(num_classes=cfg.num_classes, in_channels=in_ch)
    if arch == "iomt_mlp":
        flat = int(torch.tensor(input_shape).prod().item())
        return IoMTMLP(num_classes=cfg.num_classes, input_dim=flat)
    raise ValueError(f"unknown arch {cfg.arch}")


def build_autoencoder(cfg: ModelConfig, input_shape: Tuple[int, ...]) -> TinyAE:
    flat = int(torch.tensor(input_shape).prod().item())
    return TinyAE(input_numel=flat, bottleneck=cfg.ae_bottleneck)


def model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
