from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn


def normalize_architecture_name(name: Any) -> str:
    raw = str(name or "modern_tcn").strip().lower().replace("-", "_")
    if raw in {"tcn", "legacy_tcn"}:
        return "tcn"
    if raw in {"modern_tcn", "moderntcn", "modern"}:
        return "modern_tcn"
    raise ValueError(f"Unsupported temporal architecture: {name}")


def temporal_model_display_name(name: Any) -> str:
    return "ModernTCN" if normalize_architecture_name(name) == "modern_tcn" else "TCN"


def _as_int_list(values: Sequence[Any] | None, default: Sequence[int]) -> list[int]:
    raw = list(values) if values else list(default)
    return [int(v) for v in raw]


def _resolve_dilations(num_stages: int, raw: Sequence[Any] | None) -> list[int]:
    if num_stages <= 0:
        return []
    if raw:
        vals = [max(1, int(v)) for v in raw]
    else:
        vals = [2 ** i for i in range(num_stages)]
    if len(vals) >= num_stages:
        return vals[:num_stages]
    last = vals[-1] if vals else 1
    while len(vals) < num_stages:
        last = max(1, last * 2)
        vals.append(last)
    return vals


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, n_features: int, channels: Sequence[int] = (64, 64, 64), kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        dims = _as_int_list(channels, default=[64, 64, 64])
        layers = []
        n_in = int(n_features)
        for i, ch in enumerate(dims):
            dilation = 2 ** i
            padding = (int(kernel_size) - 1) * dilation
            layers.append(
                TemporalBlock(
                    n_inputs=n_in,
                    n_outputs=int(ch),
                    kernel_size=int(kernel_size),
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=float(dropout),
                )
            )
            n_in = int(ch)
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)
        return h[:, :, -1]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h_last = self.extract_features(x)
        logits = self.head(h_last).squeeze(-1)
        if return_features:
            return logits, h_last
        return logits


class ModernTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        expansion_ratio: int = 2,
    ):
        super().__init__()
        hidden = max(int(out_channels) * max(1, int(expansion_ratio)), int(out_channels))
        padding = ((int(kernel_size) - 1) * int(dilation)) // 2
        self.proj = nn.Identity() if int(in_channels) == int(out_channels) else nn.Conv1d(int(in_channels), int(out_channels), 1)
        self.depthwise = nn.Conv1d(
            int(out_channels),
            int(out_channels),
            int(kernel_size),
            padding=padding,
            dilation=int(dilation),
            groups=int(out_channels),
        )
        self.norm = nn.BatchNorm1d(int(out_channels))
        self.pointwise_in = nn.Conv1d(int(out_channels), hidden, 1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.pointwise_out = nn.Conv1d(hidden, int(out_channels), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.depthwise(residual)
        out = self.norm(out)
        out = self.pointwise_in(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.pointwise_out(out)
        out = self.dropout(out)
        return residual + out


class ModernTCNClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: Sequence[int] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: Sequence[int] | None = None,
        expansion_ratio: int = 2,
        pool: str = "avg",
    ):
        super().__init__()
        dims = _as_int_list(channels, default=[64, 64, 64])
        stage_dilations = _resolve_dilations(len(dims), dilations)
        blocks: list[nn.Module] = []
        in_dim = int(n_features)
        for out_dim, dilation in zip(dims, stage_dilations):
            blocks.append(
                ModernTCNBlock(
                    in_channels=in_dim,
                    out_channels=int(out_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                    expansion_ratio=int(expansion_ratio),
                )
            )
            in_dim = int(out_dim)
        self.backbone = nn.Sequential(*blocks)
        self.pool = str(pool or "avg").lower()
        self.head_norm = nn.BatchNorm1d(dims[-1])
        self.head = nn.Linear(dims[-1], 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.head_norm(h)
        if self.pool == "last":
            return h[:, :, -1]
        return h.mean(dim=-1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        pooled = self.extract_features(x)
        logits = self.head(pooled).squeeze(-1)
        if return_features:
            return logits, pooled
        return logits


def build_temporal_model(n_features: int, cfg: dict[str, Any]) -> nn.Module:
    arch = normalize_architecture_name(cfg.get("architecture", "modern_tcn"))
    channels = _as_int_list(cfg.get("channels"), default=[64, 64, 64])
    kernel_size = int(cfg.get("kernel_size", 3))
    dropout = float(cfg.get("dropout", 0.1))
    if arch == "modern_tcn":
        return ModernTCNClassifier(
            n_features=int(n_features),
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=cfg.get("dilations"),
            expansion_ratio=int(cfg.get("expansion_ratio", 2)),
            pool=str(cfg.get("pool", "avg")),
        )
    return TCNClassifier(
        n_features=int(n_features),
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
