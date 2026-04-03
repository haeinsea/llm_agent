from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PAPER_COLORS = {
    "navy": "#16324f",
    "teal": "#2a9d8f",
    "orange": "#f4a261",
    "red": "#d1495b",
    "gold": "#e9c46a",
    "slate": "#5f6c7b",
    "gray_fill": "#d8dee6",
    "grid": "#d7dde5",
    "ink": "#1f2933",
    "highlight": "#c0392b",
    "muted_blue": "#6c8ebf",
    "muted_green": "#6aa06a",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PAPER_COLORS["slate"],
            "axes.labelcolor": PAPER_COLORS["ink"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.color": PAPER_COLORS["ink"],
            "ytick.color": PAPER_COLORS["ink"],
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "legend.frameon": False,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d0d7de",
            "legend.fontsize": 9.5,
            "grid.color": PAPER_COLORS["grid"],
            "grid.alpha": 0.6,
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "savefig.pad_inches": 0.02,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axes(ax, *, add_grid: bool = True, y_grid_only: bool = False) -> None:
    if add_grid:
        ax.grid(True, axis="y" if y_grid_only else "both", zorder=0)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=PAPER_COLORS["ink"],
    )


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
