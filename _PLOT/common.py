"""
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "How LLM capability shapes human learning to collaborate in safety-critical decisions"
Author: Jiajie Shen (maintainer); other contributors listed in paper.
-------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get("HLC_DATA_DIR", PROJECT_ROOT / "_DATA")
).expanduser().resolve()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FIGURE_DPI = 600
SINGLE_FIGURE_PX = (3000, 2400)
WIDE_FIGURE_PX = (5400, 2400)

COLORS = {
    "human": "#4C78A8",
    "ai": "#6F6F6F",
    "human_ai": "#D9786D",
    "low": "#4E9A70",
    "medium": "#D9786D",
    "high": "#8B75A9",
    "grid": "#D9D9D9",
    "text": "#222222",
}


def require_data_files(*relative_paths: str) -> None:
    """Raise a clear error when the separately distributed data are missing."""
    missing = [str(DATA_DIR / name) for name in relative_paths if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Required data files were not found:\n- "
            + "\n- ".join(missing)
            + "\nDownload and extract the Figshare data archive, then set "
            "HLC_DATA_DIR to the extracted _DATA directory."
        )


def set_journal_style() -> None:
    """Apply the shared 14/12-pt manuscript style."""
    arial_path = Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
    if arial_path.exists():
        mpl.font_manager.fontManager.addfont(str(arial_path))
        family = mpl.font_manager.FontProperties(fname=arial_path).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis: plt.Axes, grid_axis: str | None = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis:
        axis.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)


def save_rgb(fig: plt.Figure, filename: str, pixel_size: tuple[int, int]) -> Path:
    """Save an opaque RGB PNG at an exact pixel size."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(pixel_size[0] / FIGURE_DPI, pixel_size[1] / FIGURE_DPI)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=FIGURE_DPI, facecolor="white", bbox_inches=None)
    plt.close(fig)
    buffer.seek(0)
    output = OUTPUT_DIR / filename
    with Image.open(buffer) as rendered:
        rendered.convert("RGB").save(output, format="PNG", dpi=(FIGURE_DPI, FIGURE_DPI))
    return output
