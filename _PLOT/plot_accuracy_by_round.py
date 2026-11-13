"""
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "How LLM capability shapes human learning to collaborate in safety-critical decisions"
Author: Jiajie Shen (maintainer); other contributors listed in paper.
-------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (
    COLORS,
    DATA_DIR,
    SINGLE_FIGURE_PX,
    clean_axis,
    require_data_files,
    save_rgb,
    set_journal_style,
)


CONDITIONS = {
    "no_ai": ("Human alone", COLORS["human"], "1_no_ai.png"),
    "claude": ("Human–AI (low)", COLORS["low"], "1_low.png"),
    "qwen": ("Human–AI (medium)", COLORS["medium"], "1_middle.png"),
    "gpt": ("Human–AI (high)", COLORS["high"], "1_high.png"),
}


def normalize_version(value: object) -> str:
    text = str(value).strip().lower()
    if text == "no_ai": return "no_ai"
    if "claude" in text: return "claude"
    if "qwen" in text: return "qwen"
    if "gpt" in text: return "gpt"
    return text


def load_trials() -> pd.DataFrame:
    require_data_files("participant_trials")
    files = sorted((DATA_DIR / "participant_trials").glob("P*.csv"))
    if not files:
        raise FileNotFoundError("No participant trial files were found")
    data = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    data["condition"] = data["ai_version"].map(normalize_version)
    data["accuracy"] = data["round_score"].eq(0).astype(float)
    return data


def draw(data: pd.DataFrame, condition: str, title: str, color: str, filename: str) -> None:
    subset = data[data["condition"].eq(condition)].copy()
    summary = subset.groupby("collaboration_round")["accuracy"].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"].fillna(0) / np.sqrt(summary["count"].clip(lower=1))
    summary["lower"] = (summary["mean"] - 1.96 * summary["se"]).clip(0, 1)
    summary["upper"] = (summary["mean"] + 1.96 * summary["se"]).clip(0, 1)
    if len(summary) != 60:
        raise ValueError(f"{condition}: expected summaries for 60 rounds, found {len(summary)}")
    x = summary["collaboration_round"].to_numpy(float)
    y = summary["mean"].to_numpy(float)
    fit = stats.linregress(x, y)
    fit_y = fit.intercept + fit.slope * x
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, height_ratios=(3.5, 0.70), left=0.14, right=0.98,
                            bottom=0.14, top=0.91, hspace=0.48)
    axis = fig.add_subplot(grid[0]); box = fig.add_subplot(grid[1])
    axis.fill_between(x, summary["lower"], summary["upper"], color=color, alpha=0.13, linewidth=0)
    axis.plot(x, y, color=color, linewidth=1.15, marker="o", markersize=2.4, markeredgewidth=0)
    axis.plot(x, fit_y, color=color, linewidth=1.0, linestyle=(0, (4, 2.5)))
    axis.set_title(title, pad=5); axis.set_ylabel("Accuracy"); axis.set_xlabel("Collaboration round", labelpad=3)
    axis.set_xlim(1, 60); axis.set_xticks([1, 20, 40, 60]); axis.set_ylim(0.50, 1.00)
    axis.set_yticks(np.arange(0.50, 1.001, 0.10)); clean_axis(axis)
    rt = pd.to_numeric(subset["reaction_time"], errors="coerce").dropna().to_numpy()
    box.boxplot(rt, vert=False, widths=0.48, patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.24, "edgecolor": color, "linewidth": 0.8},
                medianprops={"color": COLORS["text"], "linewidth": 1.0},
                whiskerprops={"color": color, "linewidth": 0.8}, capprops={"color": color, "linewidth": 0.8},
                flierprops={"marker": "o", "markersize": 1.6, "markerfacecolor": color, "markeredgecolor": "none", "alpha": 0.30})
    box.set_xlabel("Decision time (s)", labelpad=2); box.set_xlim(0, 5.25); box.set_xticks(np.arange(0, 5.1, 1.0)); box.set_yticks([])
    box.spines["top"].set_visible(False); box.spines["right"].set_visible(False); box.spines["left"].set_visible(False)
    box.grid(axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.65); box.set_axisbelow(True)
    print(save_rgb(fig, filename, SINGLE_FIGURE_PX))


def main() -> None:
    set_journal_style(); data = load_trials()
    for condition, (title, color, filename) in CONDITIONS.items():
        draw(data, condition, title, color, filename)


if __name__ == "__main__":
    main()
