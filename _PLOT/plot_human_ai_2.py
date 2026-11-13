"""
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "How LLM capability shapes human learning to collaborate in safety-critical decisions"
Author: Jiajie Shen (maintainer); other contributors listed in paper.
-------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import (
    COLORS,
    DATA_DIR,
    WIDE_FIGURE_PX,
    clean_axis,
    require_data_files,
    save_rgb,
    set_journal_style,
)


ABILITY_COLORS = {2: COLORS["low"], 3: COLORS["medium"], 4: COLORS["high"]}
ABILITY_MODELS = {
    2: (
        "claude-opus-4-6",
        "deepseek-v4-flash-thinking",
        "doubao-seed-2-0-lite",
    ),
    3: (
        "doubao-seed-2-0-pro",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gpt-5.4-mini",
        "qwen3.6-flash",
        "qwen3.6-plus",
    ),
    4: (
        "claude-sonnet-4-6",
        "deepseek-v4-pro-thinking",
        "gpt-5.5",
    ),
}
ABILITY_LABELS = {2: "L2", 3: "L3", 4: "L4"}


def normalize_condition(value: object) -> str:
    text = str(value).strip().lower()
    if text == "no_ai":
        return "no_ai"
    if "claude" in text:
        return "low"
    if "qwen" in text:
        return "medium"
    if "gpt" in text:
        return "high"
    return text


def describe(values: pd.Series | np.ndarray) -> dict[str, float]:
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        raise ValueError("Cannot summarize an empty accuracy series")
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": std,
        "n": int(len(values)),
        "ci95": 1.96 * std / np.sqrt(len(values)),
    }


def build_summary() -> pd.DataFrame:
    require_data_files("participant_trials", "scenario_statistics.csv")
    trial_files = sorted((DATA_DIR / "participant_trials").glob("P*.csv"))
    if not trial_files:
        raise FileNotFoundError("No participant trial files were found")
    trials = pd.concat((pd.read_csv(path) for path in trial_files), ignore_index=True)
    trials["condition"] = trials["ai_version"].map(normalize_condition)
    trials["accuracy"] = trials["round_score"].eq(0).astype(float)
    trials["block"] = ((trials["collaboration_round"] - 1) // 10 + 1).astype(int)

    participant_accuracy = (
        trials.groupby(["participant_id", "condition"], as_index=False)["accuracy"].mean()
    )
    rows = [{
        "series": "Human alone",
        "ability_level": 0,
        "ability_label": "Human alone",
        "block": np.nan,
        **describe(participant_accuracy.loc[
            participant_accuracy["condition"].eq("no_ai"), "accuracy"
        ]),
    }]

    block_accuracy = (
        trials.groupby(["participant_id", "condition", "block"], as_index=False)["accuracy"].mean()
    )
    condition_by_level = {2: "low", 3: "medium", 4: "high"}
    for level, condition in condition_by_level.items():
        for block in (1, 6):
            values = block_accuracy.loc[
                block_accuracy["condition"].eq(condition) & block_accuracy["block"].eq(block),
                "accuracy",
            ]
            rows.append({
                "series": f"Human-AI block {block}",
                "ability_level": level,
                "ability_label": ABILITY_LABELS[level],
                "block": block,
                **describe(values),
            })

    scenarios = pd.read_csv(DATA_DIR / "scenario_statistics.csv", encoding="utf-8-sig")
    correct = pd.to_numeric(scenarios["correct_answer"], errors="coerce")
    for level, models in ABILITY_MODELS.items():
        model_accuracies = []
        for model in models:
            column = f"output-{model}"
            if column not in scenarios:
                raise KeyError(f"scenario_statistics.csv is missing {column}")
            output = pd.to_numeric(scenarios[column], errors="coerce")
            valid = correct.notna() & output.notna()
            if not valid.any():
                raise ValueError(f"No valid scenario results for {model}")
            # The source analysis reported each model accuracy to four decimals
            # before calculating the ability-group mean and confidence interval.
            model_accuracies.append(round(float(output[valid].eq(correct[valid]).mean()), 4))
        rows.append({
            "series": "AI alone",
            "ability_level": level,
            "ability_label": ABILITY_LABELS[level],
            "block": np.nan,
            **describe(model_accuracies),
        })
    return pd.DataFrame(rows)


def point(axis, x, row, color, marker, filled=True):
    axis.errorbar(
        x,
        float(row["mean"]),
        yerr=float(row["ci95"]),
        fmt=marker,
        color=color,
        ecolor=color,
        markerfacecolor=color if filled else "white",
        markersize=6.6,
        markeredgewidth=1.0,
        elinewidth=1.35,
        capsize=3.2,
        zorder=4,
    )


def select(summary, series, level):
    rows = summary[summary["series"].eq(series) & summary["ability_level"].eq(level)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {series}, ability level {level}")
    return rows.iloc[0]


def main() -> None:
    set_journal_style()
    summary = build_summary()
    labels = [
        "Human\nalone", "AI\n(low)", "Human–AI\n(low)",
        "AI\n(medium)", "Human–AI\n(medium)", "AI\n(high)", "Human–AI\n(high)",
    ]
    positions = np.array([0.0, 1.0, 2.15, 3.35, 4.50, 5.70, 6.85])
    fig, axis = plt.subplots()
    human = select(summary, "Human alone", 0)
    axis.axhline(float(human["mean"]), color=COLORS["human"], linewidth=1.25,
                 linestyle=(0, (4, 2.5)), alpha=0.75)
    point(axis, positions[0], human, COLORS["human"], "o")
    for level, ai_x, collaboration_x in zip((2, 3, 4), positions[1::2], positions[2::2]):
        point(axis, ai_x, select(summary, "AI alone", level), COLORS["ai"], "o")
        block1 = select(summary, "Human-AI block 1", level)
        block6 = select(summary, "Human-AI block 6", level)
        xs = (collaboration_x - 0.11, collaboration_x + 0.11)
        axis.plot(xs, [block1["mean"], block6["mean"]], color=ABILITY_COLORS[level],
                  linewidth=1.35, linestyle=(0, (3, 2)), alpha=0.85)
        point(axis, xs[0], block1, ABILITY_COLORS[level], "s", filled=False)
        point(axis, xs[1], block6, ABILITY_COLORS[level], "D")
    handles = [
        Line2D([0], [0], marker="o", color=COLORS["human"], linestyle=(0, (4, 2.5)), label="Human alone"),
        Line2D([0], [0], marker="o", color=COLORS["ai"], linestyle="none", label="AI alone"),
        Line2D([0], [0], marker="s", markerfacecolor="white", color=COLORS["ai"], linestyle="none", label="Human–AI block 1"),
        Line2D([0], [0], marker="D", color=COLORS["ai"], linestyle="none", label="Human–AI block 6"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4,
               columnspacing=0.6, handlelength=1.5, handletextpad=0.35)
    axis.set_ylabel("Accuracy")
    axis.set_xticks(positions, labels)
    axis.set_xlim(-0.45, 7.25)
    axis.set_ylim(0.62, 0.95)
    axis.set_yticks(np.arange(0.65, 0.951, 0.05))
    clean_axis(axis)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.83)
    print(save_rgb(fig, "Human-AI_2.png", WIDE_FIGURE_PX))


if __name__ == "__main__":
    main()
