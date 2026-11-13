"""
-------------------------------------------------------------------------------------------------
This code accompanies the paper titled "How LLM capability shapes human learning to collaborate in safety-critical decisions"
Author: Jiajie Shen (maintainer); other contributors listed in paper.
-------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
import os
import re
import shutil
from tempfile import TemporaryDirectory
import warnings

# Use a non-interactive backend so figures can be saved on headless systems.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pkhmm")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import gaussian_kde, wasserstein_distance

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
DATA_DIR = Path(
    os.environ.get("HLC_DATA_DIR", PROJECT_ROOT / "_DATA")
).expanduser().resolve()
TRIAL_DIR = DATA_DIR / "participant_trials"
DEFAULT_USER_FILE = DATA_DIR / "participants.csv"
SCENARIO_FILE = DATA_DIR / "scenario_statistics.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PAPER_RESULTS_DIR = RESULTS_DIR / "results_paper_all"

# Compare the focal three-state model with adjacent model sizes.
K_CANDIDATES = (2, 3, 4)
N_STARTS = 20
MAX_ITER = 500
TOL = 1e-4
L2 = 0.20
RANDOM_SEED = 20260612
FILTER_WINDOW_RADIUS = 0  # Smoothing-window radius in interaction rounds.
BOOTSTRAP_SAMPLES = 10000
ACCEPTANCE_CLASSIFICATION_THRESHOLD = 0.5
FONT_SIZE = 14
MATRIX_CELL_FONT_SIZE = 10
DISTRIBUTION_CURVE_LINEWIDTH = 1.2
CM_PER_INCH = 2.54
PAPER_COMBINED_FIGSIZE = (8, 3.5)
PAPER_TALL_COMBINED_FIGSIZE = (7, 4.2)
PAPER_MATRIX_COMBINED_FIGSIZE = (8 / CM_PER_INCH, 20 / CM_PER_INCH)
SINGLE_DISTRIBUTION_FIGSIZE = (8.2, 4.0)
PAPER_DISTRIBUTION_COMBINED_FIGSIZE = (
    PAPER_TALL_COMBINED_FIGSIZE[0],
    4.0,
)
PAPER_SELECTIVE_RELIANCE_FIGSIZE = PAPER_DISTRIBUTION_COMBINED_FIGSIZE
PAPER_BOOTSTRAP_COMBINED_FIGSIZE = (12, 10)
PAPER_VISUAL_MOCKUP_FIGSIZE = (11.0, 4.5)
PAPER_STATE_PROFILE_COLORS = (
    "#8EA5C8",  # RGB(142, 165, 200)
    "#A190B4",  # RGB(161, 144, 180)
    "#B9CCB3",  # RGB(185, 204, 179)
    "#ED9375",  # RGB(237, 147, 117)
)
RELIANCE_STATE_COLORS = ("#f5b14c", "#23998f", "#3f7fbf", "#ed1b3a")

N_INTERACTIONS_PER_SUBJECT = 60
EXPECTED_INTERACTION_ROUNDS = set(range(1, N_INTERACTIONS_PER_SUBJECT + 1))

SCENE_FEATURES = [
    "decision_ego_speed",
    "decision_ego_d2i",
    "decision_ego_TTC2i",
    "decision_other_speed",
    "decision_other_d2i",
    "decision_other_TTC2i",
]
AI_OUTPUT_COLUMNS = {
    "qwen3.6-plus": "output-qwen3.6-plus",
    "claude-opus-4-6": "output-claude-opus-4-6",
    "gpt-5.5": "output-gpt-5.5",
}
AI_VERSION_LEVELS = ["claude-opus-4-6", "qwen3.6-plus", "gpt-5.5"]
PAPER_AI_LABELS = {
    "claude-opus-4-6": "Human-AI (low)",
    "qwen3.6-plus": "Human-AI (median)",
    "gpt-5.5": "Human-AI (high)",
}
# Train all three experimental AI conditions.
TRAIN_AI_VERSIONS = ("claude-opus-4-6", "qwen3.6-plus", "gpt-5.5")
PAPER_FIGURE_FILES = (
    "01_bic_comparison.png",
    "02_state_profiles_Viterbi.png",
    "02_state_profiles_Viterbi-new.png",
    "03_transition_matrix.png",
    "03_transition_matrix-new.png",
    "05_prediction_calibration.png",
    "06_state_proportion.png",
    "06_state_proportion-new.png",
    "07_reliance_fraction.png",
    "08_distribution_reliance_probability.png",
    "reliance_probability.png",
    "reliance_probability_real_distribution.png",
    "selective_reliance_probability.png",
    "selective_reliance_probability_real_distribution.png",
)
FEEDBACK_NAMES = [
    "prev_ai_correct_accept",
    "prev_ai_wrong_reject",
    "prev_ai_wrong_accept",
    "prev_ai_correct_reject",
]
FEEDBACK_LABELS = {
    0: "Appropriate acceptance",
    1: "Appropriate rejection",
    2: "Inappropriate acceptance",
    3: "Inappropriate rejection",
}
INITIAL_CATEGORY_LEVELS = {
    "LLM_experience": [
        "Never",
        "Rare (monthly or tried before)",
        "Occasional (weekly)",
        "Frequent (daily)",
        "unknown",
    ],
    "age": ["25 or younger", "26–35", "36 or older", "unknown"],
    "education": [
        "High school or below",
        "Associate degree",
        "Bachelor's degree",
        "Postgraduate degree",
        "unknown",
    ],
    "gender": ["Female", "Male", "unknown"],
}


@dataclass
class SequenceData:
    subject: str
    participant_id: str
    trials: pd.DataFrame
    initial_x: np.ndarray
    emission_x: np.ndarray
    transition_x: np.ndarray
    behavior: np.ndarray


@dataclass
class PKHMM:
    k: int
    initial_coef: np.ndarray
    transition_coef: np.ndarray
    emission_coef: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool


class NullProgress:
    """Dependency-free progress-bar fallback used when tqdm is unavailable."""

    def __init__(self, iterable=None, total: int | None = None, **_: object) -> None:
        self.iterable = iterable
        self.total = total

    def __iter__(self):
        return iter(self.iterable if self.iterable is not None else [])

    def update(self, _: int = 1) -> None:
        pass

    def set_postfix(self, **_: object) -> None:
        pass

    def set_description(self, _: str) -> None:
        pass

    def close(self) -> None:
        pass


def progress_bar(iterable=None, **kwargs):
    """Create a tqdm progress bar or use the fallback implementation."""
    if tqdm is None:
        return NullProgress(iterable=iterable, **kwargs)
    return tqdm(iterable=iterable, **kwargs)


def progress_write(message: str) -> None:
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)


def pad_and_smooth(
    values: np.ndarray,
    window_radius: int = FILTER_WINDOW_RADIUS,
) -> np.ndarray:
    """Smooth a sequence with an edge-padded moving average."""
    if window_radius <= 0 or len(values) == 0:
        return np.asarray(values, dtype=float)
    padded = np.pad(values, (window_radius, window_radius), mode="edge")
    window_size = 2 * window_radius + 1
    return np.array(
        [padded[i : i + window_size].mean() for i in range(len(values))],
        dtype=float,
    )


def bootstrap_mean_distribution(
    values: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Bootstrap a sample mean and return its empirical distribution."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=float)
    if len(values) == 1:
        return np.repeat(values[0], n_bootstrap).astype(float)
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    return values[sample_index].mean(axis=1)


def wasserstein_similarity_label(distance: float) -> str:
    """Return a descriptive similarity level on the 0-1 probability scale."""
    if distance <= 0.01:
        return "very high"
    if distance <= 0.03:
        return "high"
    if distance <= 0.05:
        return "moderate"
    if distance <= 0.10:
        return "low"
    return "very low"


def bootstrap_distribution_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    bins: np.ndarray,
) -> dict[str, float | str]:
    """Compare observed and predicted bootstrap mean distributions."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    distance = float(wasserstein_distance(observed, predicted))
    observed_hist, _ = np.histogram(observed, bins=bins)
    predicted_hist, _ = np.histogram(predicted, bins=bins)
    observed_mass = observed_hist / observed_hist.sum()
    predicted_mass = predicted_hist / predicted_hist.sum()
    overlap = float(np.minimum(observed_mass, predicted_mass).sum())
    observed_std = float(np.std(observed, ddof=1))
    predicted_std = float(np.std(predicted, ddof=1))
    pooled_std = np.sqrt((observed_std**2 + predicted_std**2) / 2.0)
    return {
        "wasserstein_distance": distance,
        "wasserstein_similarity": max(0.0, 1.0 - distance),
        "histogram_overlap": overlap,
        "similarity_level": wasserstein_similarity_label(distance),
        "observed_mean": float(np.mean(observed)),
        "predicted_mean": float(np.mean(predicted)),
        "predicted_minus_observed_mean": float(
            np.mean(predicted) - np.mean(observed)
        ),
        "observed_std": observed_std,
        "predicted_std": predicted_std,
        "standardized_wasserstein": (
            float(distance / pooled_std) if pooled_std > 1e-12 else np.nan
        ),
    }


def write_bootstrap_distribution_report(
    output_path: Path,
    model_label: str,
    rows: list[dict[str, object]],
) -> None:
    """Write distribution-distance metrics used by Figure 08."""
    lines = [
        f"Model: {model_label}",
        f"Bootstrap repetitions: {BOOTSTRAP_SAMPLES}",
        "A Wasserstein distance closer to zero indicates greater similarity.",
        "Wasserstein similarity = 1 - Wasserstein distance.",
        "Histogram overlap ranges from zero to one; larger values indicate more overlap.",
        (
            "Descriptive thresholds: <=0.01 very high, <=0.03 high, "
            "<=0.05 moderate, <=0.10 low, and >0.10 very low. "
            "These labels are not significance tests."
        ),
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"[{row['feedback']}]",
                f"Observed participant-task samples: {int(row['n_samples'])}",
                f"Wasserstein Distance: {row['wasserstein_distance']:.6f}",
                f"Wasserstein Similarity: {row['wasserstein_similarity']:.6f}",
                f"Histogram Overlap: {row['histogram_overlap']:.6f}",
                f"Descriptive similarity level: {row['similarity_level']}",
                f"Observed bootstrap mean: {row['observed_mean']:.6f}",
                f"Predicted bootstrap mean: {row['predicted_mean']:.6f}",
                (
                    "Predicted mean - observed mean: "
                    f"{row['predicted_minus_observed_mean']:.6f}"
                ),
                f"Observed bootstrap SD: {row['observed_std']:.6f}",
                f"Predicted bootstrap SD: {row['predicted_std']:.6f}",
                (
                    "Standardized Wasserstein distance: "
                    f"{row['standardized_wasserstein']:.6f}"
                ),
                "",
            ]
        )

    if rows:
        mean_distance = float(
            np.mean([float(row["wasserstein_distance"]) for row in rows])
        )
        mean_overlap = float(
            np.mean([float(row["histogram_overlap"]) for row in rows])
        )
        lines.extend(
            [
                "[Macro average across four feedback categories]",
                f"Mean Wasserstein distance: {mean_distance:.6f}",
                f"Mean Wasserstein similarity: {1.0 - mean_distance:.6f}",
                f"Mean histogram overlap: {mean_overlap:.6f}",
                f"Descriptive similarity level: {wasserstein_similarity_label(mean_distance)}",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_subject_prediction_performance(
    prediction_df: pd.DataFrame,
    result_dir: Path,
) -> None:
    """Export participant-level acceptance-prediction performance."""
    table = prediction_df.copy()
    table["predicted_accept_label"] = (
        table["predicted_accept_probability"]
        >= ACCEPTANCE_CLASSIFICATION_THRESHOLD
    ).astype(int)
    table["acceptance_prediction_correct"] = table[
        "predicted_accept_label"
    ].eq(table["observed_accept"]).astype(int)
    table["prediction_error"] = (
        table["observed_accept"] - table["predicted_accept_probability"]
    )

    group_columns = ["subject", "participant_id"]
    if "ai_version" in table.columns:
        group_columns.append("ai_version")

    performance = (
        table.groupby(group_columns, dropna=False)
        .agg(
            n_trials=("observed_accept", "size"),
            observed_acceptance_rate=("observed_accept", "mean"),
            mean_predicted_accept_probability=(
                "predicted_accept_probability",
                "mean",
            ),
            acceptance_accuracy_at_threshold=(
                "acceptance_prediction_correct",
                "mean",
            ),
            acceptance_log_loss=(
                "predicted_accept_probability",
                lambda values: float(
                    -np.mean(
                        table.loc[values.index, "observed_accept"].to_numpy(
                            dtype=float
                        )
                        * np.log(np.clip(values.to_numpy(dtype=float), 1e-12, 1.0))
                        + (
                            1.0
                            - table.loc[
                                values.index, "observed_accept"
                            ].to_numpy(dtype=float)
                        )
                        * np.log(
                            np.clip(
                                1.0 - values.to_numpy(dtype=float),
                                1e-12,
                                1.0,
                            )
                        )
                    )
                ),
            ),
            acceptance_brier_score=(
                "prediction_error",
                lambda values: float(np.mean(values.to_numpy(dtype=float) ** 2)),
            ),
        )
        .reset_index()
    )
    performance.insert(0, "row_type", "subject")
    performance["acceptance_classification_threshold"] = (
        ACCEPTANCE_CLASSIFICATION_THRESHOLD
    )

    accuracy = performance["acceptance_accuracy_at_threshold"]
    summary_rows = []
    for row_type, value in [
        ("mean", float(accuracy.mean())),
        ("std", float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0),
    ]:
        row = {column: np.nan for column in performance.columns}
        row["row_type"] = row_type
        row["subject"] = f"ALL_SUBJECTS_{row_type.upper()}"
        row["n_trials"] = int(performance["n_trials"].sum())
        row["acceptance_classification_threshold"] = (
            ACCEPTANCE_CLASSIFICATION_THRESHOLD
        )
        row["acceptance_accuracy_at_threshold"] = value
        summary_rows.append(row)

    pd.concat(
        [performance, pd.DataFrame(summary_rows)],
        ignore_index=True,
    ).to_csv(result_dir / "prediction_performance.csv", index=False)


def setup_fonts() -> tuple[fm.FontProperties, fm.FontProperties]:
    """
    Load available font files and configure the global font family.

    Arial is preferred when available, with DejaVu Sans as the fallback.
    """
    font_candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    chosen_font = None
    fallback_font_names: list[str] = []
    for font_path in font_candidates:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            if font_name not in fallback_font_names:
                fallback_font_names.append(font_name)
            if chosen_font is None:
                chosen_font = fm.FontProperties(
                    fname=font_path,
                    size=FONT_SIZE,
                )

    if chosen_font is None:
        fallback_font_names = ["DejaVu Sans"]
        chosen_font = fm.FontProperties(
            family="DejaVu Sans",
            size=FONT_SIZE,
        )
        progress_write("[Warning] Preferred fonts were unavailable; using DejaVu Sans.")

    plt.rcParams["font.sans-serif"] = fallback_font_names
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = FONT_SIZE
    return chosen_font, chosen_font


def setup_plot_style() -> None:
    """Configure consistent fonts, sizes, and output resolution."""
    setup_fonts()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).replace("\t", "").strip()
    return text or "unknown"


def safe_filename(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", text)
    return text.strip("._") or "unknown"


def normalize_user_category(column: str, value: object) -> str:
    """Validate the normalized English demographic categories."""
    text = clean_text(value)
    return text if text in INITIAL_CATEGORY_LEVELS[column] else "unknown"


def resolve_user_file() -> Path:
    """Return the anonymized demographic table shipped with the release."""
    if not DEFAULT_USER_FILE.exists():
        raise FileNotFoundError(
            f"Missing release data file: {DEFAULT_USER_FILE}\n"
            "Download and extract the Figshare data archive, then set "
            "HLC_DATA_DIR to the extracted _DATA directory."
        )
    return DEFAULT_USER_FILE


def load_users(user_file: Path) -> pd.DataFrame:
    users = pd.read_csv(user_file).rename(
        columns={"llm_experience": "LLM_experience", "age_group": "age"}
    )
    users["subject"] = users["participant_id"].map(clean_text)
    for column in INITIAL_CATEGORY_LEVELS:
        users[column] = users[column].map(
            lambda value, name=column: normalize_user_category(name, value)
        )
    return users.drop_duplicates("subject", keep="last")


def load_scenario_statistics() -> pd.DataFrame:
    if not SCENARIO_FILE.exists():
        raise FileNotFoundError(
            f"Missing release data file: {SCENARIO_FILE}\n"
            "Download and extract the Figshare data archive, then set "
            "HLC_DATA_DIR to the extracted _DATA directory."
        )
    required = [
        "filename",
        *SCENE_FEATURES,
        "type",
        "correct_answer",
        *AI_OUTPUT_COLUMNS.values(),
    ]
    scenarios = pd.read_csv(SCENARIO_FILE, encoding="utf-8-sig")
    missing = [column for column in required if column not in scenarios.columns]
    if missing:
        raise KeyError(f"scenario_statistics.csv is missing required columns: {missing}")

    scenarios["filename"] = scenarios["filename"].map(clean_text)
    scenarios = scenarios[scenarios["filename"].str.startswith("scen")].copy()
    duplicated = scenarios["filename"].duplicated(keep=False)
    if duplicated.any():
        names = sorted(scenarios.loc[duplicated, "filename"].unique())
        raise ValueError(f"scenario_statistics.csv contains duplicate scenarios: {names}")

    numeric_columns = [
        *SCENE_FEATURES,
        "correct_answer",
        *AI_OUTPUT_COLUMNS.values(),
    ]
    for column in numeric_columns:
        scenarios[column] = pd.to_numeric(scenarios[column], errors="coerce")
    return scenarios[required]


def load_trial_tables(
    scenarios: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    tables: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    trial_files = sorted(TRIAL_DIR.glob("P*.csv"))
    if not trial_files:
        raise FileNotFoundError(
            f"No participant trial files were found under {TRIAL_DIR}\n"
            "Download and extract the Figshare data archive, then set "
            "HLC_DATA_DIR to the extracted _DATA directory."
        )

    for path in trial_files:
        if path.name.startswith(("~$", ".~")):
            continue
        raw = pd.read_csv(path)
        subject = clean_text(raw["participant_id"].dropna().iloc[0])
        ai_version = clean_text(raw["ai_version"].dropna().iloc[0])
        record: dict[str, object] = {
            "file": path.name,
            "subject": subject,
            "ai_version": ai_version,
            "raw_rows": len(raw),
            "excluded_reason": "",
        }

        if ai_version.lower() not in AI_OUTPUT_COLUMNS:
            record["excluded_reason"] = (
                f"Condition not modeled: {ai_version}; expected one of "
                f"{sorted(AI_OUTPUT_COLUMNS)}"
            )
            audit.append(record)
            continue

        # 0 = reject, 1 = accept, and 2 = timeout treated as acceptance.
        valid = raw[raw["decision_type"].isin([0, 1, 2])].copy()
        valid["collaboration_round"] = pd.to_numeric(
            valid["collaboration_round"], errors="coerce"
        )
        valid = valid.dropna(subset=["collaboration_round", "Filename"])
        valid["collaboration_round"] = valid["collaboration_round"].astype(int)
        valid = valid.sort_values("collaboration_round").reset_index(drop=True)

        duplicate_rounds = sorted(
            valid.loc[
                valid["collaboration_round"].duplicated(keep=False),
                "collaboration_round",
            ].unique()
        )
        actual_rounds = set(valid["collaboration_round"])
        missing_rounds = sorted(EXPECTED_INTERACTION_ROUNDS - actual_rounds)
        unexpected_rounds = sorted(actual_rounds - EXPECTED_INTERACTION_ROUNDS)
        record["interaction_rows"] = len(valid)
        record["missing_interaction_rounds"] = ",".join(map(str, missing_rounds))
        record["unexpected_interaction_rounds"] = ",".join(map(str, unexpected_rounds))
        record["duplicate_interaction_rounds"] = ",".join(map(str, duplicate_rounds))
        if actual_rounds != EXPECTED_INTERACTION_ROUNDS or duplicate_rounds:
            record["excluded_reason"] = (
                "Incomplete interaction sequence: "
                f"missing={missing_rounds}, unexpected={unexpected_rounds}, "
                f"duplicates={duplicate_rounds}"
            )
            audit.append(record)
            continue

        valid["Filename"] = valid["Filename"].map(clean_text)
        valid = valid.merge(
            scenarios,
            how="left",
            left_on="Filename",
            right_on="filename",
            validate="many_to_one",
        )
        ai_output_column = AI_OUTPUT_COLUMNS[ai_version.lower()]
        valid["ai_output_column"] = ai_output_column
        valid["ai_advice"] = valid[ai_output_column]
        valid["behavior"] = valid["decision_type"].replace({2: 1}).astype(int)
        valid["behavior_observed"] = True
        valid["ai_correct"] = valid["ai_advice"].eq(valid["correct_answer"]).astype(int)
        valid["human_final_decision"] = np.where(
            valid["behavior"].eq(1),
            valid["ai_advice"],
            3 - valid["ai_advice"],
        ).astype(int)
        valid["human_final_correct"] = valid["human_final_decision"].eq(
            valid["correct_answer"]
        ).astype(int)

        valid["feedback"] = np.select(
            [
                valid["ai_correct"].eq(1) & valid["behavior"].eq(1),
                valid["ai_correct"].eq(0) & valid["behavior"].eq(0),
                valid["ai_correct"].eq(0) & valid["behavior"].eq(1),
                valid["ai_correct"].eq(1) & valid["behavior"].eq(0),
            ],
            [0, 1, 2, 3],
        ).astype(int)
        valid["subject"] = subject
        valid["source_file"] = path.name

        required_scene_columns = [
            "filename",
            *SCENE_FEATURES,
            "correct_answer",
            "ai_advice",
        ]
        missing_scene = int(valid[required_scene_columns].isna().any(axis=1).sum())
        record["scenario_rows_unmatched_or_incomplete"] = missing_scene
        if missing_scene:
            valid = valid.dropna(subset=required_scene_columns).reset_index(drop=True)
        score_correct = valid["round_score"].eq(0).to_numpy()
        record["score_rows_checked"] = len(valid)
        record["score_mismatch_rows"] = int(
            np.sum(valid["human_final_correct"].to_numpy(dtype=bool) != score_correct)
        )
        record["timeout_rows_treated_as_accept"] = int(
            valid["decision_type"].eq(2).sum()
        )
        record["usable_rows"] = len(valid)
        if missing_scene:
            record["excluded_reason"] = "Missing scenario statistics or AI advice"
        elif len(valid) != N_INTERACTIONS_PER_SUBJECT:
            record["excluded_reason"] = (
                f"Expected {N_INTERACTIONS_PER_SUBJECT} interactions, found {len(valid)}"
            )
        else:
            tables.append(valid)
        audit.append(record)

    return tables, audit


def standardize_scene_features(
    tables: list[pd.DataFrame],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    all_scene = pd.concat([table[SCENE_FEATURES] for table in tables], ignore_index=True)
    scale_series = all_scene.std(ddof=0)
    active_features = scale_series[scale_series > 1e-12].index.tolist()
    mean = all_scene[active_features].mean().to_numpy(dtype=float)
    scale = scale_series[active_features].to_numpy(dtype=float)
    return active_features, mean, scale


def initial_feature_frame(users: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    """Build IPM indicators with the first category as the reference."""
    columns = list(INITIAL_CATEGORY_LEVELS)
    indexed = users.set_index("subject")
    rows = []
    for subject in subjects:
        if subject in indexed.index:
            row = indexed.loc[subject, columns]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            rows.append({column: clean_text(row[column]) for column in columns})
        else:
            rows.append({column: "unknown" for column in columns})
    frame = pd.DataFrame(rows, index=subjects)
    for column, levels in INITIAL_CATEGORY_LEVELS.items():
        frame[column] = pd.Categorical(frame[column], categories=levels)
    return pd.get_dummies(frame, columns=columns, drop_first=True, dtype=float)


def build_sequences(
    tables: list[pd.DataFrame], users: pd.DataFrame
) -> tuple[list[SequenceData], dict[str, object]]:
    active_scene_features, scene_mean, scene_scale = standardize_scene_features(tables)
    subjects = [clean_text(table["subject"].iloc[0]) for table in tables]
    initial_frame = initial_feature_frame(users, subjects)
    demographic_matrix = initial_frame.to_numpy(dtype=float)

    sequences: list[SequenceData] = []
    for i, table in enumerate(tables):
        scene_x = (
            table[active_scene_features].to_numpy(dtype=float) - scene_mean
        ) / scene_scale
        previous_feedback = np.zeros((len(table), len(FEEDBACK_NAMES)), dtype=float)
        if len(table) > 1:
            previous_feedback[
                np.arange(1, len(table)), table["feedback"].to_numpy(dtype=int)[:-1]
            ] = 1.0
        model_x = np.column_stack([scene_x, previous_feedback])
        initial_x = np.concatenate([demographic_matrix[i], scene_x[0]])
        participant_id = clean_text(table["participant_id"].iloc[0])
        sequences.append(
            SequenceData(
                subject=subjects[i],
                participant_id=participant_id,
                trials=table,
                initial_x=initial_x,
                emission_x=model_x,
                transition_x=model_x,
                behavior=table["behavior"].to_numpy(dtype=int),
            )
        )

    metadata = {
        "scene_features_available": SCENE_FEATURES,
        "scene_features_used": active_scene_features,
        "n_interactions_per_subject": N_INTERACTIONS_PER_SUBJECT,
        "constant_scene_features_removed": sorted(
            set(SCENE_FEATURES) - set(active_scene_features)
        ),
        "scene_mean": scene_mean.tolist(),
        "scene_scale": scene_scale.tolist(),
        "previous_feedback_features": FEEDBACK_NAMES,
        "model_features": active_scene_features + FEEDBACK_NAMES,
        "initial_demographic_features": initial_frame.columns.tolist(),
        "initial_scene_features": [
            f"x1_{feature}" for feature in active_scene_features
        ],
        "initial_features": initial_frame.columns.tolist()
        + [f"x1_{feature}" for feature in active_scene_features],
        "initial_category_reference_levels": {
            column: levels[0] for column, levels in INITIAL_CATEGORY_LEVELS.items()
        },
    }
    return sequences, metadata


def training_data_signature(sequences: list[SequenceData]) -> str:
    """Build a stable fingerprint of the sequences supplied to the model."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "n_interactions_per_subject": N_INTERACTIONS_PER_SUBJECT,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    for sequence in sorted(sequences, key=lambda item: item.subject):
        payload = {
            "subject": sequence.subject,
            "participant_id": sequence.participant_id,
            "interaction_rounds": sequence.trials["collaboration_round"].astype(int).tolist(),
            "behavior": sequence.behavior.astype(int).tolist(),
            # Eight decimals detect meaningful changes while ignoring harmless
            # machine-precision differences across BLAS and pandas versions.
            "initial_x": np.round(sequence.initial_x, 8).tolist(),
            "emission_x": np.round(sequence.emission_x, 8).tolist(),
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
    return digest.hexdigest()


def current_training_configuration() -> dict[str, object]:
    """Return settings that affect estimation or the candidate model set."""
    return {
        "k_candidates": list(K_CANDIDATES),
        "n_starts": N_STARTS,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "l2": L2,
        "random_seed": RANDOM_SEED,
    }


def existing_training_snapshot_matches(
    sequences: list[SequenceData],
    metadata: dict[str, object],
    result_dir: Path,
) -> bool:
    """Compare an existing training snapshot with the current input."""
    data_dir = result_dir / "data"
    metadata_path = data_dir / "_training_data_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    exact_fields = {
        "n_subjects": len(sequences),
        "n_trials": int(sum(len(sequence.behavior) for sequence in sequences)),
        "n_interactions_per_subject": N_INTERACTIONS_PER_SUBJECT,
        "model_features": list(metadata["model_features"]),
        "initial_features": list(metadata["initial_features"]),
    }
    if any(saved.get(key) != value for key, value in exact_fields.items()):
        return False
    if not np.allclose(
        saved.get("scene_mean", []), metadata["scene_mean"], atol=1e-9, rtol=0
    ) or not np.allclose(
        saved.get("scene_scale", []), metadata["scene_scale"], atol=1e-9, rtol=0
    ):
        return False

    model_columns = [f"model_x_{name}" for name in metadata["model_features"]]
    initial_columns = [
        f"initial_x_{name}" for name in metadata["initial_features"]
    ]
    for sequence in sequences:
        snapshot_path = data_dir / f"{safe_filename(sequence.subject)}.csv"
        if not snapshot_path.exists():
            return False
        try:
            snapshot = pd.read_csv(snapshot_path, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError):
            return False
        required_columns = {
            "collaboration_round",
            "behavior_accept",
            "ai_version",
            *model_columns,
            *initial_columns,
        }
        if len(snapshot) != len(sequence.behavior) or not required_columns.issubset(
            snapshot.columns
        ):
            return False
        if not np.array_equal(
            snapshot["collaboration_round"].to_numpy(dtype=int),
            sequence.trials["collaboration_round"].to_numpy(dtype=int),
        ) or not np.array_equal(
            snapshot["behavior_accept"].to_numpy(dtype=int), sequence.behavior
        ):
            return False
        current_ai_version = clean_text(
            sequence.trials["ai_version"].iloc[0]
        ).lower()
        if not snapshot["ai_version"].map(clean_text).str.lower().eq(
            current_ai_version
        ).all():
            return False
        if not np.allclose(
            snapshot[model_columns].to_numpy(dtype=float),
            sequence.emission_x,
            atol=1e-8,
            rtol=0,
        ) or not np.allclose(
            snapshot.loc[0, initial_columns].to_numpy(dtype=float),
            sequence.initial_x,
            atol=1e-8,
            rtol=0,
        ):
            return False
    return True


def export_training_data_snapshot(
    sequences: list[SequenceData],
    metadata: dict[str, object],
    result_dir: Path,
    ai_version: str,
) -> None:
    """
    Export the exact PKHMM input before model fitting.

    Each participant receives one CSV. Compatible existing snapshots are reused.
    """
    data_dir = result_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    model_feature_names = list(metadata["model_features"])
    initial_feature_names = list(metadata["initial_features"])
    expected_files = [
        data_dir / f"{safe_filename(sequence.subject)}.csv"
        for sequence in sequences
    ]
    metadata_path = data_dir / "_training_data_metadata.json"
    snapshot_config = {
        "n_interactions_per_subject": N_INTERACTIONS_PER_SUBJECT,
        "training_data_signature": metadata["training_data_signature"],
        "training_data_signature_version": metadata[
            "training_data_signature_version"
        ],
    }
    existing_snapshot_metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            existing_snapshot_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            pass
    snapshot_matches = all(
        existing_snapshot_metadata.get(key) == value
        for key, value in snapshot_config.items()
    )
    if all(path.exists() for path in expected_files) and snapshot_matches:
        progress_write(f"[{ai_version}] Compatible training snapshot found; skipping export: {data_dir}")
        return

    rewrite_all = not snapshot_matches

    written = 0
    for sequence, output_path in zip(sequences, expected_files):
        if output_path.exists() and not rewrite_all:
            continue

        table = sequence.trials.copy().reset_index(drop=True)
        snapshot = pd.DataFrame(
            {
                "subject": sequence.subject,
                "participant_id": sequence.participant_id,
                "trial_index": np.arange(1, len(table) + 1),
                "collaboration_round": table["collaboration_round"],
                "scene_id": table["scene_id"],
                "filename": table["Filename"],
                "source_file": table["source_file"],
                "ai_version": table["ai_version"],
                "ai_output_column": table["ai_output_column"],
                "decision_type": table["decision_type"],
                "behavior_accept": sequence.behavior,
                "behavior_observed": table["behavior_observed"].astype(int),
                "correct_answer": table["correct_answer"],
                "ai_advice": table["ai_advice"],
                "ai_correct": table["ai_correct"],
                "human_final_decision": table["human_final_decision"],
                "human_final_correct": table["human_final_correct"],
                "feedback": table["feedback"],
                "previous_feedback": [
                    -1 if i == 0 else int(table.loc[i - 1, "feedback"])
                    for i in range(len(table))
                ],
                "round_score": table["round_score"],
            }
        )

        for feature in metadata["scene_features_available"]:
            if feature in table.columns:
                snapshot[f"raw_{feature}"] = table[feature]

        for feature, values in zip(model_feature_names, sequence.emission_x.T):
            snapshot[f"model_x_{feature}"] = values

        for feature, value in zip(initial_feature_names, sequence.initial_x):
            snapshot[f"initial_x_{feature}"] = value

        snapshot.to_csv(output_path, index=False, encoding="utf-8-sig")
        written += 1

    snapshot_metadata = {
        "description": (
            "Each CSV stores one participant's PKHMM training input. "
            "model_x_* contains current-scenario and previous-feedback inputs for the PUM/CDM; "
            "initial_x_* contains IPM inputs."
        ),
        "ai_version": ai_version,
        "n_subjects": len(sequences),
        "n_trials": int(sum(len(sequence.behavior) for sequence in sequences)),
        "subject_files": [path.name for path in expected_files],
        "model_features": model_feature_names,
        "initial_features": initial_feature_names,
        "scene_features_available": metadata["scene_features_available"],
        "scene_features_used": metadata["scene_features_used"],
        "scene_mean": metadata["scene_mean"],
        "scene_scale": metadata["scene_scale"],
        "previous_feedback_features": metadata["previous_feedback_features"],
        "initial_category_reference_levels": metadata[
            "initial_category_reference_levels"
        ],
        **snapshot_config,
    }
    metadata_path.write_text(
        json.dumps(snapshot_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    progress_write(
        f"[{ai_version}] Training snapshot export complete: {written} files written to {data_dir}"
    )


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def softmax_prob(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    logits = add_intercept(x) @ coef.T
    logits = np.column_stack([logits, np.zeros(len(x))])
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def fit_softmax(
    x: np.ndarray,
    fractional_targets: np.ndarray,
    l2: float,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Fit multinomial logistic regression with a zero reference class."""
    n_classes = fractional_targets.shape[1]
    p = x.shape[1] + 1
    x1 = add_intercept(x)
    initial = (
        np.zeros((n_classes - 1, p), dtype=float)
        if start is None
        else start.copy()
    )

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coef = flat.reshape(n_classes - 1, p)
        logits = np.column_stack([x1 @ coef.T, np.zeros(len(x))])
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        value = -float(np.sum(fractional_targets * log_prob))
        value += 0.5 * l2 * float(np.sum(coef[:, 1:] ** 2))
        prob = np.exp(log_prob)
        row_weight = fractional_targets.sum(axis=1, keepdims=True)
        grad = (prob[:, :-1] * row_weight - fractional_targets[:, :-1]).T @ x1
        grad[:, 1:] += l2 * coef[:, 1:]
        return value, grad.ravel()

    result = minimize(
        objective,
        initial.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        warnings.warn(f"Multinomial optimization did not fully converge: {result.message}", RuntimeWarning)
    return result.x.reshape(n_classes - 1, p)


def fit_binary(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    l2: float,
    start: np.ndarray | None = None,
) -> np.ndarray:
    x1 = add_intercept(x)
    initial = np.zeros(x1.shape[1]) if start is None else start.copy()

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        linear = x1 @ coef
        value = float(
            np.sum(weights * (np.logaddexp(0.0, linear) - y * linear))
        )
        value += 0.5 * l2 * float(np.sum(coef[1:] ** 2))
        grad = x1.T @ (weights * (expit(linear) - y))
        grad[1:] += l2 * coef[1:]
        return value, grad

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        warnings.warn(f"Binary optimization did not fully converge: {result.message}", RuntimeWarning)
    return result.x


def sequence_probabilities(
    model: PKHMM, sequence: SequenceData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = softmax_prob(sequence.initial_x[None, :], model.initial_coef)[0]
    emission_accept = expit(
        add_intercept(sequence.emission_x) @ model.emission_coef.T
    )
    behavior = sequence.behavior[:, None]
    emission = np.where(behavior == 1, emission_accept, 1.0 - emission_accept)
    emission = np.clip(emission, 1e-12, 1.0)

    transition = np.empty((len(sequence.behavior), model.k, model.k))
    transition[0] = np.eye(model.k)
    for origin in range(model.k):
        transition[1:, origin, :] = softmax_prob(
            sequence.transition_x[1:], model.transition_coef[origin]
        )
    return initial, transition, emission


def forward_backward(
    model: PKHMM, sequence: SequenceData
) -> tuple[float, np.ndarray, np.ndarray]:
    initial, transition, emission = sequence_probabilities(model, sequence)
    t_len, k = emission.shape
    alpha = np.zeros((t_len, k))
    scales = np.zeros(t_len)
    alpha[0] = initial * emission[0]
    scales[0] = max(alpha[0].sum(), 1e-300)
    alpha[0] /= scales[0]

    for t in range(1, t_len):
        alpha[t] = (alpha[t - 1] @ transition[t]) * emission[t]
        scales[t] = max(alpha[t].sum(), 1e-300)
        alpha[t] /= scales[t]

    beta = np.ones((t_len, k))
    for t in range(t_len - 2, -1, -1):
        beta[t] = transition[t + 1] @ (emission[t + 1] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((max(t_len - 1, 0), k, k))
    for t in range(t_len - 1):
        value = (
            alpha[t, :, None]
            * transition[t + 1]
            * (emission[t + 1] * beta[t + 1])[None, :]
        )
        xi[t] = value / max(value.sum(), 1e-300)
    return float(np.log(scales).sum()), gamma, xi


def filtered_predictions(
    model: PKHMM, sequence: SequenceData
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform online filtering and one-step-ahead prediction by round.

    prior_state[t] uses only information available before behavior at round t;
    filtered_state[t] additionally incorporates the observed behavior at t.
    """
    initial, transition, emission = sequence_probabilities(model, sequence)
    accept_by_state = expit(
        add_intercept(sequence.emission_x) @ model.emission_coef.T
    )
    t_len = len(sequence.behavior)
    prior_state = np.zeros((t_len, model.k))
    filtered_state = np.zeros((t_len, model.k))
    predicted_accept = np.zeros(t_len)
    predicted_safety = np.zeros(t_len)

    for t in range(t_len):
        prior_state[t] = (
            initial if t == 0 else filtered_state[t - 1] @ transition[t]
        )
        prior_state[t] /= max(prior_state[t].sum(), 1e-300)
        predicted_accept[t] = prior_state[t] @ accept_by_state[t]

        # In the binary task, safe behavior means accepting correct AI advice or rejecting incorrect AI advice.
        ai_correct = int(sequence.trials.iloc[t]["ai_correct"])
        predicted_safety[t] = (
            predicted_accept[t] if ai_correct else 1.0 - predicted_accept[t]
        )

        filtered_state[t] = prior_state[t] * emission[t]
        filtered_state[t] /= max(filtered_state[t].sum(), 1e-300)

    return prior_state, filtered_state, predicted_accept, predicted_safety


def initialize_model(
    sequences: list[SequenceData], k: int, rng: np.random.Generator
) -> PKHMM:
    p_initial = len(sequences[0].initial_x) + 1
    p_model = sequences[0].emission_x.shape[1] + 1
    initial_coef = rng.normal(0, 0.15, size=(k - 1, p_initial))
    transition_coef = rng.normal(0, 0.15, size=(k, k - 1, p_model))

    # Break initial state-label symmetry with distinct acceptance tendencies.
    emission_coef = rng.normal(0, 0.08, size=(k, p_model))
    emission_coef[:, 0] = np.linspace(-1.5, 1.5, k)
    return PKHMM(
        k=k,
        initial_coef=initial_coef,
        transition_coef=transition_coef,
        emission_coef=emission_coef,
        log_likelihood=-np.inf,
        n_iter=0,
        converged=False,
    )


def fit_one_model(
    sequences: list[SequenceData],
    k: int,
    rng: np.random.Generator,
    progress_label: str,
) -> PKHMM:
    model = initialize_model(sequences, k, rng)
    previous_ll = -np.inf
    em_progress = progress_bar(
        total=MAX_ITER,
        desc=progress_label,
        unit="iter",
        position=2,
        leave=False,
        dynamic_ncols=True,
    )
    try:
        for iteration in range(1, MAX_ITER + 1):
            posteriors = [forward_backward(model, sequence) for sequence in sequences]
            current_ll = sum(item[0] for item in posteriors)

            initial_x = np.vstack([sequence.initial_x for sequence in sequences])
            initial_targets = np.vstack([item[1][0] for item in posteriors])
            model.initial_coef = fit_softmax(
                initial_x, initial_targets, L2, model.initial_coef
            )

            transition_x = np.vstack(
                [sequence.transition_x[1:] for sequence in sequences]
            )
            for origin in range(k):
                targets = np.vstack([item[2][:, origin, :] for item in posteriors])
                model.transition_coef[origin] = fit_softmax(
                    transition_x,
                    targets,
                    L2,
                    model.transition_coef[origin],
                )

            emission_x = np.vstack([sequence.emission_x for sequence in sequences])
            behavior = np.concatenate([sequence.behavior for sequence in sequences])
            gamma = np.vstack([item[1] for item in posteriors])
            for state in range(k):
                model.emission_coef[state] = fit_binary(
                    emission_x,
                    behavior,
                    gamma[:, state],
                    L2,
                    model.emission_coef[state],
                )

            updated_ll = sum(
                forward_backward(model, sequence)[0] for sequence in sequences
            )
            delta = (
                np.nan if not np.isfinite(previous_ll) else updated_ll - previous_ll
            )
            model.log_likelihood = updated_ll
            model.n_iter = iteration
            em_progress.update(1)
            em_progress.set_postfix(
                logLik=f"{updated_ll:.2f}",
                delta="-" if np.isnan(delta) else f"{delta:.3g}",
            )

            if iteration > 1 and abs(updated_ll - previous_ll) <= TOL * (
                1.0 + abs(previous_ll)
            ):
                model.converged = True
                break
            previous_ll = updated_ll

            if updated_ll + 1e-6 < current_ll:
                warnings.warn(
                    f"K={k} likelihood decreased slightly after an EM update; regularized EM is not guaranteed to be strictly monotonic.",
                    RuntimeWarning,
                )
    finally:
        em_progress.close()
    return model


def parameter_count(model: PKHMM) -> int:
    return int(
        model.initial_coef.size
        + model.transition_coef.size
        + model.emission_coef.size
    )


def bic(model: PKHMM, n_observations: int) -> float:
    return -2.0 * model.log_likelihood + parameter_count(model) * np.log(
        n_observations
    )


def fit_models(
    sequences: list[SequenceData], ai_version: str
) -> tuple[PKHMM, dict[int, PKHMM], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    n_observations = sum(len(sequence.behavior) for sequence in sequences)
    candidates: dict[int, PKHMM] = {}
    records = []

    fit_progress = progress_bar(
        total=len(K_CANDIDATES) * N_STARTS,
        desc=f"{ai_version} fitting tasks",
        unit="fit",
        position=1,
        leave=True,
        dynamic_ncols=True,
    )
    try:
        for k in K_CANDIDATES:
            starts = []
            for start_index in range(1, N_STARTS + 1):
                label = f"{ai_version} K={k} start={start_index}/{N_STARTS}"
                fitted = fit_one_model(sequences, k, rng, label)
                starts.append(fitted)
                fit_progress.update(1)
                fit_progress.set_postfix(
                    K=k,
                    start=f"{start_index}/{N_STARTS}",
                    logLik=f"{fitted.log_likelihood:.2f}",
                )
                progress_write(
                    f"[{ai_version}] K={k}, start={start_index}/{N_STARTS}: "
                    f"logLik={fitted.log_likelihood:.3f}, "
                    f"iterations={fitted.n_iter}, converged={fitted.converged}"
                )
            best = max(starts, key=lambda item: item.log_likelihood)
            candidates[k] = best
            records.append(
                {
                    "K": k,
                    "log_likelihood": best.log_likelihood,
                    "n_parameters": parameter_count(best),
                    "BIC": bic(best, n_observations),
                    "iterations": best.n_iter,
                    "converged": best.converged,
                }
            )
            progress_write(
                f"[{ai_version}] K={k} complete: "
                f"logLik={best.log_likelihood:.3f}, "
                f"BIC={records[-1]['BIC']:.3f}, converged={best.converged}"
            )
    finally:
        fit_progress.close()

    comparison = pd.DataFrame(records).sort_values("BIC").reset_index(drop=True)
    selected_k = int(comparison.iloc[0]["K"])
    return candidates[selected_k], candidates, comparison


def viterbi(model: PKHMM, sequence: SequenceData) -> np.ndarray:
    initial, transition, emission = sequence_probabilities(model, sequence)
    log_emission = np.log(emission)
    t_len, k = emission.shape
    score = np.empty((t_len, k))
    pointer = np.zeros((t_len, k), dtype=int)
    score[0] = np.log(np.clip(initial, 1e-12, 1.0)) + log_emission[0]

    for t in range(1, t_len):
        candidate = score[t - 1, :, None] + np.log(
            np.clip(transition[t], 1e-12, 1.0)
        )
        pointer[t] = np.argmax(candidate, axis=0)
        score[t] = np.max(candidate, axis=0) + log_emission[t]

    path = np.empty(t_len, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(t_len - 2, -1, -1):
        path[t] = pointer[t + 1, path[t + 1]]
    return path


def state_order(model: PKHMM, sequences: list[SequenceData]) -> np.ndarray:
    """Order states by mean predicted acceptance probability."""
    x = np.vstack([sequence.emission_x for sequence in sequences])
    probabilities = expit(add_intercept(x) @ model.emission_coef.T).mean(axis=0)
    return np.argsort(probabilities)


def state_reliance_level(state: int, n_states: int) -> str:
    """Create interpretable state labels ordered by mean reliance."""
    labels = {
        2: ["State 1", "State 2"],
        3: ["State 1", "State 2", "State 3"],
        4: ["State 1", "State 2", "State 3", "State 4"],
    }
    if n_states in labels:
        return labels[n_states][state - 1]
    return f"State {state}"


def export_state_conditioned_reliance_probabilities(
    model: PKHMM,
    sequences: list[SequenceData],
    state_label: dict[int, int],
    result_dir: Path,
) -> None:
    """Export participant reliance with the prior state held fixed."""
    rows: list[dict[str, object]] = []
    for sequence in sequences:
        probabilities = expit(
            add_intercept(sequence.emission_x) @ model.emission_coef.T
        )
        ai_version = clean_text(sequence.trials["ai_version"].iloc[0]).lower()
        for old_state in range(model.k):
            ordered_state = state_label[old_state]
            rows.append(
                {
                    "subject": sequence.subject,
                    "participant_id": sequence.participant_id,
                    "ai_version": ai_version,
                    "state": ordered_state,
                    "reliance_level": state_reliance_level(
                        ordered_state, model.k
                    ),
                    "n_tasks": len(sequence.behavior),
                    "mean_estimated_reliance_probability": float(
                        probabilities[:, old_state].mean()
                    ),
                }
            )

    result = pd.DataFrame(rows)
    state_means = result.groupby("state")[
        "mean_estimated_reliance_probability"
    ].transform("mean")
    result["state_mean_estimated_reliance_probability"] = state_means
    result = result.sort_values(
        ["state_mean_estimated_reliance_probability", "subject"]
    ).reset_index(drop=True)
    result.to_csv(result_dir / "reliance_probability.csv", index=False)


def export_state_selective_reliance_probabilities(
    model: PKHMM,
    sequences: list[SequenceData],
    state_label: dict[int, int],
    result_dir: Path,
) -> None:
    """Export the acceptance-probability difference by AI correctness."""
    rows: list[dict[str, object]] = []
    for sequence in sequences:
        probabilities = expit(
            add_intercept(sequence.emission_x) @ model.emission_coef.T
        )
        ai_correct = sequence.trials["ai_correct"].to_numpy(dtype=int)
        correct_mask = ai_correct == 1
        wrong_mask = ai_correct == 0
        ai_version = clean_text(sequence.trials["ai_version"].iloc[0]).lower()

        for old_state in range(model.k):
            ordered_state = state_label[old_state]
            correct_probability = (
                float(probabilities[correct_mask, old_state].mean())
                if correct_mask.any()
                else np.nan
            )
            wrong_probability = (
                float(probabilities[wrong_mask, old_state].mean())
                if wrong_mask.any()
                else np.nan
            )
            rows.append(
                {
                    "subject": sequence.subject,
                    "participant_id": sequence.participant_id,
                    "ai_version": ai_version,
                    "state": ordered_state,
                    "reliance_level": state_reliance_level(
                        ordered_state, model.k
                    ),
                    "n_ai_correct_tasks": int(correct_mask.sum()),
                    "n_ai_wrong_tasks": int(wrong_mask.sum()),
                    "mean_acceptance_probability_ai_correct": (
                        correct_probability
                    ),
                    "mean_acceptance_probability_ai_wrong": wrong_probability,
                    "selective_reliance_probability": (
                        correct_probability - wrong_probability
                    ),
                }
            )

    result = pd.DataFrame(rows)
    state_means = result.groupby("state")[
        "selective_reliance_probability"
    ].transform("mean")
    result["state_mean_selective_reliance_probability"] = state_means
    result = result.sort_values(["state", "subject"]).reset_index(drop=True)
    result.to_csv(
        result_dir / "selective_reliance_probability.csv", index=False
    )


def export_state_real_selective_reliance_probabilities(
    state_paths: pd.DataFrame,
    ai_version: str,
    k: int,
    result_dir: Path,
) -> None:
    """Export observed selective reliance by participant and Viterbi state."""
    rows: list[dict[str, object]] = []
    grouping_columns = ["subject", "participant_id", "viterbi_state"]
    for (subject, participant_id, state), group in state_paths.groupby(
        grouping_columns,
        sort=True,
    ):
        correct = group.loc[group["ai_correct"].eq(1), "behavior_accept"]
        wrong = group.loc[group["ai_correct"].eq(0), "behavior_accept"]
        correct_probability = float(correct.mean()) if len(correct) else np.nan
        wrong_probability = float(wrong.mean()) if len(wrong) else np.nan
        rows.append(
            {
                "subject": subject,
                "participant_id": participant_id,
                "ai_version": ai_version,
                "state": int(state),
                "reliance_level": state_reliance_level(int(state), k),
                "n_state_tasks": int(len(group)),
                "n_ai_correct_tasks": int(len(correct)),
                "n_ai_wrong_tasks": int(len(wrong)),
                "n_accept_ai_correct": int(correct.sum()),
                "n_accept_ai_wrong": int(wrong.sum()),
                "real_acceptance_probability_ai_correct": correct_probability,
                "real_acceptance_probability_ai_wrong": wrong_probability,
                "real_selective_reliance_probability": (
                    correct_probability - wrong_probability
                ),
            }
        )

    result = pd.DataFrame(rows)
    state_means = result.groupby("state")[
        "real_selective_reliance_probability"
    ].transform("mean")
    result["state_mean_real_selective_reliance_probability"] = state_means
    result = result.sort_values(["state", "subject"]).reset_index(drop=True)
    result.to_csv(
        result_dir / "selective_reliance_probability_real_distribution.csv",
        index=False,
    )


def export_state_real_reliance_probabilities(
    state_paths: pd.DataFrame,
    ai_version: str,
    k: int,
    result_dir: Path,
) -> None:
    """Export observed acceptance rates by participant and Viterbi state."""
    result = (
        state_paths.groupby(
            ["subject", "participant_id", "viterbi_state"],
            as_index=False,
            sort=True,
        )
        .agg(
            n_state_tasks=("behavior_accept", "size"),
            n_accept_tasks=("behavior_accept", "sum"),
            real_reliance_probability=("behavior_accept", "mean"),
        )
        .rename(columns={"viterbi_state": "state"})
    )
    result["state"] = result["state"].astype(int)
    result.insert(2, "ai_version", ai_version)
    result.insert(
        4,
        "reliance_level",
        result["state"].map(lambda state: state_reliance_level(int(state), k)),
    )
    result["state_mean_real_reliance_probability"] = result.groupby("state")[
        "real_reliance_probability"
    ].transform("mean")
    result = result.sort_values(["state", "subject"]).reset_index(drop=True)
    result.to_csv(
        result_dir / "reliance_probability_real_distribution.csv",
        index=False,
    )


def export_results(
    model: PKHMM,
    comparison: pd.DataFrame,
    sequences: list[SequenceData],
    metadata: dict[str, object],
    audit: list[dict[str, object]],
    users: pd.DataFrame,
    ai_version: str,
    result_dir: Path,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(result_dir / "bic_comparison.csv", index=False)
    pd.DataFrame(audit).to_csv(result_dir / "data_audit.csv", index=False)

    order = state_order(model, sequences)
    state_label = {int(old): int(new + 1) for new, old in enumerate(order)}
    export_state_conditioned_reliance_probabilities(
        model, sequences, state_label, result_dir
    )
    export_state_selective_reliance_probabilities(
        model, sequences, state_label, result_dir
    )
    path_rows = []
    prediction_rows = []
    for sequence in sequences:
        _, gamma, _ = forward_backward(model, sequence)
        path = viterbi(model, sequence)
        prior, filtered, predicted_accept, predicted_safety = filtered_predictions(
            model, sequence
        )
        for t, (_, trial) in enumerate(sequence.trials.iterrows()):
            row = {
                "subject": sequence.subject,
                "participant_id": sequence.participant_id,
                "trial_index": t + 1,
                "scene_id": trial["scene_id"],
                "filename": trial["Filename"],
                "behavior_accept": int(sequence.behavior[t]),
                "correct_answer": int(trial["correct_answer"]),
                "ai_advice": int(trial["ai_advice"]),
                "ai_correct": int(trial["ai_correct"]),
                "human_final_decision": int(trial["human_final_decision"]),
                "human_final_correct": int(trial["human_final_correct"]),
                "feedback": int(trial["feedback"]),
                "behavior_observed": 1,
                "viterbi_state": state_label[int(path[t])],
            }
            for old_state in range(model.k):
                row[f"posterior_state_{state_label[old_state]}"] = gamma[t, old_state]
            path_rows.append(row)

            prediction_row = {
                "subject": sequence.subject,
                "participant_id": sequence.participant_id,
                "trial_index": t + 1,
                "scene_id": trial["scene_id"],
                "filename": trial["Filename"],
                "previous_feedback": (
                    -1 if t == 0 else int(sequence.trials.iloc[t - 1]["feedback"])
                ),
                "ai_correct": int(trial["ai_correct"]),
                "human_final_decision": int(trial["human_final_decision"]),
                "human_final_correct": int(trial["human_final_correct"]),
                "behavior_observed": 1,
                "observed_accept": int(sequence.behavior[t]),
                "predicted_accept_probability": predicted_accept[t],
                "predicted_reject_probability": 1.0 - predicted_accept[t],
                "predicted_safety_probability": predicted_safety[t],
            }
            for old_state in range(model.k):
                label = state_label[old_state]
                prediction_row[f"prior_state_{label}"] = prior[t, old_state]
                prediction_row[f"filtered_state_{label}"] = filtered[t, old_state]
            prediction_rows.append(prediction_row)
    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(result_dir / "state_paths.csv", index=False)
    export_state_real_reliance_probabilities(
        path_df,
        ai_version,
        model.k,
        result_dir,
    )
    export_state_real_selective_reliance_probabilities(
        path_df,
        ai_version,
        model.k,
        result_dir,
    )

    reasonable_reliance = (
        (path_df["ai_correct"].eq(1) & path_df["behavior_accept"].eq(1))
        | (path_df["ai_correct"].eq(0) & path_df["behavior_accept"].eq(0))
    )
    overall_results = {
        "n_subjects": int(path_df["subject"].nunique()),
        "n_trials": int(len(path_df)),
        "ai_accuracy": float(path_df["ai_correct"].mean()),
        "human_final_accuracy_collaboration": float(
            path_df["human_final_correct"].mean()
        ),
        "human_acceptance_rate": float(path_df["behavior_accept"].mean()),
        "reasonable_reliance_rate": float(reasonable_reliance.mean()),
    }
    pd.DataFrame([overall_results]).to_csv(
        result_dir / "results.csv", index=False
    )

    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df.to_csv(result_dir / "one_step_predictions.csv", index=False)
    export_subject_prediction_performance(prediction_df, result_dir)
    observed = prediction_df["observed_accept"].to_numpy(dtype=float)
    predicted = np.clip(
        prediction_df["predicted_accept_probability"].to_numpy(dtype=float),
        1e-12,
        1.0 - 1e-12,
    )
    prediction_metrics = {
        "acceptance_log_loss": float(
            -np.mean(observed * np.log(predicted) + (1.0 - observed) * np.log(1.0 - predicted))
        ),
        "acceptance_brier_score": float(np.mean((observed - predicted) ** 2)),
        "acceptance_classification_threshold": ACCEPTANCE_CLASSIFICATION_THRESHOLD,
        "acceptance_accuracy_at_threshold": float(
            np.mean((predicted >= ACCEPTANCE_CLASSIFICATION_THRESHOLD) == observed)
        ),
        "n_behavior_observed": len(prediction_df),
        "n_behavior_missing": 0,
        "mean_predicted_safety_probability": float(
            prediction_df["predicted_safety_probability"].mean()
        ),
    }
    pd.DataFrame([prediction_metrics]).to_csv(
        result_dir / "prediction_metrics.csv", index=False
    )

    model_feature_names = metadata["model_features"]
    emission_rows = []
    for old_state in range(model.k):
        for feature, value in zip(
            ["intercept"] + model_feature_names, model.emission_coef[old_state]
        ):
            emission_rows.append(
                {
                    "state": state_label[old_state],
                    "feature": feature,
                    "coefficient": value,
                    "odds_ratio": np.exp(np.clip(value, -20, 20)),
                }
            )
    pd.DataFrame(emission_rows).sort_values(["state", "feature"]).to_csv(
        result_dir / "emission_coefficients.csv", index=False
    )

    transition_rows = []
    for old_origin in range(model.k):
        for destination_nonbaseline in range(model.k - 1):
            for feature, value in zip(
                ["intercept"] + model_feature_names,
                model.transition_coef[old_origin, destination_nonbaseline],
            ):
                transition_rows.append(
                    {
                        "origin_state": state_label[old_origin],
                        "destination_state": state_label[destination_nonbaseline],
                        "reference_destination_state": state_label[model.k - 1],
                        "feature": feature,
                        "coefficient": value,
                    }
                )
    pd.DataFrame(transition_rows).to_csv(
        result_dir / "transition_coefficients.csv", index=False
    )

    initial_rows = []
    initial_names = metadata["initial_features"]
    for target_nonbaseline in range(model.k - 1):
        for feature, value in zip(
            ["intercept"] + initial_names, model.initial_coef[target_nonbaseline]
        ):
            initial_rows.append(
                {
                    "target_state": state_label[target_nonbaseline],
                    "reference_state": state_label[model.k - 1],
                    "feature": feature,
                    "coefficient": value,
                }
            )
    pd.DataFrame(initial_rows).to_csv(
        result_dir / "initial_coefficients.csv", index=False
    )

    initial_probability_rows = []
    for sequence in sequences:
        probabilities = softmax_prob(
            sequence.initial_x[None, :], model.initial_coef
        )[0]
        row = {
            "subject": sequence.subject,
            "participant_id": sequence.participant_id,
        }
        for old_state, probability in enumerate(probabilities):
            row[f"initial_state_{state_label[old_state]}_probability"] = probability
        initial_probability_rows.append(row)
    pd.DataFrame(initial_probability_rows).to_csv(
        result_dir / "initial_state_probabilities.csv", index=False
    )

    all_emission_x = np.vstack([sequence.emission_x for sequence in sequences])
    state_acceptance = expit(
        add_intercept(all_emission_x) @ model.emission_coef.T
    ).mean(axis=0)
    posterior_mass = np.zeros(model.k)
    for sequence in sequences:
        posterior_mass += forward_backward(model, sequence)[1].sum(axis=0)
    state_summary = pd.DataFrame(
        [
            {
                "state": state_label[old_state],
                "mean_predicted_acceptance": state_acceptance[old_state],
                "posterior_expected_trials": posterior_mass[old_state],
                "posterior_occupancy_rate": posterior_mass[old_state]
                / posterior_mass.sum(),
            }
            for old_state in range(model.k)
        ]
    ).sort_values("state")
    state_summary.to_csv(result_dir / "state_summary.csv", index=False)

    np.savez_compressed(
        result_dir / "model_parameters.npz",
        initial_coef=model.initial_coef,
        transition_coef=model.transition_coef,
        emission_coef=model.emission_coef,
        state_order=order,
    )

    ai_users = users[
        users["AI_version"].map(clean_text).str.lower().eq(ai_version)
    ]
    user_subjects = set(ai_users["subject"])
    trial_subjects = {sequence.subject for sequence in sequences}
    user_participant_ids = {
        row["subject"]: clean_text(row["participant_id"])
        for _, row in users.iterrows()
    }
    participant_id_mismatches = [
        {
            "subject": sequence.subject,
            "trial_participant_id": sequence.participant_id,
            "user_participant_id": user_participant_ids[sequence.subject],
        }
        for sequence in sequences
        if sequence.subject in user_participant_ids
        and sequence.participant_id != user_participant_ids[sequence.subject]
    ]
    ai_version_counts = {ai_version: len(sequences)}
    score_rows_checked = int(
        sum(int(item.get("score_rows_checked", 0) or 0) for item in audit)
    )
    score_mismatch_rows = int(
        sum(int(item.get("score_mismatch_rows", 0) or 0) for item in audit)
    )
    timeout_rows = int(
        sum(
            int(item.get("timeout_rows_treated_as_accept", 0) or 0)
            for item in audit
        )
    )
    scene_ids_corrected = int(
        sum(int(item.get("n_scene_ids_corrected", 0) or 0) for item in audit)
    )
    scene_id_correction_details = [
        {
            "file": clean_text(item.get("file")),
            "corrections": clean_text(item.get("scene_id_corrections")),
        }
        for item in audit
        if int(item.get("n_scene_ids_corrected", 0) or 0) > 0
    ]
    metadata.update(
        {
            "scenario_statistics_file": str(SCENARIO_FILE),
            "ai_output_columns": AI_OUTPUT_COLUMNS,
            "trained_ai_version": ai_version,
            "selected_K": model.k,
            "log_likelihood": model.log_likelihood,
            "n_parameters": parameter_count(model),
            "n_subjects": len(sequences),
            "n_trials": sum(len(sequence.behavior) for sequence in sequences),
            "n_trials_per_subject": N_INTERACTIONS_PER_SUBJECT,
            "ai_version_subject_counts": ai_version_counts,
            "timeout_rows_treated_as_accept": timeout_rows,
            "scene_ids_corrected": scene_ids_corrected,
            "scene_id_correction_details": scene_id_correction_details,
            "score_rows_checked": score_rows_checked,
            "score_mismatch_rows": score_mismatch_rows,
            "state_definition": "Renumbered by ascending mean predicted acceptance probability",
            "trial_subjects_missing_user_data": sorted(trial_subjects - user_subjects),
            "user_subjects_missing_trial_data": sorted(user_subjects - trial_subjects),
            "participant_id_mismatches": participant_id_mismatches,
            "prediction_metrics": prediction_metrics,
            "safety_probability_definition": (
                "Predicted probability of accepting correct AI advice or rejecting incorrect AI advice"
            ),
        }
    )
    (result_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def cleanup_ai_root_outputs(result_dir: Path) -> None:
    """Remove legacy root-level model files while retaining candidate directories and summaries."""
    result_dir.mkdir(parents=True, exist_ok=True)
    root_files = [
        "bic_comparison.csv",
        "data_audit.csv",
        "emission_coefficients.csv",
        "initial_coefficients.csv",
        "initial_state_probabilities.csv",
        "model_metadata.json",
        "model_parameters.npz",
        "one_step_predictions.csv",
        "prediction_metrics.csv",
        "prediction_performance.csv",
        "reliance_probability.csv",
        "reliance_probability_real_distribution.csv",
        "selective_reliance_probability.csv",
        "selective_reliance_probability_real_distribution.csv",
        "results.csv",
        "state_paths.csv",
        "state_summary.csv",
        "transition_coefficients.csv",
    ]
    for filename in root_files:
        path = result_dir / filename
        if path.exists():
            path.unlink()

    figures_dir = result_dir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)


def plot_bic_comparison(result_dir: Path, ai_version: str) -> None:
    comparison = pd.read_csv(result_dir / "bic_comparison.csv").sort_values("K")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(comparison["K"], comparison["BIC"], marker="o", linewidth=2)
    best = comparison.loc[comparison["BIC"].idxmin()]
    for _, row in comparison.iterrows():
        convergence = "Converged" if bool(row["converged"]) else "Not converged"
        ax.annotate(
            f"{row['BIC']:.1f}\n{convergence}",
            (row["K"], row["BIC"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
    ax.scatter(best["K"], best["BIC"], s=100, color="#c44e52", zorder=3)
    ax.set_xticks(comparison["K"].astype(int))
    ax.set_xlabel("Number of Latent States (K)")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title(f"{ai_version}: Model Comparison")
    ax.grid(alpha=0.25)
    save_figure(fig, result_dir / "figures" / "01_bic_comparison.png")


def plot_state_profiles(result_dir: Path, ai_version: str) -> None:
    paths = pd.read_csv(result_dir / "state_paths.csv")
    rows = []
    for state, group in paths.groupby("viterbi_state", sort=True):
        correct = group[group["ai_correct"].eq(1)]
        wrong = group[group["ai_correct"].eq(0)]
        rows.append(
            {
                "state": int(state),
                "Viterbi occupancy": len(group) / len(paths),
                "Overall acceptance": group["behavior_accept"].mean(),
                "Acceptance rate (LLM correct)": correct[
                    "behavior_accept"
                ].mean(),
                "Acceptance rate (LLM wrong)": wrong[
                    "behavior_accept"
                ].mean(),
            }
        )
    profile = pd.DataFrame(rows).set_index("state")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    profile.plot(kind="bar", ax=ax, width=0.78)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("State")
    ax.set_ylabel("Proportion")
    ax.set_title(f"{ai_version}: Latent-State Behavior Profiles")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(
        fig,
        result_dir / "figures" / "02_state_profiles_Viterbi.png",
    )


def plot_transition_matrix(result_dir: Path, ai_version: str) -> None:
    paths = pd.read_csv(result_dir / "state_paths.csv").sort_values(
        ["subject", "trial_index"]
    )
    paths["next_state"] = paths.groupby("subject")["viterbi_state"].shift(-1)
    transitions = paths.dropna(subset=["next_state"]).copy()
    transitions["next_state"] = transitions["next_state"].astype(int)
    matrix = pd.crosstab(
        transitions["viterbi_state"],
        transitions["next_state"],
        normalize="index",
    )
    states = sorted(paths["viterbi_state"].unique())
    matrix = matrix.reindex(index=states, columns=states, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    for i in range(len(states)):
        for j in range(len(states)):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=MATRIX_CELL_FONT_SIZE,
            )
    ax.set_xticks(range(len(states)), [f"State {s}" for s in states])
    ax.set_yticks(range(len(states)), [f"State {s}" for s in states])
    ax.set_xlabel("Current state")
    ax.set_ylabel("Previous state")
    # ax.set_title(f"{ai_version}: Empirical State Transitions")
    fig.colorbar(image, ax=ax, label="Transition probability")
    save_figure(fig, result_dir / "figures" / "03_transition_matrix.png")


def plot_state_paths(result_dir: Path, ai_version: str) -> None:
    paths = pd.read_csv(result_dir / "state_paths.csv")
    state_table = paths.pivot(
        index="subject", columns="trial_index", values="viterbi_state"
    )
    state_table = state_table.sort_index()
    n_states = int(paths["viterbi_state"].max())

    fig_height = max(4.5, 0.30 * len(state_table) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    image = ax.imshow(
        state_table.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=plt.get_cmap("viridis", n_states),
        vmin=0.5,
        vmax=n_states + 0.5,
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Participant")
    ax.set_title(f"{ai_version}: Participant Latent-State Paths")
    ax.set_yticks(range(len(state_table)), state_table.index, fontsize=7)
    tick_positions = np.arange(0, state_table.shape[1], 10)
    ax.set_xticks(tick_positions, state_table.columns[tick_positions])
    colorbar = fig.colorbar(image, ax=ax, ticks=range(1, n_states + 1))
    colorbar.set_label("Viterbi State")
    save_figure(fig, result_dir / "figures" / "04_subject_state_paths.png")


def plot_prediction_calibration(result_dir: Path, ai_version: str) -> None:
    predictions = pd.read_csv(result_dir / "one_step_predictions.csv")
    unique_probabilities = predictions["predicted_accept_probability"].nunique()
    n_bins = min(10, unique_probabilities)
    predictions["probability_bin"] = pd.qcut(
        predictions["predicted_accept_probability"],
        q=n_bins,
        duplicates="drop",
    )
    calibration = predictions.groupby(
        "probability_bin", observed=True
    ).agg(
        mean_predicted=("predicted_accept_probability", "mean"),
        observed_acceptance=("observed_accept", "mean"),
        n=("observed_accept", "size"),
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.plot(
        calibration["mean_predicted"],
        calibration["observed_acceptance"],
        marker="o",
        linewidth=2,
        label="Model",
    )
    for _, row in calibration.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["mean_predicted"], row["observed_acceptance"]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted acceptance probability")
    ax.set_ylabel("Observed acceptance rate")
    ax.set_title(f"{ai_version}: Acceptance-probability calibration")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, result_dir / "figures" / "05_prediction_calibration.png")


def plot_state_proportions(result_dir: Path, ai_version: str) -> None:
    paths = pd.read_csv(result_dir / "state_paths.csv")
    states = sorted(paths["viterbi_state"].unique())
    state_rate = (
        paths.groupby("trial_index")
        .agg(
            **{
                f"state_{state}_proportion": (
                    "viterbi_state",
                    lambda values, target=state: values.eq(target).mean(),
                )
                for state in states
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = RELIANCE_STATE_COLORS
    for index, state in enumerate(states):
        smoothed_rate = pad_and_smooth(
            state_rate[f"state_{state}_proportion"].to_numpy(dtype=float),
            window_radius=FILTER_WINDOW_RADIUS,
        )
        ax.plot(
            state_rate["trial_index"],
            smoothed_rate,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=colors[index % len(colors)],
            label=f"State {state}",
        )
    ax.set_xlim(
        state_rate["trial_index"].min(), state_rate["trial_index"].max()
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Interaction Index")
    ax.set_ylabel("Participant Proportion")
    ax.set_title(
        f"{ai_version}: State Prevalence Across Trials "
        f"(Smoothing Radius={FILTER_WINDOW_RADIUS})"
    )
    ax.legend(frameon=False, ncol=min(len(states), 4))
    ax.grid(alpha=0.25)
    save_figure(fig, result_dir / "figures" / "06_state_proportion.png")


def plot_reliance_fraction(result_dir: Path, ai_version: str) -> None:
    """Compare observed and predicted acceptance proportions by round."""
    predictions = pd.read_csv(result_dir / "one_step_predictions.csv")
    reliance_rate = (
        predictions.groupby("trial_index")
        .agg(
            observed_reliance_fraction=("observed_accept", "mean"),
            predicted_reliance_fraction=(
                "predicted_accept_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(
        reliance_rate["trial_index"],
        reliance_rate["observed_reliance_fraction"],
        color="tab:blue",
        linewidth=2.0,
        label="Observed reliance",
    )
    ax.plot(
        reliance_rate["trial_index"],
        reliance_rate["predicted_reliance_fraction"],
        color="tab:red",
        linewidth=2.0,
        label="Model-predicted reliance",
    )
    ax.set_xlim(
        reliance_rate["trial_index"].min(),
        reliance_rate["trial_index"].max(),
    )
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.2))
    ax.set_xlabel("Collaboration round")
    ax.set_ylabel("Accepting LLM advice proportion")
    ax.set_title(f"{ai_version}: Observed and predicted LLM reliance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, result_dir / "figures" / "07_reliance_fraction.png")


def plot_state_conditioned_reliance_probability(
    result_dir: Path,
    ai_version: str,
    output_path: Path | None = None,
) -> None:
    """Plot participant mean-reliance distributions with each latent state fixed."""
    reliance = pd.read_csv(result_dir / "reliance_probability.csv")
    state_summary = (
        reliance.groupby(["state", "reliance_level"], as_index=False)
        .agg(
            state_mean=("mean_estimated_reliance_probability", "mean"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("state_mean")
    )

    colors = RELIANCE_STATE_COLORS
    bins = np.arange(0.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=SINGLE_DISTRIBUTION_FIGSIZE)

    for index, row in state_summary.reset_index(drop=True).iterrows():
        state = int(row["state"])
        values = reliance.loc[
            reliance["state"].eq(state),
            "mean_estimated_reliance_probability",
        ].to_numpy(dtype=float)
        color = colors[index % len(colors)]
        legend_label = f"{row['reliance_level']} (mean={row['state_mean']:.3f})"
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 1.0 / len(values)),
            color=color,
            alpha=0.28,
            edgecolor=color,
            linewidth=0.8,
            label=legend_label,
        )
        if len(values) > 1 and np.ptp(values) > 1e-10:
            try:
                density = gaussian_kde(values)(probability_grid) * bin_width
                ax.plot(
                    probability_grid,
                    density,
                    color=color,
                    linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                )
            except np.linalg.LinAlgError:
                pass

    ax.set_xlim(0, 1)
    ax.set_xlabel("Reliance (model estimated)")
    ax.set_ylabel("Relative frequency")
    ax.set_title(
        f"{ai_version}: Estimated Reliance by Prior Knowledge State"
    )
    ax.legend(frameon=False, ncol=min(len(state_summary), 4))
    ax.grid(alpha=0.22)
    save_figure(
        fig,
        output_path or result_dir / "figures" / "reliance_probability.png",
    )


def plot_state_real_reliance_probability(
    result_dir: Path,
    ai_version: str,
    output_path: Path | None = None,
) -> None:
    """Plot participant observed-acceptance distributions within Viterbi states."""
    reliance = pd.read_csv(
        result_dir / "reliance_probability_real_distribution.csv"
    )
    value_column = "real_reliance_probability"
    state_summary = (
        reliance.groupby(["state", "reliance_level"], as_index=False)
        .agg(
            state_mean=(value_column, "mean"),
            n_subjects=(value_column, "count"),
        )
        .sort_values("state")
    )

    colors = RELIANCE_STATE_COLORS
    bins = np.arange(0.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=SINGLE_DISTRIBUTION_FIGSIZE)

    for index, row in state_summary.reset_index(drop=True).iterrows():
        state = int(row["state"])
        values = reliance.loc[
            reliance["state"].eq(state),
            value_column,
        ].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        color = colors[index % len(colors)]
        legend_label = f"{row['reliance_level']} (mean={row['state_mean']:.3f})"
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 1.0 / len(values)),
            color=color,
            alpha=0.28,
            edgecolor=color,
            linewidth=0.8,
            label=legend_label,
        )
        if len(values) > 1 and np.ptp(values) > 1e-10:
            try:
                density = gaussian_kde(values)(probability_grid) * bin_width
                ax.plot(
                    probability_grid,
                    density,
                    color=color,
                    linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                )
            except np.linalg.LinAlgError:
                pass

    ax.set_xlim(0, 1)
    ax.set_xlabel("Reliance (real data)")
    ax.set_ylabel("Relative frequency")
    ax.set_title(f"{ai_version}: Observed Reliance by Viterbi State")
    ax.legend(frameon=False, ncol=min(len(state_summary), 4))
    ax.grid(alpha=0.22)
    save_figure(
        fig,
        output_path
        or result_dir / "figures" / "reliance_probability_real_distribution.png",
    )


def plot_state_selective_reliance_probability(
    result_dir: Path,
    ai_version: str,
    output_path: Path | None = None,
) -> None:
    """Plot participant selective-reliance distributions with each latent state fixed."""
    selective = pd.read_csv(
        result_dir / "selective_reliance_probability.csv"
    )
    state_summary = (
        selective.groupby(["state", "reliance_level"], as_index=False)
        .agg(
            state_mean=("selective_reliance_probability", "mean"),
            n_subjects=("selective_reliance_probability", "count"),
        )
        .sort_values("state")
    )

    colors = RELIANCE_STATE_COLORS
    bins = np.arange(-1.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(-1.0, 1.0, 800)
    fig, ax = plt.subplots(figsize=SINGLE_DISTRIBUTION_FIGSIZE)

    for index, row in state_summary.reset_index(drop=True).iterrows():
        state = int(row["state"])
        values = selective.loc[
            selective["state"].eq(state),
            "selective_reliance_probability",
        ].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        color = colors[index % len(colors)]
        legend_label = f"{row['reliance_level']} (mean={row['state_mean']:.3f})"
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 1.0 / len(values)),
            color=color,
            alpha=0.28,
            edgecolor=color,
            linewidth=0.8,
            label=legend_label,
        )
        if len(values) > 1 and np.ptp(values) > 1e-10:
            try:
                density = gaussian_kde(values)(probability_grid) * bin_width
                ax.plot(
                    probability_grid,
                    density,
                    color=color,
                    linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                )
            except np.linalg.LinAlgError:
                pass

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Selective reliance (model estimated)")
    ax.set_ylabel("Relative frequency")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title(
        f"{ai_version}: Selective Reliance by Prior Knowledge State"
    )
    ax.legend(frameon=False, ncol=min(len(state_summary), 4))
    ax.grid(alpha=0.22)
    save_figure(
        fig,
        output_path
        or result_dir / "figures" / "selective_reliance_probability.png",
    )


def plot_state_real_selective_reliance_probability(
    result_dir: Path,
    ai_version: str,
    output_path: Path | None = None,
) -> None:
    """Plot observed selective-reliance distributions within Viterbi states."""
    selective = pd.read_csv(
        result_dir / "selective_reliance_probability_real_distribution.csv"
    )
    value_column = "real_selective_reliance_probability"
    state_summary = (
        selective.groupby(["state", "reliance_level"], as_index=False)
        .agg(
            state_mean=(value_column, "mean"),
            n_subjects=(value_column, "count"),
        )
        .sort_values("state")
    )

    colors = RELIANCE_STATE_COLORS
    bins = np.arange(-1.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(-1.0, 1.0, 800)
    fig, ax = plt.subplots(figsize=SINGLE_DISTRIBUTION_FIGSIZE)

    for index, row in state_summary.reset_index(drop=True).iterrows():
        state = int(row["state"])
        values = selective.loc[
            selective["state"].eq(state),
            value_column,
        ].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        color = colors[index % len(colors)]
        legend_label = f"{row['reliance_level']} (mean={row['state_mean']:.3f})"
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 1.0 / len(values)),
            color=color,
            alpha=0.28,
            edgecolor=color,
            linewidth=0.8,
            label=legend_label,
        )
        if len(values) > 1 and np.ptp(values) > 1e-10:
            try:
                density = gaussian_kde(values)(probability_grid) * bin_width
                ax.plot(
                    probability_grid,
                    density,
                    color=color,
                    linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                )
            except np.linalg.LinAlgError:
                pass

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Selective reliance (real data)")
    ax.set_ylabel("Relative frequency")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_title(
        f"{ai_version}: Observed Selective Reliance by Viterbi State"
    )
    ax.legend(frameon=False, ncol=min(len(state_summary), 4))
    ax.grid(alpha=0.22)
    save_figure(
        fig,
        output_path
        or result_dir
        / "figures"
        / "selective_reliance_probability_real_distribution.png",
    )


def plot_feedback_bootstrap_panel(
    axis: plt.Axes,
    predictions: pd.DataFrame,
    feedback_code: int,
    bins: np.ndarray,
) -> dict[str, object] | None:
    """Plot the bootstrap distribution for one feedback category and return comparison metrics."""
    subset = predictions.loc[
        predictions["previous_feedback"].eq(feedback_code)
    ].copy()
    observed_bootstrap = bootstrap_mean_distribution(
        subset["observed_accept"].to_numpy(dtype=float),
        seed=RANDOM_SEED + feedback_code,
    )
    predicted_bootstrap = bootstrap_mean_distribution(
        subset["predicted_accept_probability"].to_numpy(dtype=float),
        seed=RANDOM_SEED + 100 + feedback_code,
    )

    metrics: dict[str, object] | None = None
    if len(subset) > 0:
        metrics = {
            "feedback": FEEDBACK_LABELS[feedback_code],
            "n_samples": len(subset),
            **bootstrap_distribution_metrics(
                observed_bootstrap,
                predicted_bootstrap,
                bins,
            ),
        }
        axis.hist(
            observed_bootstrap,
            bins=bins,
            color="tab:blue",
            alpha=0.55,
            weights=np.full(
                len(observed_bootstrap),
                1.0 / len(observed_bootstrap),
            ),
            label="Observed",
        )
        axis.hist(
            predicted_bootstrap,
            bins=bins,
            color="tab:red",
            alpha=0.45,
            weights=np.full(
                len(predicted_bootstrap),
                1.0 / len(predicted_bootstrap),
            ),
            label="Model-predicted",
        )
        for values, color in (
            (observed_bootstrap, "tab:blue"),
            (predicted_bootstrap, "tab:red"),
        ):
            axis.axvline(
                np.mean(values),
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )
    else:
        axis.text(
            0.5,
            0.5,
            "No samples",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.text(
        0.03,
        0.95,
        (
            f"{FEEDBACK_LABELS[feedback_code].replace(' ', chr(10), 1)}"
            f"\nn={len(subset)}"
        ),
        ha="left",
        va="top",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    axis.set_xlim(0.60, 1.00)
    x_ticks = np.arange(0.60, 1.001, 0.05)
    axis.set_xticks(x_ticks)
    axis.set_xticklabels([f"{value:.2f}" for value in x_ticks])
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.20)
    axis.set_axisbelow(True)
    return metrics


def plot_reliance_probability_distribution(
    result_dir: Path,
    ai_version: str,
    output_path: Path | None = None,
) -> None:
    """Plot bootstrap distributions of next-round AI reliance by previous feedback."""
    predictions = pd.read_csv(result_dir / "one_step_predictions.csv")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.2),
        sharex=True,
        sharey=True,
    )
    histogram_bin_width = 0.005
    bins = np.arange(
        0.0,
        1.0 + histogram_bin_width / 2,
        histogram_bin_width,
    )
    similarity_rows: list[dict[str, object]] = []
    for axis, feedback_code in zip(axes.flat, sorted(FEEDBACK_LABELS)):
        metrics = plot_feedback_bootstrap_panel(
            axis,
            predictions,
            feedback_code,
            bins,
        )
        if metrics is not None:
            similarity_rows.append(metrics)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
        )
    fig.supxlabel("Reliance probability", y=0.025)
    fig.supylabel("Proportion", x=0.012)
    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        bottom=0.12,
        top=0.90,
        hspace=0.16,
        wspace=0.10,
    )
    save_paper_combined_figure(
        fig,
        output_path
        or result_dir
        / "figures"
        / "08_distribution_reliance_probability.png",
    )
    if output_path is None:
        write_bootstrap_distribution_report(
            result_dir
            / "figures"
            / "08_distribution_reliance_probability.txt",
            ai_version,
            similarity_rows,
        )


def generate_result_plots(result_dir: Path, ai_version: str) -> None:
    """Generate core model diagnostic and interpretation figures from exported CSV files."""
    setup_plot_style()
    plot_bic_comparison(result_dir, ai_version)
    plot_state_profiles(result_dir, ai_version)
    plot_transition_matrix(result_dir, ai_version)
    plot_state_paths(result_dir, ai_version)
    plot_prediction_calibration(result_dir, ai_version)
    plot_state_proportions(result_dir, ai_version)
    plot_reliance_fraction(result_dir, ai_version)
    plot_state_conditioned_reliance_probability(result_dir, ai_version)
    plot_state_real_reliance_probability(result_dir, ai_version)
    plot_state_selective_reliance_probability(result_dir, ai_version)
    plot_state_real_selective_reliance_probability(result_dir, ai_version)
    plot_reliance_probability_distribution(result_dir, ai_version)
    progress_write(f"[{ai_version}] Figures saved to: {result_dir / 'figures'}")


def save_paper_combined_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    exact_size: bool = False,
) -> None:
    """Save a manually arranged publication figure without automatic boundary cropping."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options: dict[str, object] = {
        "dpi": 300,
        "facecolor": "white",
    }
    if not exact_size:
        save_options.update({"bbox_inches": "tight", "pad_inches": 0.04})
    fig.savefig(output_path, **save_options)
    plt.close(fig)


def paper_candidate_dir(ai_version: str, k: int) -> Path:
    return RESULTS_DIR / ai_version / f"K{k}"


def plot_paper_bic_comparisons(k: int, output_path: Path) -> None:
    """Redraw the BIC comparison across candidate state counts and AI conditions."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_TALL_COMBINED_FIGSIZE,
        sharex=True,
        sharey=False,
    )
    legend_handles = None
    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        comparison = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "bic_comparison.csv"
        ).sort_values("K")
        line = axis.plot(
            comparison["K"],
            comparison["BIC"],
            color=RELIANCE_STATE_COLORS[2],
            marker="o",
            linewidth=1.8,
            label="Candidate models",
        )[0]
        best = comparison.loc[comparison["BIC"].idxmin()]
        best_marker = axis.scatter(
            best["K"],
            best["BIC"],
            s=70,
            color="tab:red",
            zorder=3,
            label="Minimum BIC",
        )
        k_min = int(comparison["K"].min())
        k_max = int(comparison["K"].max())
        for _, row in comparison.iterrows():
            current_k = int(row["K"])
            horizontal_alignment = (
                "left" if current_k == k_min else "right" if current_k == k_max else "center"
            )
            axis.annotate(
                f"{row['BIC']:.0f}",
                (row["K"], row["BIC"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha=horizontal_alignment,
            )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xticks(comparison["K"].astype(int))
        axis.margins(x=0.12, y=0.18)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        if panel_index == 0:
            legend_handles = ([line, best_marker], ["Candidate models", "Minimum BIC"])

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
        )
    fig.supxlabel("Number of states (K)", y=0.025)
    fig.supylabel("BIC", x=0.012)
    fig.subplots_adjust(
        left=0.12,
        right=0.975,
        bottom=0.16,
        top=0.78,
        wspace=0.34,
    )
    save_paper_combined_figure(fig, output_path, exact_size=True)


def load_calibration_for_paper(result_dir: Path) -> tuple[pd.DataFrame, int]:
    """Summarize calibration curves by predicted-probability quantile."""
    predictions = pd.read_csv(result_dir / "one_step_predictions.csv")
    n_bins = min(10, predictions["predicted_accept_probability"].nunique())
    predictions["probability_bin"] = pd.qcut(
        predictions["predicted_accept_probability"],
        q=n_bins,
        duplicates="drop",
    )
    calibration = (
        predictions.groupby("probability_bin", observed=True)
        .agg(
            mean_predicted=("predicted_accept_probability", "mean"),
            observed_acceptance=("observed_accept", "mean"),
            n=("observed_accept", "size"),
        )
        .reset_index(drop=True)
    )
    return calibration, len(predictions)


def plot_paper_prediction_calibrations(k: int, output_path: Path) -> None:
    """Redraw acceptance-probability calibration curves for all AI conditions."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_TALL_COMBINED_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    legend_handles = None
    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        calibration, n_predictions = load_calibration_for_paper(
            paper_candidate_dir(ai_version, k)
        )
        ideal = axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            linewidth=1.4,
            label="Perfect calibration",
        )[0]
        model_line = axis.plot(
            calibration["mean_predicted"],
            calibration["observed_acceptance"],
            color=RELIANCE_STATE_COLORS[2],
            marker="o",
            linewidth=1.8,
            label="Model",
        )[0]
        axis.text(
            0.04,
            0.95,
            f"n={n_predictions}",
            ha="left",
            va="top",
            transform=axis.transAxes,
        )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        x_ticks = np.arange(0, 1.01, 0.25)
        x_tick_labels = [f"{value:.2f}" for value in x_ticks]
        if panel_index > 0:
            x_tick_labels[0] = ""
        if panel_index < len(AI_VERSION_LEVELS) - 1:
            x_tick_labels[-1] = ""
        axis.set_xticks(x_ticks, x_tick_labels)
        axis.set_yticks(np.arange(0, 1.01, 0.25))
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(labelleft=panel_index == 0)
        if panel_index == 0:
            legend_handles = (
                [ideal, model_line],
                ["Perfect calibration", "Model"],
            )

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
        )
    fig.supxlabel("Mean predicted acceptance probability", y=0.025)
    fig.supylabel("Observed acceptance rate", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.975,
        bottom=0.16,
        top=0.80,
        wspace=0.16,
    )
    save_paper_combined_figure(fig, output_path, exact_size=True)


def plot_paper_feedback_bootstrap_distributions(
    k: int,
    output_path: Path,
) -> None:
    """Plot bootstrap distributions for four feedback categories across three AI conditions."""
    feedback_codes = sorted(FEEDBACK_LABELS)
    fig, axes = plt.subplots(
        len(feedback_codes),
        len(AI_VERSION_LEVELS),
        figsize=PAPER_BOOTSTRAP_COMBINED_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    histogram_bin_width = 0.005
    bins = np.arange(
        0.0,
        1.0 + histogram_bin_width / 2,
        histogram_bin_width,
    )

    for column_index, ai_version in enumerate(AI_VERSION_LEVELS):
        predictions = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "one_step_predictions.csv"
        )
        for row_index, feedback_code in enumerate(feedback_codes):
            axis = axes[row_index, column_index]
            plot_feedback_bootstrap_panel(
                axis,
                predictions,
                feedback_code,
                bins,
            )
            axis.set_ylim(0.0, 0.6)
            axis.set_yticks(np.arange(0.0, 0.601, 0.15))
            axis.tick_params(
                labelleft=column_index == 0,
                labelbottom=row_index == len(feedback_codes) - 1,
            )
            if row_index == 0:
                axis.set_title(PAPER_AI_LABELS[ai_version])

    for column_index, axis in enumerate(axes[-1]):
        tick_labels = axis.get_xticklabels()
        if column_index > 0 and tick_labels:
            tick_labels[0].set_visible(False)
        if column_index < len(AI_VERSION_LEVELS) - 1 and tick_labels:
            tick_labels[-1].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
        )
    fig.supxlabel("Reliance probability", y=0.025)
    fig.supylabel("Proportion", x=0.012)
    fig.subplots_adjust(
        left=0.075,
        right=0.980,
        bottom=0.10,
        top=0.93,
        hspace=0.10,
        wspace=0.08,
    )
    save_paper_combined_figure(fig, output_path, exact_size=True)


def load_state_profile_for_paper(result_dir: Path) -> pd.DataFrame:
    """Rebuild the state-behavior profile for Figure 02 from numeric results rather than PNG files."""
    paths = pd.read_csv(result_dir / "state_paths.csv")
    rows = []
    for state, group in paths.groupby("viterbi_state", sort=True):
        correct = group[group["ai_correct"].eq(1)]
        wrong = group[group["ai_correct"].eq(0)]
        rows.append(
            {
                "state": int(state),
                "Viterbi occupancy": len(group) / len(paths),
                "Overall acceptance": group["behavior_accept"].mean(),
                "Acceptance rate (LLM correct)": correct[
                    "behavior_accept"
                ].mean(),
                "Acceptance rate (LLM wrong)": wrong[
                    "behavior_accept"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("state")


def plot_paper_state_profiles(k: int, output_path: Path) -> None:
    """Redraw state profiles for all AI conditions with shared axis labels."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_TALL_COMBINED_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    columns = [
        "Viterbi occupancy",
        "Overall acceptance",
        "Acceptance rate (LLM correct)",
        "Acceptance rate (LLM wrong)",
    ]
    colors = PAPER_STATE_PROFILE_COLORS
    bar_width = 0.19
    legend_handles = None

    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        profile = load_state_profile_for_paper(
            paper_candidate_dir(ai_version, k)
        )
        x = np.arange(len(profile.index), dtype=float)
        for column_index, (column, color) in enumerate(zip(columns, colors)):
            offset = (column_index - (len(columns) - 1) / 2) * bar_width
            axis.bar(
                x + offset,
                profile[column].to_numpy(dtype=float),
                width=bar_width,
                color=color,
                label=column,
            )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xticks(x, [str(int(state)) for state in profile.index])
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(labelleft=panel_index == 0)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
        )
    fig.supxlabel("State", y=0.025)
    fig.supylabel("Proportion", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.16,
        top=0.73,
        wspace=0.08,
    )
    save_paper_combined_figure(fig, output_path, exact_size=True)


def wilson_confidence_interval(
    successes: int,
    total: int,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    """Calculate a Wilson 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (
        proportion + z_value**2 / (2.0 * total)
    ) / denominator
    radius = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    return center - radius, center + radius


def load_state_profile_for_visual_mockup(
    result_dir: Path,
    k: int,
) -> pd.DataFrame:
    """Summarize occupancy and acceptance by Viterbi state and AI correctness."""
    paths = pd.read_csv(result_dir / "state_paths.csv")
    rows: list[dict[str, float | int]] = []
    for state in range(1, k + 1):
        group = paths[paths["viterbi_state"].eq(state)]
        row: dict[str, float | int] = {
            "state": state,
            "occupancy": len(group) / len(paths),
        }
        for condition_name, ai_correct in (("correct", 1), ("wrong", 0)):
            condition = group[group["ai_correct"].eq(ai_correct)]
            total = len(condition)
            successes = int(condition["behavior_accept"].sum())
            lower, upper = wilson_confidence_interval(successes, total)
            row[f"{condition_name}_n"] = total
            row[f"{condition_name}_rate"] = (
                successes / total if total else np.nan
            )
            row[f"{condition_name}_lower"] = lower
            row[f"{condition_name}_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def plot_paper_state_profiles_visual_mockup(
    k: int,
    output_path: Path,
) -> None:
    """Redraw state occupancy and conditional acceptance without overwriting Figure 02."""
    fig = plt.figure(figsize=PAPER_VISUAL_MOCKUP_FIGSIZE)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(0.18, 1.0),
        left=0.095,
        right=0.985,
        bottom=0.18,
        top=0.80,
        hspace=0.35,
        wspace=0.18,
    )
    state_colors = RELIANCE_STATE_COLORS
    marker_color = "black"

    for panel_index, ai_version in enumerate(AI_VERSION_LEVELS):
        profile = load_state_profile_for_visual_mockup(
            paper_candidate_dir(ai_version, k),
            k,
        )
        occupancy_axis = fig.add_subplot(grid[0, panel_index])
        left = 0.0
        for state_index, state in enumerate(profile.index):
            occupancy = float(profile.loc[state, "occupancy"])
            color = state_colors[state_index % len(state_colors)]
            occupancy_axis.barh(
                0,
                occupancy,
                left=left,
                height=0.72,
                color=color,
                edgecolor="white",
                linewidth=1.0,
            )
            if occupancy >= 0.07:
                occupancy_axis.text(
                    left + occupancy / 2.0,
                    0,
                    f"{occupancy:.0%}",
                    ha="center",
                    va="center",
                    color="black" if state_index == 0 else "white",
                    fontsize=FONT_SIZE,
                )
            left += occupancy
        occupancy_axis.set_xlim(0.0, 1.0)
        occupancy_axis.set_ylim(-0.55, 0.55)
        occupancy_axis.set_title(PAPER_AI_LABELS[ai_version], pad=10)
        if panel_index == 0:
            occupancy_axis.text(
                -0.035,
                0.5,
                "Viterbi\noccupancy",
                transform=occupancy_axis.transAxes,
                ha="right",
                va="center",
            )
        occupancy_axis.set_axis_off()

        axis = fig.add_subplot(grid[1, panel_index])
        y_positions = np.arange(1, k + 1, dtype=float)
        for condition_name, offset, color, filled in (
            ("correct", -0.10, marker_color, True),
            ("wrong", 0.10, marker_color, False),
        ):
            estimates = profile[f"{condition_name}_rate"].to_numpy(dtype=float)
            lower = profile[f"{condition_name}_lower"].to_numpy(dtype=float)
            upper = profile[f"{condition_name}_upper"].to_numpy(dtype=float)
            valid = np.isfinite(estimates) & np.isfinite(lower) & np.isfinite(upper)
            axis.errorbar(
                estimates[valid] * 100.0,
                y_positions[valid] + offset,
                xerr=np.vstack(
                    [
                        np.maximum(
                            0.0,
                            (estimates[valid] - lower[valid]) * 100.0,
                        ),
                        np.maximum(
                            0.0,
                            (upper[valid] - estimates[valid]) * 100.0,
                        ),
                    ]
                ),
                fmt="o",
                markersize=4.5,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                zorder=3,
            )
        axis.set_xlim(0.0, 100.0)
        axis.set_xticks([0, 50, 100])
        axis.set_ylim(k + 0.55, 0.45)
        axis.set_yticks(
            y_positions,
            [f"State {state}" for state in range(1, k + 1)],
        )
        axis.tick_params(axis="y", labelleft=panel_index == 0, length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_bounds(0.0, 100.0)
        axis.margins(x=0.0)
        axis.grid(False)

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                markersize=5,
                linestyle="none",
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                label="Acceptance rate (LLM correct)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markersize=5,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=marker_color,
                markeredgewidth=1.1,
                label="Acceptance rate (LLM wrong)",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.675),
        ncol=2,
        frameon=False,
    )
    fig.supxlabel("Acceptance probability (%)", y=0.045)
    save_paper_combined_figure(fig, output_path, exact_size=True)


def plot_paper_transition_matrices(k: int, output_path: Path) -> None:
    """Redraw three transition matrices with one shared color bar."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=PAPER_MATRIX_COMBINED_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    image = None
    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        paths = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "state_paths.csv"
        ).sort_values(["subject", "trial_index"])
        paths["next_state"] = paths.groupby("subject")["viterbi_state"].shift(-1)
        transitions = paths.dropna(subset=["next_state"]).copy()
        transitions["next_state"] = transitions["next_state"].astype(int)
        states = list(range(1, k + 1))
        matrix = pd.crosstab(
            transitions["viterbi_state"],
            transitions["next_state"],
            normalize="index",
        ).reindex(index=states, columns=states, fill_value=0.0)

        image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        for row_index in range(k):
            for column_index in range(k):
                value = matrix.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=MATRIX_CELL_FONT_SIZE,
                )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xticks(range(k), [str(state) for state in states])
        axis.set_yticks(range(k), [str(state) for state in states])
        axis.set_ylabel("Previous state", fontsize=FONT_SIZE)
        axis.tick_params(labelleft=True, labelbottom=True)

    fig.subplots_adjust(
        left=0.23,
        right=0.96,
        bottom=0.19,
        top=0.97,
        hspace=0.34,
    )
    bottom_axis_position = axes[-1].get_position()
    current_state_center = (
        bottom_axis_position.x0 + bottom_axis_position.x1
    ) / 2.0
    fig.supxlabel(
        "Current state",
        x=current_state_center,
        y=0.125,
        fontsize=FONT_SIZE,
    )
    colorbar_axis = fig.add_axes([0.18, 0.045, 0.73, 0.018]) # Position of the transition-probability color bar
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Transition probability")
    colorbar.ax.xaxis.set_label_position("top")
    save_paper_combined_figure(fig, output_path, exact_size=True)


def plot_paper_transition_matrices_visual_mockup(
    k: int,
    output_path: Path,
) -> None:
    """Arrange three transition matrices horizontally with a shared color scale."""
    fig = plt.figure(figsize=PAPER_VISUAL_MOCKUP_FIGSIZE)
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        left=0.11,
        right=0.93,
        bottom=0.18,
        top=0.82,
        wspace=0.28,
    )
    axes = [fig.add_subplot(grid[0, panel_index]) for panel_index in range(3)]
    colorbar_axis = fig.add_subplot(grid[0, 3])
    image = None
    states = list(range(1, k + 1))
    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        paths = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "state_paths.csv"
        ).sort_values(["subject", "trial_index"])
        paths["next_state"] = paths.groupby("subject")["viterbi_state"].shift(-1)
        transitions = paths.dropna(subset=["next_state"]).copy()
        transitions["next_state"] = transitions["next_state"].astype(int)
        matrix = pd.crosstab(
            transitions["viterbi_state"],
            transitions["next_state"],
            normalize="index",
        ).reindex(index=states, columns=states, fill_value=0.0)

        image = axis.imshow(
            matrix.to_numpy(),
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
            aspect="equal",
        )
        for row_index in range(k):
            for column_index in range(k):
                value = float(matrix.iloc[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=FONT_SIZE,
                )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xticks(range(k), [str(state) for state in states])
        axis.set_yticks(range(k), [str(state) for state in states])
        axis.tick_params(length=0, labelleft=panel_index == 0)
        axis.set_xticks(np.arange(-0.5, k, 1.0), minor=True)
        axis.set_yticks(np.arange(-0.5, k, 1.0), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    fig.supxlabel("To state", y=0.035)
    fig.supylabel("From state", x=0.025)
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.set_label("Transition probability")
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    save_paper_combined_figure(fig, output_path, exact_size=True)


def plot_paper_state_proportions(k: int, output_path: Path) -> None:
    """Redraw three state-occupancy time series with one shared y-axis label."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_TALL_COMBINED_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    states = list(range(1, k + 1))
    colors = RELIANCE_STATE_COLORS
    legend_handles = None

    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        paths = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "state_paths.csv"
        )
        state_rate = (
            paths.groupby("trial_index")
            .agg(
                **{
                    f"state_{state}_proportion": (
                        "viterbi_state",
                        lambda values, target=state: values.eq(target).mean(),
                    )
                    for state in states
                }
            )
            .reset_index()
        )
        for state_index, state in enumerate(states):
            smoothed_rate = pad_and_smooth(
                state_rate[f"state_{state}_proportion"].to_numpy(dtype=float),
                window_radius=FILTER_WINDOW_RADIUS,
            )
            axis.plot(
                state_rate["trial_index"],
                smoothed_rate,
                linewidth=1.8,
                color=colors[state_index % len(colors)],
                label=f"State {state}",
            )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(
            state_rate["trial_index"].min(),
            state_rate["trial_index"].max(),
        )
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.grid(alpha=0.25)
        axis.tick_params(labelleft=panel_index == 0)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=k,
            frameon=False,
        )
    fig.supxlabel("Collaboration round", y=0.025)
    fig.supylabel("Proportion", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.16,
        top=0.80,
        wspace=0.08,
    )
    save_paper_combined_figure(fig, output_path)


def plot_paper_state_proportions_visual_mockup(
    k: int,
    output_path: Path,
) -> None:
    """Aggregate round-level Viterbi occupancy into six interaction blocks."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_VISUAL_MOCKUP_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    states = list(range(1, k + 1))
    markers = ("o", "s", "^", "D")
    legend_handles = None

    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        paths = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "state_paths.csv"
        )
        state_rate = (
            paths.groupby("trial_index")
            .agg(
                **{
                    f"state_{state}_proportion": (
                        "viterbi_state",
                        lambda values, target=state: values.eq(target).mean(),
                    )
                    for state in states
                }
            )
            .reset_index()
            .sort_values("trial_index")
        )
        trial_indices = state_rate["trial_index"].drop_duplicates().tolist()
        block_map = {
            trial_index: min(
                6,
                int(position * 6 / len(trial_indices)) + 1,
            )
            for position, trial_index in enumerate(trial_indices)
        }
        state_rate["block"] = state_rate["trial_index"].map(block_map)
        block_rate = state_rate.groupby("block", as_index=False).mean(
            numeric_only=True
        )

        for state_index, state in enumerate(states):
            axis.plot(
                block_rate["block"],
                block_rate[f"state_{state}_proportion"] * 100.0,
                color=RELIANCE_STATE_COLORS[
                    state_index % len(RELIANCE_STATE_COLORS)
                ],
                linewidth=1.8,
                marker=markers[state_index % len(markers)],
                markersize=5.0,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"State {state}",
            )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(0.85, 6.15)
        axis.set_xticks(range(1, 7))
        axis.set_ylim(0.0, 100.0)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.tick_params(labelleft=panel_index == 0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=k,
            frameon=False,
        )
    fig.supxlabel("Collaboration block", y=0.035)
    fig.supylabel("State proportion (%)", x=0.025)
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.19,
        top=0.80,
        wspace=0.12,
    )
    save_paper_combined_figure(fig, output_path, exact_size=True)


def plot_paper_reliance_fractions(k: int, output_path: Path) -> None:
    """Redraw observed and predicted reliance curves with shared labels and legend."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_TALL_COMBINED_FIGSIZE,
        sharex=True,
        sharey=True,
    )
    legend_handles = None
    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        predictions = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "one_step_predictions.csv"
        )
        reliance_rate = (
            predictions.groupby("trial_index")
            .agg(
                observed_reliance_fraction=("observed_accept", "mean"),
                predicted_reliance_fraction=(
                    "predicted_accept_probability",
                    "mean",
                ),
            )
            .reset_index()
        )
        axis.plot(
            reliance_rate["trial_index"],
            reliance_rate["observed_reliance_fraction"],
            color="tab:blue",
            linewidth=1.8,
            label="Observed reliance",
        )
        axis.plot(
            reliance_rate["trial_index"],
            reliance_rate["predicted_reliance_fraction"],
            color="tab:red",
            linewidth=1.8,
            label="Model-predicted reliance",
        )
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(
            reliance_rate["trial_index"].min(),
            reliance_rate["trial_index"].max(),
        )
        axis.set_ylim(0.4, 1.0)
        axis.set_yticks(np.arange(0.4, 1.01, 0.2))
        axis.grid(alpha=0.25)
        axis.tick_params(labelleft=panel_index == 0)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
        )
    fig.supxlabel("Collaboration round", y=0.025)
    fig.supylabel("Accepting LLM advice proportion ", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.16,
        top=0.80,
        wspace=0.08,
    )
    save_paper_combined_figure(fig, output_path)


def plot_paper_reliance_probability_distributions(
    k: int,
    output_path: Path,
) -> None:
    """Plot state-reliance distributions for all AI conditions on one canvas."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_DISTRIBUTION_COMBINED_FIGSIZE,
        sharex=True,
        sharey=False,
    )
    colors = RELIANCE_STATE_COLORS
    bins = np.arange(0.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(0.0, 1.0, 400)
    legend_handles = None

    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        reliance = pd.read_csv(
            paper_candidate_dir(ai_version, k) / "reliance_probability.csv"
        )
        state_summary = (
            reliance.groupby(["state", "reliance_level"], as_index=False)
            .agg(
                state_mean=("mean_estimated_reliance_probability", "mean"),
                n_subjects=("subject", "nunique"),
            )
            .sort_values("state")
        )
        for state_index, row in state_summary.reset_index(drop=True).iterrows():
            state = int(row["state"])
            values = reliance.loc[
                reliance["state"].eq(state),
                "mean_estimated_reliance_probability",
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            color = colors[state_index % len(colors)]
            axis.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1.0 / len(values)),
                color=color,
                alpha=0.28,
                edgecolor=color,
                linewidth=0.8,
                label=str(row["reliance_level"]),
            )
            if len(values) > 1 and np.ptp(values) > 1e-10:
                try:
                    density = gaussian_kde(values)(probability_grid) * bin_width
                    axis.plot(
                        probability_grid,
                        density,
                        color=color,
                        linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                    )
                except np.linalg.LinAlgError:
                    pass

        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(0, 1)
        axis.set_xticks(np.arange(0.0, 1.01, 0.25))
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=k,
            frameon=False,
        )
    fig.supxlabel("Reliance (model estimated)", y=0.025)
    fig.supylabel("Relative frequency", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.16,
        top=0.80,
        wspace=0.30,
    )
    save_paper_combined_figure(fig, output_path)


def plot_paper_real_reliance_probability_distributions(
    k: int,
    output_path: Path,
) -> None:
    """Plot observed acceptance distributions by Viterbi state and AI condition."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_DISTRIBUTION_COMBINED_FIGSIZE,
        sharex=True,
        sharey=False,
    )
    colors = RELIANCE_STATE_COLORS
    bins = np.arange(0.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(0.0, 1.0, 400)
    value_column = "real_reliance_probability"
    legend_handles = None

    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        reliance = pd.read_csv(
            paper_candidate_dir(ai_version, k)
            / "reliance_probability_real_distribution.csv"
        )
        state_summary = (
            reliance.groupby(["state", "reliance_level"], as_index=False)
            .agg(
                state_mean=(value_column, "mean"),
                n_subjects=(value_column, "count"),
            )
            .sort_values("state")
        )
        for state_index, row in state_summary.reset_index(drop=True).iterrows():
            state = int(row["state"])
            values = reliance.loc[
                reliance["state"].eq(state),
                value_column,
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            color = colors[state_index % len(colors)]
            axis.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1.0 / len(values)),
                color=color,
                alpha=0.28,
                edgecolor=color,
                linewidth=0.8,
                label=str(row["reliance_level"]),
            )
            if len(values) > 1 and np.ptp(values) > 1e-10:
                try:
                    density = gaussian_kde(values)(probability_grid) * bin_width
                    axis.plot(
                        probability_grid,
                        density,
                        color=color,
                        linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                    )
                except np.linalg.LinAlgError:
                    pass

        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(0, 1)
        axis.set_xticks(np.arange(0.0, 1.01, 0.25))
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=k,
            frameon=False,
        )
    fig.supxlabel("Reliance (real data)", y=0.025)
    fig.supylabel("Relative frequency", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.16,
        top=0.80,
        wspace=0.30,
    )
    save_paper_combined_figure(fig, output_path)


def write_paper_selective_reliance_report(k: int, output_path: Path) -> None:
    """Save mean reliance and selective reliance omitted from the combined figure."""
    lines = [
        f"K={k} reliance probability summary",
        "",
        "Selective reliance = P(accept | LLM correct) "
        "- P(accept | LLM wrong)",
        "",
        (
            "condition\tstate\treliance_level\tn_subjects\t"
            "mean_reliance_probability\t"
            "mean_acceptance_probability_ai_correct\t"
            "mean_acceptance_probability_ai_wrong\t"
            "mean_selective_reliance_probability"
        ),
    ]
    for ai_version in AI_VERSION_LEVELS:
        result_dir = paper_candidate_dir(ai_version, k)
        reliance = pd.read_csv(result_dir / "reliance_probability.csv")
        selective = pd.read_csv(
            result_dir / "selective_reliance_probability.csv"
        )
        reliance_summary = (
            reliance.groupby(["state", "reliance_level"], as_index=False)
            .agg(
                n_subjects=("subject", "nunique"),
                mean_reliance_probability=(
                    "mean_estimated_reliance_probability",
                    "mean",
                ),
            )
        )
        selective_summary = (
            selective.groupby(["state", "reliance_level"], as_index=False)
            .agg(
                mean_acceptance_probability_ai_correct=(
                    "mean_acceptance_probability_ai_correct",
                    "mean",
                ),
                mean_acceptance_probability_ai_wrong=(
                    "mean_acceptance_probability_ai_wrong",
                    "mean",
                ),
                mean_selective_reliance_probability=(
                    "selective_reliance_probability",
                    "mean",
                ),
            )
        )
        summary = reliance_summary.merge(
            selective_summary,
            on=["state", "reliance_level"],
            how="outer",
        ).sort_values("state")
        for _, row in summary.iterrows():
            lines.append(
                "\t".join(
                    [
                        PAPER_AI_LABELS[ai_version],
                        str(int(row["state"])),
                        str(row["reliance_level"]),
                        str(int(row["n_subjects"])),
                        f"{row['mean_reliance_probability']:.6f}",
                        f"{row['mean_acceptance_probability_ai_correct']:.6f}",
                        f"{row['mean_acceptance_probability_ai_wrong']:.6f}",
                        f"{row['mean_selective_reliance_probability']:.6f}",
                    ]
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_paper_selective_reliance_probability_distributions(
    k: int,
    output_path: Path,
    report_path: Path,
) -> None:
    """Plot selective-reliance distributions for all AI conditions and save a mean report."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_SELECTIVE_RELIANCE_FIGSIZE,
        sharex=True,
        sharey=False,
    )
    colors = RELIANCE_STATE_COLORS
    bins = np.arange(-1.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(-1.0, 1.0, 800)
    x_ticks = np.arange(-1.0, 1.01, 0.50)
    legend_handles = None

    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        selective = pd.read_csv(
            paper_candidate_dir(ai_version, k)
            / "selective_reliance_probability.csv"
        )
        state_summary = (
            selective.groupby(["state", "reliance_level"], as_index=False)
            .agg(
                state_mean=("selective_reliance_probability", "mean"),
                n_subjects=("selective_reliance_probability", "count"),
            )
            .sort_values("state")
        )
        for state_index, row in state_summary.reset_index(drop=True).iterrows():
            state = int(row["state"])
            values = selective.loc[
                selective["state"].eq(state),
                "selective_reliance_probability",
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            color = colors[state_index % len(colors)]
            axis.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1.0 / len(values)),
                color=color,
                alpha=0.28,
                edgecolor=color,
                linewidth=0.8,
                label=str(row["reliance_level"]),
            )
            if len(values) > 1 and np.ptp(values) > 1e-10:
                try:
                    density = gaussian_kde(values)(probability_grid) * bin_width
                    axis.plot(
                        probability_grid,
                        density,
                        color=color,
                        linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                    )
                except np.linalg.LinAlgError:
                    pass

        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(-1, 1)
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(
            [f"{value:.1f}" for value in x_ticks],
            rotation=0,
            ha="center",
        )
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=k,
            frameon=False,
        )
    fig.supxlabel("Selective reliance (model estimated)", y=0.025)
    fig.supylabel("Relative frequency", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.16,
        top=0.80,
        wspace=0.30,
    )
    save_paper_combined_figure(fig, output_path)
    write_paper_selective_reliance_report(k, report_path)


def plot_paper_real_selective_reliance_probability_distributions(
    k: int,
    output_path: Path,
) -> None:
    """Plot observed selective-reliance distributions by Viterbi state and AI condition."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PAPER_SELECTIVE_RELIANCE_FIGSIZE,
        sharex=True,
        sharey=False,
    )
    colors = RELIANCE_STATE_COLORS
    bins = np.arange(-1.0, 1.0 + 0.025, 0.025)
    bin_width = float(bins[1] - bins[0])
    probability_grid = np.linspace(-1.0, 1.0, 800)
    x_ticks = np.arange(-1.0, 1.01, 0.50)
    value_column = "real_selective_reliance_probability"
    legend_handles = None

    for panel_index, (axis, ai_version) in enumerate(
        zip(axes, AI_VERSION_LEVELS)
    ):
        selective = pd.read_csv(
            paper_candidate_dir(ai_version, k)
            / "selective_reliance_probability_real_distribution.csv"
        )
        state_summary = (
            selective.groupby(["state", "reliance_level"], as_index=False)
            .agg(
                state_mean=(value_column, "mean"),
                n_subjects=(value_column, "count"),
            )
            .sort_values("state")
        )
        for state_index, row in state_summary.reset_index(drop=True).iterrows():
            state = int(row["state"])
            values = selective.loc[
                selective["state"].eq(state),
                value_column,
            ].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            color = colors[state_index % len(colors)]
            axis.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1.0 / len(values)),
                color=color,
                alpha=0.28,
                edgecolor=color,
                linewidth=0.8,
                label=str(row["reliance_level"]),
            )
            if len(values) > 1 and np.ptp(values) > 1e-10:
                try:
                    density = gaussian_kde(values)(probability_grid) * bin_width
                    axis.plot(
                        probability_grid,
                        density,
                        color=color,
                        linewidth=DISTRIBUTION_CURVE_LINEWIDTH,
                    )
                except np.linalg.LinAlgError:
                    pass

        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.set_title(PAPER_AI_LABELS[ai_version])
        axis.set_xlim(-1, 1)
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(
            [f"{value:.1f}" for value in x_ticks],
            rotation=0,
            ha="center",
        )
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)
        if panel_index == 0:
            legend_handles = axis.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            *legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=k,
            frameon=False,
        )
    fig.supxlabel("Selective reliance (real data)", y=0.025)
    fig.supylabel("Relative frequency", x=0.012)
    fig.subplots_adjust(
        left=0.105,
        right=0.995,
        bottom=0.16,
        top=0.80,
        wspace=0.30,
    )
    save_paper_combined_figure(fig, output_path)


def generate_paper_combined_figures() -> None:
    """Combine results horizontally across LLM conditions by K and figure type."""
    setup_plot_style()
    paper_panel_plotters = {
        "08_distribution_reliance_probability.png": (
            plot_reliance_probability_distribution
        ),
    }
    for k in K_CANDIDATES:
        output_dir = PAPER_RESULTS_DIR / f"K{k}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for figure_name in PAPER_FIGURE_FILES:
            output_path = output_dir / figure_name
            if figure_name == "01_bic_comparison.png":
                plot_paper_bic_comparisons(k, output_path)
                continue
            if figure_name == "02_state_profiles_Viterbi.png":
                plot_paper_state_profiles(k, output_path)
                continue
            if figure_name == "02_state_profiles_Viterbi-new.png":
                plot_paper_state_profiles_visual_mockup(k, output_path)
                continue
            if figure_name == "03_transition_matrix.png":
                plot_paper_transition_matrices(k, output_path)
                continue
            if figure_name == "03_transition_matrix-new.png":
                plot_paper_transition_matrices_visual_mockup(k, output_path)
                continue
            if figure_name == "05_prediction_calibration.png":
                plot_paper_prediction_calibrations(k, output_path)
                continue
            if figure_name == "06_state_proportion.png":
                plot_paper_state_proportions(k, output_path)
                continue
            if figure_name == "06_state_proportion-new.png":
                plot_paper_state_proportions_visual_mockup(k, output_path)
                continue
            if figure_name == "07_reliance_fraction.png":
                plot_paper_reliance_fractions(k, output_path)
                continue
            if figure_name == "08_distribution_reliance_probability.png":
                plot_paper_feedback_bootstrap_distributions(k, output_path)
                continue
            if figure_name == "reliance_probability.png":
                plot_paper_reliance_probability_distributions(k, output_path)
                continue
            if figure_name == "reliance_probability_real_distribution.png":
                plot_paper_real_reliance_probability_distributions(k, output_path)
                continue
            if figure_name == "selective_reliance_probability.png":
                plot_paper_selective_reliance_probability_distributions(
                    k,
                    output_path,
                    output_dir / "selective_reliance_probability.txt",
                )
                continue
            if (
                figure_name
                == "selective_reliance_probability_real_distribution.png"
            ):
                plot_paper_real_selective_reliance_probability_distributions(
                    k,
                    output_path,
                )
                continue

            panel_plotter = paper_panel_plotters.get(figure_name)
            with TemporaryDirectory(prefix="pkhmm-paper-panels-") as temp_dir:
                if panel_plotter is not None:
                    source_paths = []
                    for panel_index, ai_version in enumerate(AI_VERSION_LEVELS):
                        panel_path = Path(temp_dir) / f"panel_{panel_index}.png"
                        panel_plotter(
                            paper_candidate_dir(ai_version, k),
                            PAPER_AI_LABELS[ai_version],
                            output_path=panel_path,
                        )
                        source_paths.append(panel_path)
                else:
                    source_paths = [
                        RESULTS_DIR
                        / ai_version
                        / f"K{k}"
                        / "figures"
                        / figure_name
                        for ai_version in AI_VERSION_LEVELS
                    ]
                    missing = [path for path in source_paths if not path.exists()]
                    if missing:
                        missing_text = "\n".join(str(path) for path in missing)
                        raise FileNotFoundError(
                            "Cannot generate the combined publication figure; missing source images:\n"
                            f"{missing_text}"
                        )

                fig, axes = plt.subplots(1, 3, figsize=PAPER_COMBINED_FIGSIZE)
                for axis, source_path in zip(axes, source_paths):
                    axis.imshow(plt.imread(source_path), interpolation="lanczos")
                    axis.set_axis_off()
                fig.subplots_adjust(
                    left=0.002,
                    right=0.998,
                    bottom=0.002,
                    top=0.998,
                    wspace=0.015,
                )
                fig.savefig(
                    output_path,
                    dpi=300,
                    facecolor="white",
                )
                plt.close(fig)
        progress_write(f"[Combined publication figure] K={k} saved to: {output_dir}")


def all_candidate_parameters_exist(
    result_dir: Path,
    metadata: dict[str, object],
    sequences: list[SequenceData],
) -> bool:
    """Reuse parameters only when their structure, configuration, and input are compatible."""
    expected_config = {
        "n_interactions_per_subject": N_INTERACTIONS_PER_SUBJECT,
    }
    expected_training_configuration = metadata["training_configuration"]
    expected_initial_width = len(sequences[0].initial_x) + 1
    expected_model_width = sequences[0].emission_x.shape[1] + 1
    legacy_snapshot_matches: bool | None = None
    for k in K_CANDIDATES:
        candidate_dir = result_dir / f"K{k}"
        parameter_path = candidate_dir / "model_parameters.npz"
        if not parameter_path.exists():
            return False
        metadata_path = candidate_dir / "model_metadata.json"
        if not metadata_path.exists():
            return False
        try:
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if any(
            saved_metadata.get(key) != value
            for key, value in expected_config.items()
        ):
            return False
        if saved_metadata.get("training_data_signature") != metadata[
            "training_data_signature"
        ]:
            if saved_metadata.get("training_data_signature_version") is not None:
                return False
            if legacy_snapshot_matches is None:
                legacy_snapshot_matches = existing_training_snapshot_matches(
                    sequences, metadata, result_dir
                )
            if not legacy_snapshot_matches:
                return False
        saved_training_configuration = saved_metadata.get("training_configuration")
        if (
            saved_training_configuration is not None
            and saved_training_configuration != expected_training_configuration
        ):
            return False
        try:
            with np.load(parameter_path) as parameters:
                if parameters["initial_coef"].shape != (
                    k - 1,
                    expected_initial_width,
                ):
                    return False
                if parameters["transition_coef"].shape != (
                    k,
                    k - 1,
                    expected_model_width,
                ):
                    return False
                if parameters["emission_coef"].shape != (
                    k,
                    expected_model_width,
                ):
                    return False
        except (OSError, KeyError, ValueError):
            return False
    return True


def load_candidate_comparison(
    result_dir: Path,
    candidate_models: dict[int, PKHMM],
    sequences: list[SequenceData],
) -> pd.DataFrame:
    """
    Load the candidate-model summary, rebuilding it when missing or incompatible with K_CANDIDATES.
    """
    summary_path = result_dir / "candidate_model_summary.csv"
    required_columns = [
        "K",
        "log_likelihood",
        "n_parameters",
        "BIC",
        "iterations",
        "converged",
    ]
    if summary_path.exists():
        comparison = pd.read_csv(summary_path)
        if set(required_columns).issubset(comparison.columns):
            available_k = set(comparison["K"].astype(int))
            if set(K_CANDIDATES).issubset(available_k):
                comparison = comparison[comparison["K"].astype(int).isin(K_CANDIDATES)]
                return comparison[required_columns].sort_values("BIC").reset_index(
                    drop=True
                )

    n_observations = sum(len(sequence.behavior) for sequence in sequences)
    records = []
    for k in K_CANDIDATES:
        model = candidate_models[k]
        model.log_likelihood = sum(
            forward_backward(model, sequence)[0] for sequence in sequences
        )
        records.append(
            {
                "K": k,
                "log_likelihood": model.log_likelihood,
                "n_parameters": parameter_count(model),
                "BIC": bic(model, n_observations),
                "iterations": model.n_iter,
                "converged": model.converged,
            }
        )
    return pd.DataFrame(records).sort_values("BIC").reset_index(drop=True)


def load_models_from_existing_parameters(
    result_dir: Path, sequences: list[SequenceData]
) -> tuple[PKHMM, dict[int, PKHMM], pd.DataFrame]:
    """Load models from each K directory without rerunning EM."""
    candidate_models: dict[int, PKHMM] = {}
    summary_path = result_dir / "candidate_model_summary.csv"
    summary = (
        pd.read_csv(summary_path).set_index("K")
        if summary_path.exists()
        else pd.DataFrame().set_index(pd.Index([], name="K"))
    )

    for k in K_CANDIDATES:
        parameter_path = result_dir / f"K{k}" / "model_parameters.npz"
        parameters = np.load(parameter_path)
        row = summary.loc[k] if k in summary.index else None
        candidate_models[k] = PKHMM(
            k=k,
            initial_coef=parameters["initial_coef"],
            transition_coef=parameters["transition_coef"],
            emission_coef=parameters["emission_coef"],
            log_likelihood=(
                float(row["log_likelihood"])
                if row is not None and "log_likelihood" in row
                else np.nan
            ),
            n_iter=(
                int(row["iterations"])
                if row is not None and "iterations" in row
                else 0
            ),
            converged=(
                bool(row["converged"])
                if row is not None and "converged" in row
                else True
            ),
        )

    comparison = load_candidate_comparison(result_dir, candidate_models, sequences)
    for _, row in comparison.iterrows():
        k = int(row["K"])
        candidate_models[k].log_likelihood = float(row["log_likelihood"])
        candidate_models[k].n_iter = int(row["iterations"])
        candidate_models[k].converged = bool(row["converged"])

    selected_k = int(comparison.iloc[0]["K"])
    return candidate_models[selected_k], candidate_models, comparison


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    user_file = resolve_user_file()
    users = load_users(user_file)
    scenarios = load_scenario_statistics()
    tables, audit = load_trial_tables(scenarios)
    pd.DataFrame(audit).to_csv(RESULTS_DIR / "all_data_audit.csv", index=False)

    model_summaries = []
    ai_progress = progress_bar(
        TRAIN_AI_VERSIONS,
        total=len(TRAIN_AI_VERSIONS),
        desc="Overall AI-model progress",
        unit="AI",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for ai_version in ai_progress:
        ai_progress.set_description(f"AI model: {ai_version}")
        version_tables = [
            table
            for table in tables
            if clean_text(table["ai_version"].iloc[0]).lower() == ai_version
        ]
        if not version_tables:
            progress_write(
                f"Skipping {ai_version}: no eligible "
                f"{N_INTERACTIONS_PER_SUBJECT} interaction rounds"
            )
            continue

        version_audit = [
            item
            for item in audit
            if clean_text(item.get("ai_version")).lower() == ai_version
        ]
        sequences, metadata = build_sequences(version_tables, users)
        metadata["user_file"] = str(user_file)
        metadata["training_data_signature"] = training_data_signature(sequences)
        metadata["training_data_signature_version"] = 2
        metadata["training_configuration"] = current_training_configuration()
        result_dir = RESULTS_DIR / ai_version
        reuse_existing_parameters = all_candidate_parameters_exist(
            result_dir, metadata, sequences
        )
        export_training_data_snapshot(sequences, metadata, result_dir, ai_version)
        progress_write(
            f"[{ai_version}] Modeling data: {len(sequences)} participants, "
            f"{sum(len(sequence.behavior) for sequence in sequences)} interactions"
        )
        if reuse_existing_parameters:
            progress_write(
                f"[{ai_version}] Compatible candidate-K parameters detected; "
                "skipping EM fitting."
            )
            model, candidate_models, comparison = load_models_from_existing_parameters(
                result_dir, sequences
            )
        else:
            model, candidate_models, comparison = fit_models(sequences, ai_version)
        cleanup_ai_root_outputs(result_dir)

        for candidate_k in K_CANDIDATES:
            candidate = candidate_models[candidate_k]
            candidate_dir = result_dir / f"K{candidate_k}"
            candidate_metadata = deepcopy(metadata)
            candidate_metadata.update(
                {
                    "candidate_K": candidate_k,
                    "is_bic_selected": candidate_k == model.k,
                }
            )
            export_results(
                candidate,
                comparison,
                sequences,
                candidate_metadata,
                version_audit,
                users,
                ai_version,
                candidate_dir,
            )
            generate_result_plots(
                candidate_dir, f"{ai_version} K={candidate_k}"
            )
            progress_write(
                f"[{ai_version}] K={candidate_k} results saved: "
                f"{candidate_dir}"
            )

        candidate_summary = comparison.copy()
        candidate_summary["is_bic_selected"] = candidate_summary["K"].eq(model.k)
        candidate_summary["result_dir"] = candidate_summary["K"].map(
            lambda k: str(result_dir / f"K{int(k)}")
        )
        candidate_summary.to_csv(
            result_dir / "candidate_model_summary.csv", index=False
        )

        model_summaries.append(
            {
                "ai_version": ai_version,
                "n_subjects": len(sequences),
                "n_trials": sum(len(sequence.behavior) for sequence in sequences),
                "selected_K": model.k,
                "log_likelihood": model.log_likelihood,
                "BIC": float(
                    comparison.loc[comparison["K"].eq(model.k), "BIC"].iloc[0]
                ),
                "converged": model.converged,
                "result_dir": str(result_dir / f"K{model.k}"),
                "candidate_summary_file": str(
                    result_dir / "candidate_model_summary.csv"
                ),
            }
        )
        progress_write(
            f"[{ai_version}] BIC selected K={model.k}; "
            f"see {result_dir / f'K{model.k}'}"
        )
    ai_progress.close()

    if not model_summaries:
        raise RuntimeError(
            f"No complete PKHMM data are available for the configured training list {TRAIN_AI_VERSIONS}."
        )
    pd.DataFrame(model_summaries).to_csv(
        RESULTS_DIR / "model_summary_by_ai.csv", index=False
    )
    generate_paper_combined_figures()


if __name__ == "__main__":
    main()
