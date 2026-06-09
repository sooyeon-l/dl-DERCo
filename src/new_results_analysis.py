from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


MODEL_LABELS = {
    "cnn": "CNN-v1",
    "cnn_v2": "CNN-v2",
    "eegnet": "EEGNet",
}

MODEL_ORDER = ["cnn", "cnn_v2", "eegnet"]


def infer_derco_root() -> Path:
    """Infer the DERCo project/data root across RunPod and Colab."""
    candidates = [
        os.environ.get("DERCO_ROOT"),
        "/workspace/data/DERCo",
        "/content/drive/MyDrive/Colab_Notebooks/DERCo",
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        root = Path(candidate)
        if root.exists():
            return root

    # Fallback keeps the old Colab path, but users can override with set_derco_root().
    return Path("/content/drive/MyDrive/Colab_Notebooks/DERCo")


DERCO_ROOT = infer_derco_root()
RUNS_ROOT = DERCO_ROOT / "outputs" / "runs"
ANALYSIS_ROOT = DERCO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_ROOT / "tables"
FIGURES_DIR = ANALYSIS_ROOT / "figures"


def set_derco_root(root: str | Path) -> Path:
    """
    Update module-level paths after import.

    Use this in notebooks if your data moved:
        import results_analysis as ra
        ra.set_derco_root("/workspace/data/DERCo")
    """
    global DERCO_ROOT, RUNS_ROOT, ANALYSIS_ROOT, TABLES_DIR, FIGURES_DIR

    DERCO_ROOT = Path(root)
    RUNS_ROOT = DERCO_ROOT / "outputs" / "runs"
    ANALYSIS_ROOT = DERCO_ROOT / "analysis"
    TABLES_DIR = ANALYSIS_ROOT / "tables"
    FIGURES_DIR = ANALYSIS_ROOT / "figures"

    return DERCO_ROOT


def ensure_analysis_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(value, default: float = 0.5) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _model_sort_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def _normalize_subject_set(value) -> str:
    """
    Normalize selected_subjects for grouping.

    selected_subjects may be a list from JSON, a stringified list, or missing.
    We only need stable equality, not parsing for computation.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return "|".join(sorted(map(str, value)))

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def compute_oof_metrics(oof_path: Path, threshold: float = 0.5) -> dict:
    data = np.load(oof_path)

    y_prob = data["oof_probs"].reshape(-1)
    y_true = data["oof_labels"].reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "oof_auc": roc_auc_score(y_true, y_prob),
        "oof_accuracy": accuracy_score(y_true, y_pred),
        "oof_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "oof_f1": f1_score(y_true, y_pred),
        "n_oof_samples": len(y_true),
        "threshold_used": threshold,
    }


def collect_run(run_dir: Path) -> dict | None:
    """
    Collect one completed run folder.

    Expected files:
        run_summary.json
        config_snapshot.json
        best_summary.csv
        oof_predictions.npz
    """
    run_summary_path = run_dir / "run_summary.json"
    config_snapshot_path = run_dir / "config_snapshot.json"
    oof_path = run_dir / "oof_predictions.npz"
    best_summary_path = run_dir / "best_summary.csv"

    if not run_summary_path.exists():
        return None

    run_summary = load_json(run_summary_path)
    row = {
        "run_dir": str(run_dir),
        **run_summary,
    }

    if config_snapshot_path.exists():
        config = load_json(config_snapshot_path)

        row.update({
            "model_name": config.get("model_name"),
            "train_val_path": config.get("train_val_path"),
            "X_train_val_shape": config.get("X_train_val_shape"),
            "y_train_val_shape": config.get("y_train_val_shape"),
            "subjects_train_val_shape": config.get("subjects_train_val_shape"),
            "n_all_unique_subjects": config.get("n_all_unique_subjects"),
            "n_unique_subjects": config.get("n_unique_subjects"),
            "n_selected_subjects_config": config.get("n_selected_subjects"),
            "selected_subjects_config": config.get("selected_subjects"),
            "random_seed": config.get("random_seed"),
            "checkpoint_dir": config.get("checkpoint_dir"),
            "cnn_kernel_len": config.get("cnn_kernel_len"),
            "eegnet_kernel_len": config.get("eegnet_kernel_len"),
        })

        for key in [
            "x_path",
            "y_path",
            "subjects_path",
            "expected_timepoints",
            "actual_timepoints",
        ]:
            if key in config:
                row[key] = config[key]

    if best_summary_path.exists():
        best_df = pd.read_csv(best_summary_path)

        if "best_val_roc_auc" in best_df.columns:
            row["mean_fold_auc"] = best_df["best_val_roc_auc"].mean()
            row["std_fold_auc"] = best_df["best_val_roc_auc"].std()

        if "best_val_balanced_accuracy" in best_df.columns:
            row["mean_fold_bal_acc"] = best_df["best_val_balanced_accuracy"].mean()
            row["std_fold_bal_acc"] = best_df["best_val_balanced_accuracy"].std()

        if "best_epoch" in best_df.columns:
            row["mean_best_epoch"] = best_df["best_epoch"].mean()

    if oof_path.exists():
        threshold = _safe_float(run_summary.get("global_threshold"), default=0.5)
        row.update(compute_oof_metrics(oof_path, threshold=threshold))

    return row


def add_run_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["category"] = "unknown"
    run_dir = df["run_dir"].astype(str)

    df.loc[run_dir.str.contains("/baseline/|\\\\baseline\\\\", regex=True), "category"] = "baseline"
    df.loc[
        run_dir.str.contains("/subject_ablation/|\\\\subject_ablation\\\\", regex=True),
        "category",
    ] = "subject_ablation"

    return df


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "model" in df.columns:
        df["model_display"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
        df["model_order"] = df["model"].map(_model_sort_key)

    if "selected_subjects" in df.columns:
        df["selected_subjects_key"] = df["selected_subjects"].map(_normalize_subject_set)
    elif "selected_subjects_config" in df.columns:
        df["selected_subjects_key"] = df["selected_subjects_config"].map(_normalize_subject_set)

    return df


def collect_all_runs(runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    rows = []

    for run_summary_path in Path(runs_root).rglob("run_summary.json"):
        row = collect_run(run_summary_path.parent)

        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No run_summary.json files found under {runs_root}")

    df = pd.DataFrame(rows)
    df = add_run_category(df)
    df = add_display_columns(df)

    return df


def make_run_integrity_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit_cols = [
        "category",
        "run_name",
        "model",
        "model_display",
        "model_name",
        "window",
        "sfreq",
        "shuffle_labels",
        "n_subjects_requested",
        "ablation_seed",
        "n_selected_subjects",
        "n_selected_subjects_config",
        "n_all_unique_subjects",
        "n_unique_subjects",
        "n_folds",
        "X_train_val_shape",
        "expected_timepoints",
        "actual_timepoints",
        "train_val_path",
        "x_path",
        "checkpoint_dir",
        "run_dir",
    ]

    existing_cols = [c for c in audit_cols if c in df.columns]
    sort_cols = [
        c for c in ["category", "model_order", "sfreq", "window", "n_subjects_requested", "ablation_seed"]
        if c in df.columns
    ]

    audit = df.copy()
    if sort_cols:
        audit = audit.sort_values(sort_cols)

    return audit[existing_cols].reset_index(drop=True)


def find_possible_integrity_issues(audit: pd.DataFrame) -> pd.DataFrame:
    issues = []

    for _, row in audit.iterrows():
        run_name = row.get("run_name", "unknown")

        if "expected_timepoints" in audit.columns and "actual_timepoints" in audit.columns:
            expected = row.get("expected_timepoints")
            actual = row.get("actual_timepoints")

            if pd.notna(expected) and pd.notna(actual) and int(expected) != int(actual):
                issues.append({
                    "run_name": run_name,
                    "issue": "expected_timepoints != actual_timepoints",
                    "expected": expected,
                    "actual": actual,
                })

        if row.get("category") == "subject_ablation":
            if pd.isna(row.get("n_subjects_requested")):
                issues.append({
                    "run_name": run_name,
                    "issue": "subject_ablation run missing n_subjects_requested",
                    "expected": "not null",
                    "actual": row.get("n_subjects_requested"),
                })

            if pd.isna(row.get("ablation_seed")):
                issues.append({
                    "run_name": run_name,
                    "issue": "subject_ablation run missing ablation_seed",
                    "expected": "not null",
                    "actual": row.get("ablation_seed"),
                })

        if row.get("category") == "baseline":
            if pd.notna(row.get("n_subjects_requested")):
                issues.append({
                    "run_name": run_name,
                    "issue": "baseline run has n_subjects_requested",
                    "expected": "NaN/None",
                    "actual": row.get("n_subjects_requested"),
                })

    return pd.DataFrame(issues)


def make_baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["category"] == "baseline"].copy()

    if baseline.empty:
        return baseline

    cols = [
        "run_name",
        "model",
        "model_display",
        "window",
        "sfreq",
        "shuffle_labels",
        "oof_auc",
        "oof_accuracy",
        "oof_balanced_accuracy",
        "oof_f1",
        "cv_balanced_accuracy_at_global_threshold",
        "global_threshold",
        "threshold_used",
        "n_oof_samples",
        "X_train_val_shape",
        "x_path",
        "checkpoint_dir",
        "run_dir",
    ]

    existing_cols = [c for c in cols if c in baseline.columns]
    sort_cols = [c for c in ["sfreq", "model_order", "window", "shuffle_labels"] if c in baseline.columns]

    if sort_cols:
        baseline = baseline.sort_values(sort_cols)

    return baseline[existing_cols].reset_index(drop=True)


def make_shuffled_sanity_table(baseline_summary: pd.DataFrame) -> pd.DataFrame:
    if baseline_summary.empty:
        return pd.DataFrame()

    baseline = baseline_summary.copy()
    shuffled = baseline[baseline["shuffle_labels"] == True].copy()
    real = baseline[baseline["shuffle_labels"] == False].copy()

    rows = []

    for _, shuffle_row in shuffled.iterrows():
        model = shuffle_row["model"]
        window = shuffle_row["window"]
        sfreq = shuffle_row["sfreq"]

        matched = real[
            (real["model"] == model)
            & (real["window"] == window)
            & (real["sfreq"] == sfreq)
        ]

        if matched.empty:
            rows.append({
                "model": model,
                "model_display": MODEL_LABELS.get(model, model),
                "window": window,
                "sfreq": sfreq,
                "real_run": None,
                "shuffle_run": shuffle_row["run_name"],
                "real_auc": np.nan,
                "shuffle_auc": shuffle_row["oof_auc"],
                "auc_above_shuffle": np.nan,
                "real_bal_acc": np.nan,
                "shuffle_bal_acc": shuffle_row["oof_balanced_accuracy"],
                "bal_acc_above_shuffle": np.nan,
            })
            continue

        real_row = matched.iloc[0]

        rows.append({
            "model": model,
            "model_display": MODEL_LABELS.get(model, model),
            "window": window,
            "sfreq": sfreq,
            "real_run": real_row["run_name"],
            "shuffle_run": shuffle_row["run_name"],
            "real_auc": real_row["oof_auc"],
            "shuffle_auc": shuffle_row["oof_auc"],
            "auc_above_shuffle": real_row["oof_auc"] - shuffle_row["oof_auc"],
            "real_bal_acc": real_row["oof_balanced_accuracy"],
            "shuffle_bal_acc": shuffle_row["oof_balanced_accuracy"],
            "bal_acc_above_shuffle": (
                real_row["oof_balanced_accuracy"] - shuffle_row["oof_balanced_accuracy"]
            ),
        })

    return pd.DataFrame(rows)


def make_architecture_comparison(
    baseline_summary: pd.DataFrame,
    run_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compare full-window architecture baselines.

    Missing runs are preserved with found=False. This is intentional because CNN-v2
    baseline/shuffle may not exist yet even if CNN-v2 subject_ablation exists.
    """
    if run_names is None:
        run_names = [
            "cnn_0800_250hz_shuffle",
            "cnn_0800_250hz",
            "cnn_v2_0800_250hz_shuffle",
            "cnn_v2_0800_250hz",
            "eegnet_0800_250hz_shuffle",
            "eegnet_0800_250hz",
        ]

    rows = []

    for run_name in run_names:
        matched = baseline_summary[baseline_summary["run_name"] == run_name]

        if matched.empty:
            rows.append({
                "found": False,
                "run_name": run_name,
            })
        else:
            row = matched.iloc[0].to_dict()
            row["found"] = True
            rows.append(row)

    comp = pd.DataFrame(rows)

    preferred_cols = [
        "found",
        "run_name",
        "model",
        "model_display",
        "window",
        "sfreq",
        "shuffle_labels",
        "oof_auc",
        "oof_balanced_accuracy",
        "oof_f1",
        "threshold_used",
        "n_oof_samples",
        "run_dir",
    ]

    existing_cols = [c for c in preferred_cols if c in comp.columns]
    return comp[existing_cols]


def make_architecture_difference_table(architecture_df: pd.DataFrame) -> pd.DataFrame:
    available = architecture_df[architecture_df["found"] == True].copy()

    if available.empty:
        return pd.DataFrame()

    df = available.set_index("run_name")

    comparisons = [
        ("CNN-v2 minus CNN-v1", "cnn_v2_0800_250hz", "cnn_0800_250hz"),
        ("EEGNet minus CNN-v1", "eegnet_0800_250hz", "cnn_0800_250hz"),
        ("EEGNet minus CNN-v2", "eegnet_0800_250hz", "cnn_v2_0800_250hz"),
        ("CNN-v1 minus CNN-v1 shuffled", "cnn_0800_250hz", "cnn_0800_250hz_shuffle"),
        ("CNN-v2 minus CNN-v2 shuffled", "cnn_v2_0800_250hz", "cnn_v2_0800_250hz_shuffle"),
        ("EEGNet minus EEGNet shuffled", "eegnet_0800_250hz", "eegnet_0800_250hz_shuffle"),
    ]

    rows = []

    for label, a, b in comparisons:
        if a not in df.index or b not in df.index:
            rows.append({
                "comparison": label,
                "run_a": a,
                "run_b": b,
                "available": False,
                "auc_difference": np.nan,
                "balanced_accuracy_difference": np.nan,
                "f1_difference": np.nan,
            })
            continue

        rows.append({
            "comparison": label,
            "run_a": a,
            "run_b": b,
            "available": True,
            "auc_difference": df.loc[a, "oof_auc"] - df.loc[b, "oof_auc"],
            "balanced_accuracy_difference": (
                df.loc[a, "oof_balanced_accuracy"] - df.loc[b, "oof_balanced_accuracy"]
            ),
            "f1_difference": df.loc[a, "oof_f1"] - df.loc[b, "oof_f1"],
        })

    return pd.DataFrame(rows)


def plot_architecture_comparison(
    architecture_df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    plot_df = architecture_df[architecture_df["found"] == True].copy()

    if plot_df.empty:
        raise ValueError("No available architecture runs to plot.")

    label_map = {
        "cnn_0800_250hz_shuffle": "CNN-v1 shuffled",
        "cnn_0800_250hz": "CNN-v1",
        "cnn_v2_0800_250hz_shuffle": "CNN-v2 shuffled",
        "cnn_v2_0800_250hz": "CNN-v2",
        "eegnet_0800_250hz_shuffle": "EEGNet shuffled",
        "eegnet_0800_250hz": "EEGNet",
    }

    plot_df["label"] = plot_df["run_name"].map(label_map).fillna(plot_df["run_name"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(plot_df))

    ax.bar(x, plot_df["oof_auc"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=20, ha="right")
    ax.set_ylabel("Out-of-fold ROC AUC")
    ax.set_title("Architecture comparison on 0–800 ms, 250 Hz")
    ax.set_ylim(0.50, max(plot_df["oof_auc"]) + 0.04)

    for i, val in enumerate(plot_df["oof_auc"].values):
        ax.text(i, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def make_temporal_window_summary(baseline_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline-only temporal-window summary.

    If CNN-v2 baseline windows are missing, this returns only CNN-v1 rows.
    That is correct: CNN-v2 subject_ablation runs are not temporal-window baselines.
    """
    if baseline_summary.empty:
        return baseline_summary

    wanted_models = ["cnn", "cnn_v2"]
    temporal = baseline_summary[
        (baseline_summary["model"].isin(wanted_models))
        & (baseline_summary["sfreq"] == 250)
        & (baseline_summary["shuffle_labels"] == False)
    ].copy()

    if temporal.empty:
        return temporal

    window_label_map = {
        "0200": "0–200 ms",
        "300500": "300–500 ms",
        "500800": "500–800 ms",
        "0800": "0–800 ms",
    }

    window_order = ["0200", "300500", "500800", "0800"]

    temporal["window_label"] = temporal["window"].map(window_label_map)
    temporal["window"] = pd.Categorical(
        temporal["window"],
        categories=window_order,
        ordered=True,
    )

    cols = [
        "run_name",
        "model",
        "model_display",
        "window",
        "window_label",
        "sfreq",
        "oof_auc",
        "oof_balanced_accuracy",
        "oof_f1",
        "threshold_used",
        "n_oof_samples",
    ]

    existing_cols = [c for c in cols if c in temporal.columns]

    return (
        temporal[existing_cols]
        .sort_values(["window", "model"])
        .reset_index(drop=True)
    )


def make_cnn_v2_window_delta_table(temporal_summary: pd.DataFrame) -> pd.DataFrame:
    if temporal_summary.empty:
        return pd.DataFrame()

    cnn = temporal_summary[temporal_summary["model"] == "cnn"].copy()
    cnn_v2 = temporal_summary[temporal_summary["model"] == "cnn_v2"].copy()

    if cnn.empty or cnn_v2.empty:
        return pd.DataFrame()

    merged = cnn.merge(
        cnn_v2,
        on=["window", "window_label", "sfreq"],
        suffixes=("_cnn", "_cnn_v2"),
    )

    return pd.DataFrame({
        "window": merged["window"],
        "window_label": merged["window_label"],
        "cnn_auc": merged["oof_auc_cnn"],
        "cnn_v2_auc": merged["oof_auc_cnn_v2"],
        "auc_delta_v2_minus_cnn": merged["oof_auc_cnn_v2"] - merged["oof_auc_cnn"],
        "cnn_bal_acc": merged["oof_balanced_accuracy_cnn"],
        "cnn_v2_bal_acc": merged["oof_balanced_accuracy_cnn_v2"],
        "bal_acc_delta_v2_minus_cnn": (
            merged["oof_balanced_accuracy_cnn_v2"] - merged["oof_balanced_accuracy_cnn"]
        ),
        "cnn_f1": merged["oof_f1_cnn"],
        "cnn_v2_f1": merged["oof_f1_cnn_v2"],
        "f1_delta_v2_minus_cnn": merged["oof_f1_cnn_v2"] - merged["oof_f1_cnn"],
    }).reset_index(drop=True)


def plot_temporal_window_auc(
    temporal_summary: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    if temporal_summary.empty:
        raise ValueError("Temporal-window summary is empty.")

    window_order = ["0–200 ms", "300–500 ms", "500–800 ms", "0–800 ms"]
    models = [m for m in MODEL_ORDER if m in temporal_summary["model"].unique()]

    pivot = (
        temporal_summary
        .pivot(index="window_label", columns="model", values="oof_auc")
        .reindex(window_order)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(window_order))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        if model not in pivot.columns:
            continue

        offset = (i - (len(models) - 1) / 2) * width
        vals = pivot[model].values
        label = MODEL_LABELS.get(model, model)

        ax.bar(x + offset, vals, width, label=label)

        for j, val in enumerate(vals):
            if pd.notna(val):
                ax.text(
                    j + offset,
                    val + 0.003,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(window_order)
    ax.set_ylabel("Out-of-fold ROC AUC")
    ax.set_title("CNN-family temporal-window comparison")
    ax.set_ylim(0.50, np.nanmax(pivot.values) + 0.04)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_cnn_v2_window_delta(
    delta_df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    if delta_df.empty:
        raise ValueError("CNN-v2 window delta table is empty. CNN-v2 baseline windows may be missing.")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(delta_df))
    vals = delta_df["auc_delta_v2_minus_cnn"].values

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(delta_df["window_label"])
    ax.set_ylabel("AUC difference: CNN-v2 − CNN-v1")
    ax.set_title("CNN-v2 improvement over CNN-v1 by window")

    for i, val in enumerate(vals):
        va = "bottom" if val >= 0 else "top"
        offset = 0.001 if val >= 0 else -0.001
        ax.text(i, val + offset, f"{val:+.3f}", ha="center", va=va, fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def make_subject_ablation_runs(df: pd.DataFrame) -> pd.DataFrame:
    ablation = df[df["category"] == "subject_ablation"].copy()

    if ablation.empty:
        return ablation

    if "n_selected_subjects" in ablation.columns:
        ablation["subject_count"] = ablation["n_selected_subjects"]
    else:
        ablation["subject_count"] = ablation["n_subjects_requested"]

    if "selected_subjects_key" not in ablation.columns:
        if "selected_subjects" in ablation.columns:
            ablation["selected_subjects_key"] = ablation["selected_subjects"].map(_normalize_subject_set)
        elif "selected_subjects_config" in ablation.columns:
            ablation["selected_subjects_key"] = ablation["selected_subjects_config"].map(_normalize_subject_set)
        else:
            ablation["selected_subjects_key"] = ""

    cols = [
        "run_name",
        "model",
        "model_display",
        "subject_count",
        "n_subjects_requested",
        "ablation_seed",
        "selected_subjects",
        "selected_subjects_key",
        "n_folds",
        "oof_auc",
        "oof_accuracy",
        "oof_balanced_accuracy",
        "oof_f1",
        "global_threshold",
        "threshold_used",
        "n_oof_samples",
        "run_dir",
    ]

    existing_cols = [c for c in cols if c in ablation.columns]

    return (
        ablation[existing_cols]
        .sort_values(["model", "subject_count", "ablation_seed"])
        .reset_index(drop=True)
    )


def safe_std(x):
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1)


def safe_sem(x):
    if len(x) <= 1:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))


def make_subject_ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    ablation = make_subject_ablation_runs(df)

    if ablation.empty:
        return ablation

    summary = (
        ablation
        .groupby(["model", "model_display", "subject_count"], as_index=False)
        .agg(
            mean_auc=("oof_auc", "mean"),
            std_auc=("oof_auc", safe_std),
            sem_auc=("oof_auc", safe_sem),

            mean_bal_acc=("oof_balanced_accuracy", "mean"),
            std_bal_acc=("oof_balanced_accuracy", safe_std),
            sem_bal_acc=("oof_balanced_accuracy", safe_sem),

            n_runs=("run_name", "count"),
            n_unique_subject_sets=("selected_subjects_key", lambda x: x.nunique()),
            mean_n_oof_samples=("n_oof_samples", "mean"),
        )
        .reset_index(drop=True)
    )

    # If a condition only has one unique subject set, error bars represent no
    # random-subset variability. This especially fixes n=18 for CNN-v1/EEGNet,
    # where repeated seeds select the same full train/val subject pool.
    no_subset_variance = summary["n_unique_subject_sets"] <= 1

    summary.loc[
        no_subset_variance,
        ["std_auc", "sem_auc", "std_bal_acc", "sem_bal_acc"]
    ] = np.nan

    summary["model_order"] = summary["model"].map(_model_sort_key)

    return (
        summary
        .sort_values(["model_order", "subject_count"])
        .drop(columns=["model_order"])
        .reset_index(drop=True)
    )


def get_model_specific_shuffle_auc(
    baseline_summary: pd.DataFrame,
    window: str = "0800",
    sfreq: int = 250,
) -> dict[str, float]:
    if baseline_summary.empty:
        return {}

    rows = baseline_summary[
        (baseline_summary["shuffle_labels"] == True)
        & (baseline_summary["window"] == window)
        & (baseline_summary["sfreq"] == sfreq)
    ]

    return {
        row["model"]: float(row["oof_auc"])
        for _, row in rows.iterrows()
    }


def add_auc_above_model_shuffle(
    ablation_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    window: str = "0800",
    sfreq: int = 250,
) -> pd.DataFrame:
    adjusted = ablation_summary.copy()
    shuffle_auc = get_model_specific_shuffle_auc(baseline_summary, window=window, sfreq=sfreq)

    adjusted["model_shuffle_auc"] = adjusted["model"].map(shuffle_auc)
    adjusted["mean_auc_above_shuffle"] = adjusted["mean_auc"] - adjusted["model_shuffle_auc"]

    return adjusted


def plot_subject_ablation_auc(
    summary: pd.DataFrame,
    save_path: Optional[Path] = None,
    title: str = "Subject-count ablation: model sample efficiency",
):
    if summary.empty:
        raise ValueError("Subject ablation summary is empty.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for model, src in summary.groupby("model", sort=False):
        src = src.sort_values("subject_count")

        ax.errorbar(
            src["subject_count"],
            src["mean_auc"],
            yerr=src["sem_auc"],
            marker="o",
            capsize=3,
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xlabel("Number of subjects included in CV pool")
    ax.set_ylabel("Out-of-fold ROC AUC")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_subject_ablation_bal_acc(
    summary: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    if summary.empty:
        raise ValueError("Subject ablation summary is empty.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for model, src in summary.groupby("model", sort=False):
        src = src.sort_values("subject_count")

        ax.errorbar(
            src["subject_count"],
            src["mean_bal_acc"],
            yerr=src["sem_bal_acc"],
            marker="o",
            capsize=3,
            label=MODEL_LABELS.get(model, model),
        )

    ax.set_xlabel("Number of subjects included in CV pool")
    ax.set_ylabel("Out-of-fold balanced accuracy")
    ax.set_title("Subject-count ablation: balanced accuracy")
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_subject_ablation_auc_above_shuffle(
    adjusted_summary: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    plot_df = adjusted_summary.dropna(subset=["mean_auc_above_shuffle"]).copy()

    if plot_df.empty:
        raise ValueError("No model-specific shuffled baselines available for adjusted plot.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for model, src in plot_df.groupby("model", sort=False):
        src = src.sort_values("subject_count")

        ax.errorbar(
            src["subject_count"],
            src["mean_auc_above_shuffle"],
            yerr=src["sem_auc"],
            marker="o",
            capsize=3,
            label=MODEL_LABELS.get(model, model),
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of subjects included in CV pool")
    ax.set_ylabel("AUC above model-specific shuffled baseline")
    ax.set_title("Subject-count ablation normalized by model-specific shuffled baseline")
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def save_priority_tables(
    all_runs: pd.DataFrame,
    integrity_audit: pd.DataFrame,
    integrity_issues: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    shuffled_sanity: pd.DataFrame,
    architecture_comparison: pd.DataFrame,
    architecture_differences: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    temporal_delta: pd.DataFrame,
    ablation_runs: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    ablation_adjusted: Optional[pd.DataFrame] = None,
) -> None:
    ensure_analysis_dirs()

    all_runs.to_csv(TABLES_DIR / "all_runs_summary.csv", index=False)
    integrity_audit.to_csv(TABLES_DIR / "run_integrity_audit.csv", index=False)
    integrity_issues.to_csv(TABLES_DIR / "run_integrity_issues.csv", index=False)
    baseline_summary.to_csv(TABLES_DIR / "baseline_oof_summary.csv", index=False)
    shuffled_sanity.to_csv(TABLES_DIR / "shuffled_label_sanity_checks.csv", index=False)
    architecture_comparison.to_csv(TABLES_DIR / "architecture_comparison_0800.csv", index=False)
    architecture_differences.to_csv(TABLES_DIR / "architecture_differences_0800.csv", index=False)
    temporal_summary.to_csv(TABLES_DIR / "temporal_window_summary.csv", index=False)
    temporal_delta.to_csv(TABLES_DIR / "cnn_v2_temporal_delta.csv", index=False)
    ablation_runs.to_csv(TABLES_DIR / "subject_ablation_runs.csv", index=False)
    ablation_summary.to_csv(TABLES_DIR / "subject_ablation_summary.csv", index=False)

    if ablation_adjusted is not None:
        ablation_adjusted.to_csv(
            TABLES_DIR / "subject_ablation_auc_above_model_shuffle.csv",
            index=False,
        )


# ── Additional imports needed for statistical analyses ────────────────────────
from scipy import stats as _scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# 10. Paired t-tests + Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

# n=18 is excluded: CNN-v1 and EEGNet have std=0 across seeds (all seeds select
# the same complete subject pool), making paired t-tests meaningless. CNN-v2
# has only one seed at n=18 for the same reason.
# n=12 is excluded from three-way comparisons: only CNN-v2 has data there.
DEFAULT_TTEST_N = [2, 4, 6, 8, 10, 14]


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1))


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def make_paired_ttest_results(
    ablation_runs: pd.DataFrame,
    valid_n: list[int] | None = None,
) -> pd.DataFrame:
    """
    Paired t-tests and Cohen's d for all pairwise model comparisons.

    Pairing is by ablation_seed: the same seed selects the same subjects
    for all models at a given n_subjects level, making the pairing valid.

    Bonferroni correction is applied across all tests in the table.
    """
    if ablation_runs.empty:
        return pd.DataFrame()

    if valid_n is None:
        valid_n = DEFAULT_TTEST_N

    pairs = [
        ("cnn", "eegnet"),
        ("cnn", "cnn_v2"),
        ("eegnet", "cnn_v2"),
    ]

    rows = []

    for model_a, model_b in pairs:
        for n in valid_n:
            a = (
                ablation_runs[
                    (ablation_runs["model"] == model_a)
                    & (ablation_runs["subject_count"] == n)
                ]
                .sort_values("ablation_seed")["oof_auc"]
                .values
            )
            b = (
                ablation_runs[
                    (ablation_runs["model"] == model_b)
                    & (ablation_runs["subject_count"] == n)
                ]
                .sort_values("ablation_seed")["oof_auc"]
                .values
            )

            if len(a) != 5 or len(b) != 5:
                continue

            diff = a - b
            t_stat, p_val = _scipy_stats.ttest_rel(a, b)
            d = _cohens_d_paired(a, b)

            rows.append(
                {
                    "comparison": f"{model_a} vs {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_subjects": int(n),
                    "mean_a": round(float(a.mean()), 4),
                    "mean_b": round(float(b.mean()), 4),
                    "mean_diff": round(float(diff.mean()), 4),
                    "std_diff": round(float(diff.std(ddof=1)), 4),
                    "t_stat": round(float(t_stat), 3),
                    "p_value": round(float(p_val), 4),
                    "cohens_d": round(d, 3),
                }
            )

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)

    # Bonferroni correction across all tests in the table
    n_tests = len(results)
    results["p_bonferroni"] = (results["p_value"] * n_tests).clip(upper=1.0).round(4)
    results["sig_raw"] = results["p_value"].apply(_sig_stars)
    results["sig_bonferroni"] = results["p_bonferroni"].apply(_sig_stars)

    return results


def plot_cohens_d_by_n(
    ttest_results: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple:
    """
    Line plot of Cohen's d across subject counts for each pairwise comparison.

    Positive d = model_a > model_b. A horizontal dashed line at d=0 marks
    no difference. Bonferroni-significant points are annotated with stars.
    """
    if ttest_results.empty:
        raise ValueError("Paired t-test results are empty.")

    comparisons = ttest_results["comparison"].unique()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for comp in comparisons:
        sub = ttest_results[ttest_results["comparison"] == comp].sort_values("n_subjects")
        ax.plot(sub["n_subjects"], sub["cohens_d"], marker="o", label=comp)

        for _, row in sub.iterrows():
            if row["sig_bonferroni"] != "ns":
                ax.annotate(
                    row["sig_bonferroni"],
                    xy=(row["n_subjects"], row["cohens_d"]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Number of subjects included in CV pool")
    ax.set_ylabel("Cohen's d (paired)")
    ax.set_title("Pairwise model differences: Cohen's d by subject count")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_ttest_significance_grid(
    ttest_results: pd.DataFrame,
    save_path: Path | None = None,
) -> tuple:
    """
    Heatmap showing significance (Bonferroni-corrected) across comparisons × n_subjects.

    Cell colour encodes Cohen's d magnitude; cell text shows significance stars.
    """
    if ttest_results.empty:
        raise ValueError("Paired t-test results are empty.")

    comparisons = ttest_results["comparison"].unique()
    n_values = sorted(ttest_results["n_subjects"].unique())

    d_matrix = np.full((len(comparisons), len(n_values)), np.nan)
    sig_matrix = [[""] * len(n_values) for _ in comparisons]

    comp_idx = {c: i for i, c in enumerate(comparisons)}
    n_idx = {n: i for i, n in enumerate(n_values)}

    for _, row in ttest_results.iterrows():
        i = comp_idx[row["comparison"]]
        j = n_idx[row["n_subjects"]]
        d_matrix[i, j] = row["cohens_d"]
        sig_matrix[i][j] = row["sig_bonferroni"]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(d_matrix, cmap="RdBu_r", aspect="auto", vmin=-14, vmax=14)

    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([f"n={n}" for n in n_values])
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels(comparisons)

    for i in range(len(comparisons)):
        for j in range(len(n_values)):
            d_val = d_matrix[i, j]
            stars = sig_matrix[i][j]
            if not np.isnan(d_val):
                text = f"{d_val:.1f}\n{stars}" if stars != "ns" else f"{d_val:.1f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="Cohen's d")
    ax.set_title("Pairwise comparisons: Cohen's d (Bonferroni-corrected stars)")
    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 11. OOF predicted probability distributions
# ─────────────────────────────────────────────────────────────────────────────

def load_oof_predictions(run_dir: str | Path) -> dict | None:
    """
    Load oof_probs and oof_labels from a run directory.
    Returns None if the npz file is missing.
    """
    path = Path(run_dir) / "oof_predictions.npz"
    if not path.exists():
        return None

    data = np.load(path)
    return {
        "oof_probs":  data["oof_probs"].reshape(-1),
        "oof_labels": data["oof_labels"].reshape(-1),
    }


def make_oof_prob_distributions(
    baseline_summary: pd.DataFrame,
    target_runs: list[str] | None = None,
) -> dict[str, dict]:
    """
    Collect OOF probability distributions for baseline runs.

    Returns a dict keyed by run_name, each with:
        probs_high: probabilities for high-cloze trials (y=1)
        probs_low:  probabilities for low-cloze trials  (y=0)
        model:      model name
        model_display: display label
    """
    if baseline_summary.empty:
        return {}

    if target_runs is not None:
        rows = baseline_summary[baseline_summary["run_name"].isin(target_runs)]
    else:
        # Default: real-label baselines only, 0–800 ms, 250 Hz
        rows = baseline_summary[
            (baseline_summary["shuffle_labels"] == False)
            & (baseline_summary["window"] == "0800")
            & (baseline_summary["sfreq"] == 250)
        ]

    distributions = {}

    for _, row in rows.iterrows():
        preds = load_oof_predictions(row["run_dir"])
        if preds is None:
            continue

        probs  = preds["oof_probs"]
        labels = preds["oof_labels"]

        distributions[row["run_name"]] = {
            "probs_high":    probs[labels == 1],
            "probs_low":     probs[labels == 0],
            "model":         row["model"],
            "model_display": MODEL_LABELS.get(row["model"], row["model"]),
        }

    return distributions


def plot_oof_probability_distributions(
    distributions: dict,
    save_path: Path | None = None,
) -> tuple:
    """
    KDE plot of predicted probabilities split by true label.

    One subplot per model. Blue = low-cloze (y=0), orange = high-cloze (y=1).
    Separation between the two distributions is the signal the model found.
    """
    from scipy.stats import gaussian_kde

    if not distributions:
        raise ValueError("No OOF distributions to plot.")

    model_order = [m for m in MODEL_ORDER if any(
        v["model"] == m for v in distributions.values()
    )]
    n_models = len(model_order)

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    x_grid = np.linspace(0, 1, 300)

    for ax, model in zip(axes, model_order):
        entries = [v for v in distributions.values() if v["model"] == model]
        if not entries:
            ax.set_visible(False)
            continue

        # Aggregate across runs with the same model
        probs_low  = np.concatenate([e["probs_low"]  for e in entries])
        probs_high = np.concatenate([e["probs_high"] for e in entries])

        for probs, label, color in [
            (probs_low,  "Low cloze (y=0)",  "steelblue"),
            (probs_high, "High cloze (y=1)", "darkorange"),
        ]:
            if len(probs) < 2:
                continue
            kde = gaussian_kde(probs)
            ax.plot(x_grid, kde(x_grid), label=label, color=color)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=color)

        display_label = MODEL_LABELS.get(model, model)
        ax.set_title(display_label)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle("OOF predicted probability distributions by true label", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# 12. Gradient saliency maps
# ─────────────────────────────────────────────────────────────────────────────

# Standard 32-channel 10-20 layout matching the DERCo dataset.
# Override by passing channel_names explicitly to any plot function.
DERCO_CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]


def _load_fold_checkpoint(ckpt_path: Path, device: str = "cpu") -> tuple[dict, dict]:
    """Load a fold checkpoint. Returns (state_dict, extras)."""
    import torch

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        return ckpt, {}

    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt:
            state_dict = ckpt[key]
            extras = {k: v for k, v in ckpt.items() if k != key}
            return state_dict, extras

    # Checkpoint IS the state dict
    return ckpt, {}


def compute_gradient_saliency(
    run_dir: str | Path,
    X_raw: np.ndarray,
    y: np.ndarray,
    model_name: str,
    sfreq: int = 250,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """
    Compute gradient-based saliency maps from a trained model's fold checkpoints.

    For each fold checkpoint:
      1. Loads the fold's z-scoring statistics (saved inside the checkpoint)
      2. Applies those statistics to z-score the full input dataset
      3. Computes |∂output/∂input| for every trial in batches
      4. Separates results by label (high-cloze y=1, low-cloze y=0)

    All fold maps are averaged to produce a stable final map.

    Parameters
    ----------
    run_dir    : Directory containing fold{k}_best.pt checkpoints
    X_raw      : Raw (unscaled) EEG, shape (n_trials, n_channels, T)
    y          : Binary labels, shape (n_trials,)
    model_name : 'cnn', 'cnn_v2', or 'eegnet'
    sfreq      : Sampling frequency (Hz)
    device     : 'cpu' or 'cuda'
    batch_size : Trials per forward pass (reduce if OOM)

    Returns
    -------
    dict with:
        saliency_mean  : (n_channels, T) mean |gradient| across all trials + folds
        saliency_high  : (n_channels, T) mean |gradient| for high-cloze trials
        saliency_low   : (n_channels, T) mean |gradient| for low-cloze trials
        saliency_diff  : (n_channels, T) saliency_high − saliency_low
        n_folds        : number of fold checkpoints averaged
        n_trials_high  : number of high-cloze trials
        n_trials_low   : number of low-cloze trials
    """
    import torch

    try:
        from src.trainer import build_model
    except ImportError:
        raise ImportError(
            "Could not import build_model from src.trainer. "
            "Ensure PROJECT_ROOT is in sys.path before calling this function."
        )

    run_dir = Path(run_dir)
    ckpt_paths = sorted(run_dir.glob("fold*_best.pt"))

    if not ckpt_paths:
        raise FileNotFoundError(f"No fold checkpoints (fold*_best.pt) found in {run_dir}")

    T = X_raw.shape[2]
    y = np.asarray(y)

    fold_sal_all  = []
    fold_sal_high = []
    fold_sal_low  = []

    for ckpt_path in ckpt_paths:
        state_dict, extras = _load_fold_checkpoint(ckpt_path, device=device)

        # Z-score using this fold's training statistics
        zscore_mean = extras.get("zscore_mean")
        zscore_std  = extras.get("zscore_std")

        if zscore_mean is not None and zscore_std is not None:
            zscore_mean = np.asarray(zscore_mean)   # (1, 32, 1)
            zscore_std  = np.asarray(zscore_std)
        else:
            print(f"  Warning: z-score stats missing from {ckpt_path.name}; "
                  "falling back to dataset-level stats.")
            zscore_mean = X_raw.mean(axis=(0, 2), keepdims=True)
            zscore_std  = X_raw.std(axis=(0, 2),  keepdims=True)

        X_z = (X_raw - zscore_mean) / (zscore_std + 1e-6)

        # Rebuild model and load weights
        model = build_model(
            model_name=model_name,
            sfreq=sfreq,
            num_timepoints=T,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # Gradient computation in batches
        grad_chunks = []

        for start in range(0, len(X_z), batch_size):
            batch_np = X_z[start : start + batch_size]

            x = torch.tensor(
                batch_np[:, np.newaxis, :, :],   # (B, 1, n_channels, T)
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )

            out = model(x)        # (B, 1)
            out.sum().backward()  # gradient of sum = gradient per-trial

            grad_chunks.append(
                x.grad.detach().cpu().numpy()[:, 0, :, :]  # (B, n_channels, T)
            )

        grads = np.concatenate(grad_chunks, axis=0)   # (N, n_channels, T)
        abs_grads = np.abs(grads)

        fold_sal_all.append(abs_grads.mean(axis=0))
        fold_sal_high.append(abs_grads[y == 1].mean(axis=0) if (y == 1).any() else np.zeros_like(abs_grads[0]))
        fold_sal_low.append( abs_grads[y == 0].mean(axis=0) if (y == 0).any() else np.zeros_like(abs_grads[0]))

    saliency_mean = np.stack(fold_sal_all).mean(axis=0)
    saliency_high = np.stack(fold_sal_high).mean(axis=0)
    saliency_low  = np.stack(fold_sal_low).mean(axis=0)

    return {
        "saliency_mean":  saliency_mean,
        "saliency_high":  saliency_high,
        "saliency_low":   saliency_low,
        "saliency_diff":  saliency_high - saliency_low,
        "n_folds":        len(ckpt_paths),
        "n_trials_high":  int((y == 1).sum()),
        "n_trials_low":   int((y == 0).sum()),
    }


def _saliency_times(T: int, window_ms: tuple[int, int]) -> np.ndarray:
    return np.linspace(window_ms[0], window_ms[1], T)


def _saliency_time_index(t_ms: int, T: int, window_ms: tuple[int, int]) -> int:
    times = _saliency_times(T, window_ms)
    return int(np.argmin(np.abs(times - t_ms)))


def plot_saliency_single(
    saliency: dict,
    model_display: str,
    channel_names: list[str] | None = None,
    window_ms: tuple[int, int] = (0, 800),
    peak_ms: int = 400,
    mne_info=None,
    key: str = "saliency_mean",
    save_path: Path | None = None,
) -> tuple:
    """
    Heatmap (channels × time) + optional scalp topomap at peak_ms for one model.

    Parameters
    ----------
    saliency      : Output of compute_gradient_saliency()
    model_display : Label for plot title (e.g. 'CNN-v2')
    channel_names : List of 32 channel name strings. Defaults to DERCO_CHANNEL_NAMES.
    window_ms     : (start_ms, end_ms) of the epoch window
    peak_ms       : Timepoint for the topomap (ms). Ignored if mne_info is None.
    mne_info      : mne.Info object for topomap (load from any DERCo FIF).
                    If None, only the heatmap is shown.
    key           : Which saliency map to plot. One of:
                    'saliency_mean', 'saliency_high', 'saliency_low', 'saliency_diff'
    save_path     : Optional path to save the figure.
    """
    if channel_names is None:
        channel_names = DERCO_CHANNEL_NAMES

    sal_map = saliency[key]   # (n_channels, T)
    T = sal_map.shape[1]
    times = _saliency_times(T, window_ms)

    n_panels = 2 if mne_info is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ── Heatmap ───────────────────────────────────────────────────────────────
    ax_heat = axes[0]
    im = ax_heat.imshow(
        sal_map,
        aspect="auto",
        origin="upper",
        extent=[times[0], times[-1], sal_map.shape[0] - 0.5, -0.5],
        cmap="hot",
    )
    ax_heat.axvspan(300, 500, alpha=0.15, color="cyan", label="N400 window")
    ax_heat.set_xlabel("Time (ms)")
    ax_heat.set_ylabel("Channel")
    ax_heat.set_yticks(range(len(channel_names)))
    ax_heat.set_yticklabels(channel_names, fontsize=6)
    ax_heat.set_title(f"{model_display} — {key.replace('_', ' ')} saliency")
    plt.colorbar(im, ax=ax_heat, label="|gradient|")
    ax_heat.legend(fontsize=8)

    # ── Topomap ───────────────────────────────────────────────────────────────
    if mne_info is not None:
        import mne as _mne

        ax_topo = axes[1]
        t_idx = _saliency_time_index(peak_ms, T, window_ms)
        topo_data = sal_map[:, t_idx]

        _mne.viz.plot_topomap(
            data=topo_data,
            pos=mne_info,
            ch_type="eeg",
            axes=ax_topo,
            show=False,
            cmap="hot",
            sphere="eeg",
            image_interp="cubic",
            sensors="k.",
        )
        ax_topo.set_title(f"Scalp saliency @ {peak_ms} ms")

    fig.suptitle(f"Gradient saliency — {model_display}", fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_saliency_comparison(
    saliency_a: dict,
    saliency_b: dict,
    label_a: str,
    label_b: str,
    channel_names: list[str] | None = None,
    window_ms: tuple[int, int] = (0, 800),
    peak_ms: int = 400,
    mne_info=None,
    key: str = "saliency_mean",
    save_path: Path | None = None,
) -> tuple:
    """
    Side-by-side gradient saliency comparison for two models.

    Layout with mne_info:   2×2 grid (heatmap + topomap per model)
    Layout without mne_info: 1×2 grid (heatmaps only)

    Parameters
    ----------
    saliency_a, saliency_b : Outputs of compute_gradient_saliency()
    label_a, label_b       : Display names (e.g. 'CNN-v2', 'EEGNet')
    channel_names          : Channel name list. Defaults to DERCO_CHANNEL_NAMES.
    window_ms              : (start_ms, end_ms) of the epoch window
    peak_ms                : Timepoint for topomaps (ms)
    mne_info               : mne.Info for topomaps. None → heatmaps only.
    key                    : Which saliency map to compare.
    save_path              : Optional save path.
    """
    if channel_names is None:
        channel_names = DERCO_CHANNEL_NAMES

    maps = [saliency_a[key], saliency_b[key]]
    labels = [label_a, label_b]

    # Shared colour scale across both models
    vmax = max(m.max() for m in maps)

    if mne_info is not None:
        import mne as _mne
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        heat_axes = axes[0]
        topo_axes = axes[1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        heat_axes = axes
        topo_axes = [None, None]

    for i, (sal_map, lbl) in enumerate(zip(maps, labels)):
        T = sal_map.shape[1]
        times = _saliency_times(T, window_ms)

        # ── Heatmap ───────────────────────────────────────────────────────────
        ax_h = heat_axes[i]
        im = ax_h.imshow(
            sal_map,
            aspect="auto",
            origin="upper",
            extent=[times[0], times[-1], sal_map.shape[0] - 0.5, -0.5],
            cmap="hot",
            vmin=0,
            vmax=vmax,
        )
        ax_h.axvspan(300, 500, alpha=0.15, color="cyan", label="N400 window")
        ax_h.set_xlabel("Time (ms)")
        ax_h.set_title(f"{lbl}")
        if i == 0:
            ax_h.set_ylabel("Channel")
            ax_h.set_yticks(range(len(channel_names)))
            ax_h.set_yticklabels(channel_names, fontsize=6)
        else:
            ax_h.set_yticks([])
        plt.colorbar(im, ax=ax_h, label="|gradient|")
        ax_h.legend(fontsize=8)

        # ── Topomap ───────────────────────────────────────────────────────────
        if topo_axes[i] is not None and mne_info is not None:
            t_idx = _saliency_time_index(peak_ms, T, window_ms)
            _mne.viz.plot_topomap(
                data=sal_map[:, t_idx],
                pos=mne_info,
                ch_type="eeg",
                axes=topo_axes[i],
                show=False,
                cmap="hot",
                sphere="eeg",
                image_interp="cubic",
                sensors="k.",
            )
            topo_axes[i].set_title(f"{lbl} @ {peak_ms} ms")

    fig.suptitle(
        f"Gradient saliency comparison — {key.replace('_', ' ')}",
        fontsize=13,
    )
    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes
