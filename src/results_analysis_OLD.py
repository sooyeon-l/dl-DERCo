from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


DERCO_ROOT = Path("/content/drive/MyDrive/Colab_Notebooks/DERCo")
RUNS_ROOT = DERCO_ROOT / "outputs" / "runs"
ANALYSIS_ROOT = DERCO_ROOT / "analysis"
TABLES_DIR = ANALYSIS_ROOT / "tables"
FIGURES_DIR = ANALYSIS_ROOT / "figures"


def ensure_analysis_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


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
        threshold = float(run_summary.get("global_threshold", 0.5))
        row.update(compute_oof_metrics(oof_path, threshold=threshold))

    return row


def collect_all_runs(runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    rows = []

    for run_summary_path in runs_root.rglob("run_summary.json"):
        run_dir = run_summary_path.parent
        row = collect_run(run_dir)

        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No run_summary.json files found under {runs_root}")

    df = pd.DataFrame(rows)
    return add_run_category(df)


def add_run_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["category"] = "unknown"

    df.loc[df["run_dir"].str.contains("/baseline/"), "category"] = "baseline"
    df.loc[df["run_dir"].str.contains("/subject_ablation/"), "category"] = "subject_ablation"

    return df


def make_baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["category"] == "baseline"].copy()

    if baseline.empty:
        return baseline

    cols = [
        "run_name",
        "model",
        "window",
        "sfreq",
        "shuffle_labels",
        "oof_auc",
        "oof_accuracy",
        "oof_balanced_accuracy",
        "oof_f1",
        "cv_balanced_accuracy_at_global_threshold",
        "global_threshold",
        "n_oof_samples",
        "X_train_val_shape",
        "x_path",
        "checkpoint_dir",
        "run_dir",
    ]

    existing_cols = [c for c in cols if c in baseline.columns]

    return (
        baseline[existing_cols]
        .sort_values(
            by=[c for c in ["sfreq", "model", "window", "shuffle_labels"] if c in existing_cols],
            ascending=True,
        )
        .reset_index(drop=True)
    )


def make_subject_ablation_runs(df: pd.DataFrame) -> pd.DataFrame:
    ablation = df[df["category"] == "subject_ablation"].copy()

    if ablation.empty:
        return ablation

    if "n_selected_subjects" in ablation.columns:
        ablation["subject_count"] = ablation["n_selected_subjects"]
    else:
        ablation["subject_count"] = ablation["n_subjects_requested"]

    cols = [
        "run_name",
        "model",
        "subject_count",
        "n_subjects_requested",
        "ablation_seed",
        "selected_subjects",
        "n_folds",
        "oof_auc",
        "oof_accuracy",
        "oof_balanced_accuracy",
        "oof_f1",
        "global_threshold",
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
        .groupby(["model", "subject_count"], as_index=False)
        .agg(
            mean_auc=("oof_auc", "mean"),
            std_auc=("oof_auc", safe_std),
            sem_auc=("oof_auc", safe_sem),

            mean_bal_acc=("oof_balanced_accuracy", "mean"),
            std_bal_acc=("oof_balanced_accuracy", safe_std),
            sem_bal_acc=("oof_balanced_accuracy", safe_sem),

            n_runs=("run_name", "count"),
            mean_n_oof_samples=("n_oof_samples", "mean"),
        )
        .sort_values(["model", "subject_count"])
        .reset_index(drop=True)
    )

    full_pool_n = summary["subject_count"].max()
    full_pool_mask = summary["subject_count"] == full_pool_n

    summary.loc[
        full_pool_mask,
        ["std_auc", "sem_auc", "std_bal_acc", "sem_bal_acc"]
    ] = np.nan

    return summary


def save_tables(
    all_runs: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    ablation_runs: pd.DataFrame,
    ablation_summary: pd.DataFrame,
):
    ensure_analysis_dirs()

    all_runs.to_csv(TABLES_DIR / "all_runs_summary.csv", index=False)
    baseline_summary.to_csv(TABLES_DIR / "baseline_oof_summary.csv", index=False)
    ablation_runs.to_csv(TABLES_DIR / "subject_ablation_runs.csv", index=False)
    ablation_summary.to_csv(TABLES_DIR / "subject_ablation_summary.csv", index=False)


def plot_subject_ablation_auc(
    summary: pd.DataFrame,
    permutation_auc: float | None = 0.5315441095218898,
    save_path: Path | None = None,
):
    if summary.empty:
        raise ValueError("Subject ablation summary is empty.")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, src in summary.groupby("model"):
        src = src.sort_values("subject_count")

        ax.errorbar(
            src["subject_count"],
            src["mean_auc"],
            yerr=src["sem_auc"],
            marker="o",
            capsize=3,
            label=model.upper(),
        )

    if permutation_auc is not None:
        ax.axhline(
            permutation_auc,
            linestyle="--",
            linewidth=1,
            label=f"Shuffled-label baseline AUC = {permutation_auc:.3f}",
        )

    ax.set_xlabel("Number of subjects included in CV pool")
    ax.set_ylabel("Out-of-fold ROC AUC")
    ax.set_title("Subject-count ablation: model sample efficiency")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        ensure_analysis_dirs()
        fig.savefig(save_path, dpi=300)

    return fig, ax


def plot_subject_ablation_bal_acc(
    summary: pd.DataFrame,
    save_path: Path | None = None,
):
    if summary.empty:
        raise ValueError("Subject ablation summary is empty.")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, src in summary.groupby("model"):
        src = src.sort_values("subject_count")

        ax.errorbar(
            src["subject_count"],
            src["mean_bal_acc"],
            yerr=src["sem_bal_acc"],
            marker="o",
            capsize=3,
            label=model.upper(),
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