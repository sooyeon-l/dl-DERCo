import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Threshold search
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold_bal_acc(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Grid-search the probability threshold that maximises balanced accuracy.
    Used during cross-validation to derive a validation-calibrated threshold.
    NOT to be applied to the held-out test set.
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_threshold = 0.5
    best_score = -float("inf")

    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-aware model reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _build_legacy_cnn(
    kernel_len: int,
    num_channels: int,
    dropout_p: dict,
) -> nn.Module:
    """
    Recreate the original CNNModel with AdaptiveAvgPool2d and bias=True on
    conv layers. Used to load checkpoints trained before the temporal-collapse
    fix was applied.
    """
    padding = (kernel_len - 1) // 2

    class _LegacyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal_block = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(1, kernel_len),
                          padding=(0, padding)),           # bias=True (default)
                nn.BatchNorm2d(8),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout2d(dropout_p["conv"]),
            )
            self.spatial_block = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=(num_channels, 1)),  # bias=True
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Dropout2d(dropout_p["conv"]),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 32),
                nn.ELU(),
                nn.Dropout(dropout_p["classifier"]),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            x = self.temporal_block(x)
            x = self.spatial_block(x)
            return self.classifier(x)

    return _LegacyCNN()


def rebuild_model_from_state_dict(
    state_dict: dict,
    model_name: str,
    sfreq: int,
    num_timepoints: int,
) -> nn.Module:
    """
    Reconstruct a model whose architecture exactly matches the given state dict.

    Handles:
    - Legacy CNNModel (AdaptiveAvgPool, bias=True on convs)
    - Current CNNModel (no AdaptiveAvgPool, bias=False)
    - CNNV2Model (two temporal convs, doubled filters)
    - EEGNetModel

    Architecture is inferred from state dict key names and weight shapes —
    no manual flags needed.
    """
    try:
        import src.config as config
        from src.trainer import build_model
    except ImportError:
        raise ImportError(
            "Could not import from src. Ensure PROJECT_ROOT is in sys.path."
        )

    if model_name == "eegnet":
        return build_model(model_name, sfreq, num_timepoints)

    # ── Detect architecture from state dict ───────────────────────────────
    has_adaptive_pool   = "classifier.2.weight" in state_dict
    has_second_temporal = "temporal_block.3.weight" in state_dict
    has_conv_bias       = "temporal_block.0.bias" in state_dict

    kernel_len   = state_dict["temporal_block.0.weight"].shape[-1]
    num_channels = state_dict["spatial_block.0.weight"].shape[2]

    if has_adaptive_pool:
        # Original CNNModel trained before the temporal-collapse fix
        print(f"  Detected legacy CNN architecture "
              f"(AdaptiveAvgPool, bias=True, kernel={kernel_len})")
        return _build_legacy_cnn(kernel_len, num_channels, config.DROPOUT_P)

    if has_second_temporal:
        # CNNV2Model
        sep_kernel_len = state_dict["temporal_block.3.weight"].shape[-1]
        print(f"  Detected CNN-v2 architecture "
              f"(kernel={kernel_len}, sep_kernel={sep_kernel_len})")
        from src.models.cnn import CNNV2Model
        return CNNV2Model(
            dropout_p=config.DROPOUT_P,
            num_timepoints=num_timepoints,
            kernel_len=kernel_len,
            sep_kernel_len=sep_kernel_len,
            num_channels=num_channels,
        )

    # Current CNNModel (no AdaptiveAvgPool, bias=False)
    print(f"  Detected current CNN architecture "
          f"(no AdaptiveAvgPool, kernel={kernel_len})")
    return build_model("cnn", sfreq, num_timepoints)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_proba(
    model: nn.Module,
    X_z: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run inference on z-scored input and return predicted probabilities.

    Parameters
    ----------
    model     : Trained model in eval mode
    X_z       : Z-scored EEG, shape (n_trials, n_channels, T)
    device    : 'cpu' or 'cuda'
    batch_size: Trials per forward pass

    Returns
    -------
    probs : (n_trials,) predicted probabilities
    """
    model.eval()

    X_tensor = torch.tensor(
        X_z[:, np.newaxis, :, :],   # (N, 1, n_channels, T)
        dtype=torch.float32,
    )
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_probs = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs).reshape(-1)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """ROC AUC, PR AUC, and balanced accuracy at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":           float(roc_auc_score(y_true, y_prob)),
        "pr_auc":            float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-sample test evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_cv_ensemble(
    checkpoint_dir: str | Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    sfreq: int = 250,
    global_threshold: float | None = None,
    output_dir: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """
    Evaluate a CV-trained model ensemble on a completely held-out test set.

    For each fold checkpoint:
      1. Loads the checkpoint and auto-detects its architecture from the
         state dict (handles both legacy and current CNN variants)
      2. Z-scores X_test with that fold's training mean/std
      3. Predicts probabilities on the test set

    Probabilities are averaged across folds before computing metrics.
    The validation-derived global_threshold is loaded from output_dir/
    run_summary.json when not supplied explicitly.

    Parameters
    ----------
    checkpoint_dir   : Directory containing fold*_best.pt checkpoints
    X_test           : Raw (unscaled) test EEG, shape (n_trials, n_channels, T)
    y_test           : Binary labels, shape (n_trials,)
    model_name       : 'cnn', 'cnn_v2', or 'eegnet'
    sfreq            : Sampling frequency (Hz)
    global_threshold : Validation-derived threshold. Auto-loaded if None.
    output_dir       : Run output directory containing run_summary.json.
    device           : 'cpu' or 'cuda'
    batch_size       : Trials per forward pass

    Returns
    -------
    dict with test_roc_auc, test_pr_auc, test_bal_acc_05,
    test_bal_acc_val_threshold, val_threshold, test_probs,
    n_folds, n_test_samples
    """
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_paths = sorted(checkpoint_dir.glob("fold*_best.pt"))

    if not ckpt_paths:
        raise FileNotFoundError(
            f"No fold checkpoints (fold*_best.pt) found in {checkpoint_dir}"
        )

    # ── Resolve validation-derived threshold ──────────────────────────────
    if global_threshold is None:
        if output_dir is not None:
            summary_path = Path(output_dir) / "run_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    run_summary = json.load(f)
                global_threshold = float(
                    run_summary.get("global_threshold", 0.5)
                )
                print(f"  global_threshold={global_threshold:.4f} "
                      f"(from run_summary.json)")
            else:
                print(f"  Warning: run_summary.json not found. "
                      "Using threshold=0.5.")
                global_threshold = 0.5
        else:
            print("  No output_dir supplied. Using threshold=0.5.")
            global_threshold = 0.5

    T      = X_test.shape[2]
    y_test = np.asarray(y_test).reshape(-1)
    fold_probs = []

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # ── Extract state dict ─────────────────────────────────────────────
        state_dict = None
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                state_dict = ckpt[key]
                break
        if state_dict is None:
            state_dict = ckpt

        # ── Rebuild model matching this checkpoint's architecture ──────────
        model = rebuild_model_from_state_dict(
            state_dict=state_dict,
            model_name=model_name,
            sfreq=sfreq,
            num_timepoints=T,
        ).to(device)
        model.load_state_dict(state_dict)

        # ── Z-score with this fold's training statistics ───────────────────
        zscore_mean = ckpt.get("zscore_mean")
        zscore_std  = ckpt.get("zscore_std")

        if zscore_mean is not None and zscore_std is not None:
            zscore_mean = np.asarray(zscore_mean)
            zscore_std  = np.asarray(zscore_std)
        else:
            raise ValueError(
                f"Missing zscore_mean/zscore_std in {ckpt_path.name}. "
                "Held-out evaluation requires fold-specific training z-score statistics."
            )

        X_z = (X_test - zscore_mean) / (zscore_std + 1e-6)

        probs = predict_proba(model, X_z, device, batch_size)
        fold_probs.append(probs)
        print(f"  {ckpt_path.name}: AUC = {roc_auc_score(y_test, probs):.4f}")

    # ── Ensemble: average probabilities across folds ───────────────────────
    ensemble_probs = np.stack(fold_probs).mean(axis=0)

    metrics_05  = compute_binary_metrics(y_test, ensemble_probs, threshold=0.5)
    metrics_val = compute_binary_metrics(
        y_test, ensemble_probs, threshold=global_threshold
    )

    return {
        "test_roc_auc":                metrics_05["roc_auc"],
        "test_pr_auc":                 metrics_05["pr_auc"],
        "test_bal_acc_05":             metrics_05["balanced_accuracy"],
        "test_bal_acc_val_threshold":  metrics_val["balanced_accuracy"],
        "val_threshold":               global_threshold,
        "test_probs":                  ensemble_probs,
        "n_folds":                     len(ckpt_paths),
        "n_test_samples":              int(len(y_test)),
    }
