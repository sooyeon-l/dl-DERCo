import torch
import copy
import json
import torch.nn as nn
import numpy as np
import pandas as pd
import src.config as config
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from pathlib import Path
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from src.dataset import EEGDataset
from src.models.cnn import CNNModel, CNNV2Model
from src.models.eegnet import EEGNetModel
from src.eval import find_best_threshold_bal_acc

def make_config_snapshot(
    run_name:str,
    model_name:str,
    window:str,
    sfreq:int,
    shuffle_labels:bool,
    device:str,
    train_val_path:Path,
    checkpoint_dir:Path,
    X_train_val,
    y_train_val,
    subjects_train_val,
    unique_subjects,
    all_unique_subjects=None,
    selected_subjects=None,
    n_subjects=None,
    ablation_seed=None,
    n_folds=None,
):
    snapshot = {
        # Run identity
        "run_name": run_name,
        "model_name": model_name,
        "window": window,
        "sfreq": sfreq,
        "shuffle_labels": shuffle_labels,
        "device": str(device),

        # Data
        "train_val_path": str(train_val_path),
        "X_train_val_shape": str(tuple(X_train_val.shape)),
        "y_train_val_shape": str(tuple(y_train_val.shape)),
        "subjects_train_val_shape": str(tuple(subjects_train_val.shape)),
        "n_subjects_requested": None if n_subjects is None else int(n_subjects),
        "ablation_seed": None if ablation_seed is None else int(ablation_seed),
        "n_all_unique_subjects": (None if all_unique_subjects is None else int(len(all_unique_subjects))),
        "n_selected_subjects": None if selected_subjects is None else int(len(selected_subjects)),
        "selected_subjects": None if selected_subjects is None else [str(s) for s in selected_subjects],
        "n_unique_subjects": int(len(unique_subjects)),
        "n_folds": int(n_folds) if n_folds is not None else config.N_FOLDS,

        # Training hyperparameters
        "random_seed": config.RANDOM_SEED,
        "batch_size": config.BATCH_SIZE,
        "max_epochs": config.MAX_EPOCHS,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "learning_rate": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "scheduler_factor": config.SCHEDULER_FACTOR,
        "scheduler_patience": config.SCHEDULER_PATIENCE,
        "grad_clip": config.GRAD_CLIP,
        "threshold": config.THRESHOLD,

        # Paths
        "checkpoint_dir": str(checkpoint_dir),
    }

    if model_name == "cnn":
        cnn_cfg = config.CNN_CONFIGS[sfreq]
        snapshot.update({
            "cnn_kernel_len": cnn_cfg.get("kernel_len"),
            "cnn_padding": cnn_cfg.get("padding"),
            "cnn_dropout": str(config.DROPOUT_P),
            "num_channels": config.NUM_CHANNELS,
        })

    elif model_name == "eegnet":
        eeg_cfg = config.EEGNET_CONFIGS[sfreq]
        snapshot.update({
            "eegnet_F1": eeg_cfg.get("F1"),
            "eegnet_D": eeg_cfg.get("D"),
            "eegnet_F2": eeg_cfg.get("F2"),
            "eegnet_kernel_len": eeg_cfg.get("kernel_len"),
            "eegnet_kernel_padding": eeg_cfg.get("kernel_padding"),
            "eegnet_sep_kernel_len": eeg_cfg.get("sep_kernel_len"),
            "eegnet_sep_padding": eeg_cfg.get("sep_padding"),
            "eegnet_dropout": eeg_cfg.get("dropout"),
            "num_channels": config.NUM_CHANNELS,
            "num_timepoints": config.N_TIMEPOINTS[sfreq],
        })

    return snapshot

def get_class_balance(
        y, 
        run_name:str,
        split_name:str, 
        model_name:str, 
        window:str, 
        shuffle_labels:bool=False, 
        sfreq:int=250, 
        fold_idx:int=None
):
    y = np.asarray(y).reshape(-1)
    n_low = int((y == 0).sum())
    n_high = int((y == 1).sum())
    total = int(len(y))

    return {
        "run_name": run_name,
        "model": model_name, 
        "window": window, 
        "sfreq": sfreq,
        "shuffle_labels": shuffle_labels, 
        "fold": fold_idx,
        "split": split_name,
        "total": total,
        "n_low": n_low,
        "n_high": n_high,
        "prop_low": n_low / total,
        "prop_high": n_high / total,
    }

def train_one_epoch(
        model, 
        train_loader:DataLoader, 
        criterion, 
        optimizer, 
        device:str, 
        grad_clip=None, 
        epoch=None
):
    batch_losses = []
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
       
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        batch_losses.append({
            'epoch': epoch, 
            'batch': batch_idx, 
            'train_loss': loss.item(), 
            'batch_size': batch_size
        })
    avg_train_loss = total_loss / n_samples

    return avg_train_loss, batch_losses

def evaluate_model(
        model, 
        val_loader:DataLoader, 
        criterion, 
        device:str, 
        threshold=0.5): 
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_samples = 0

    model.eval()

    with torch.no_grad(): 
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
    
    val_loss = total_loss / n_samples

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probabilities = torch.sigmoid(all_logits)

    probabilities_np = probabilities.numpy().reshape(-1)
    labels_np = all_labels.numpy().reshape(-1)

    if len(np.unique(labels_np)) < 2: 
        val_roc_auc = np.nan
        val_pr_auc = np.nan
    else: 
        val_roc_auc = roc_auc_score(labels_np, probabilities_np)
        val_pr_auc = average_precision_score(labels_np, probabilities_np)
    preds_np = (probabilities_np >= threshold).astype(int)
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)

    return {
        "val_loss": val_loss,
        "val_roc_auc": val_roc_auc,
        "val_pr_auc": val_pr_auc,
        "val_balanced_accuracy": balanced_acc,
        "probabilities": probabilities_np,
        "labels": labels_np,
    }

def train_model(
    model,
    optimizer,
    scheduler,
    train_loader:DataLoader,
    val_loader:DataLoader,
    max_epochs:int,
    early_stopping_patience,
    device:str,
    grad_clip=1.0, 
    threshold=0.5, 
    progress_desc:str="Training", 
    checkpoint_path:Path=None,
    checkpoint_extra:dict=None,
):
    best_val_roc_auc = -float('inf')
    epochs_without_improvement = 0
    criterion = nn.BCEWithLogitsLoss()

    epoch_history = []
    batch_history_all = []
    best_info = {}

    for epoch in range(max_epochs): 
        avg_train_loss, batch_history = train_one_epoch(
            model=model, 
            train_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            grad_clip=grad_clip, 
            epoch=epoch,
        )

        batch_history_all.extend(batch_history)
        
        metrics = evaluate_model(
            model=model, 
            val_loader=val_loader, 
            criterion=criterion, 
            device=device, 
            threshold=threshold)
        if not np.isnan(metrics['val_roc_auc']): 
            scheduler.step(metrics['val_roc_auc'])
        
        epoch_history.append({
            'epoch': epoch,
            'avg_train_loss': avg_train_loss,
            'val_loss': metrics['val_loss'],
            'val_roc_auc': metrics['val_roc_auc'],
            'val_pr_auc': metrics['val_pr_auc'],
            'val_balanced_accuracy': metrics['val_balanced_accuracy'],
            'lr': optimizer.param_groups[0]["lr"],
            'threshold': threshold,
        })

        if not np.isnan(metrics['val_roc_auc']) and metrics['val_roc_auc'] > best_val_roc_auc: 
            best_val_roc_auc = metrics['val_roc_auc']
            best_info = {
                'epoch': epoch, 
                'model_state_dict': copy.deepcopy(model.state_dict()), 
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()), 
                'best_val_roc_auc': float(best_val_roc_auc),
                'best_val_pr_auc': float(metrics['val_pr_auc']),
                'best_val_loss': float(metrics['val_loss']),
                'best_val_balanced_accuracy': float(metrics['val_balanced_accuracy']),

                # For out-of-fold threshold selection
                'best_val_probabilities': metrics['probabilities'].copy(),
                'best_val_labels': metrics['labels'].copy(),
            }

            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save({**best_info, **(checkpoint_extra or {})}, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            print(
                f"New best: epoch {epoch} | "
                f"ROC AUC={best_val_roc_auc:.4f} | "
                f"PR AUC={metrics['val_pr_auc']:.4f} | "
                f"BalAcc={metrics['val_balanced_accuracy']:.4f}"
            )
            should_log = (
                epoch == 0
                or (epoch + 1) % config.LOG_EVERY == 0
                or epoch == max_epochs - 1
            )

            if should_log:
                print(
                    f"{progress_desc} | "
                    f"epoch {epoch + 1}/{max_epochs} | "
                    f"train_loss={avg_train_loss:.4f} | "
                    f"val_loss={metrics['val_loss']:.4f} | "
                    f"roc={metrics['val_roc_auc']:.4f} | "
                    f"pr={metrics['val_pr_auc']:.4f} | "
                    f"bal={metrics['val_balanced_accuracy']:.4f} | "
                    f"best={best_val_roc_auc:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                    f"pat={epochs_without_improvement}",
                    flush=True,
                )

            epochs_without_improvement = 0
        else: 
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience: 
                print("Early stopping triggered.")
                break
        
        
    return epoch_history, batch_history_all, best_info

def build_model(model_name: str, sfreq: int, num_timepoints: int):
    if model_name == "cnn":
        cnn_cfg = config.CNN_CONFIGS[sfreq]
        return CNNModel(
            dropout_p=config.DROPOUT_P,
            num_timepoints=num_timepoints,
            kernel_len=cnn_cfg["kernel_len"],
            num_channels=config.NUM_CHANNELS,
        )

    if model_name == "cnn_v2":
        cnn_cfg = config.CNN_CONFIGS[sfreq]
        return CNNV2Model(
            dropout_p=config.DROPOUT_P,
            num_timepoints=num_timepoints,
            kernel_len=cnn_cfg["kernel_len"],
            sep_kernel_len=cnn_cfg["sep_kernel_len"],
            num_channels=config.NUM_CHANNELS,
        )

    if model_name == "eegnet":
        eeg_cfg = config.EEGNET_CONFIGS[sfreq]
        return EEGNetModel(
            dropout_p=eeg_cfg["dropout"],
            kernel_len=eeg_cfg["kernel_len"],
            sep_kernel_len=eeg_cfg["sep_kernel_len"],
            num_channels=config.NUM_CHANNELS,
            num_timepoints=config.N_TIMEPOINTS[sfreq],
            F1=eeg_cfg["F1"],
            D=eeg_cfg["D"],
        )

    raise ValueError(f"Unknown model_name: {model_name}")

def select_subject_subset(unique_subjects, n_subjects=None, ablation_seed=None):
    unique_subjects = np.asarray(unique_subjects)

    if n_subjects is None:
        return unique_subjects

    if n_subjects < 2:
        raise ValueError(
            f"n_subjects must be at least 2 for subject-disjoint CV, got {n_subjects}."
        )

    if n_subjects > len(unique_subjects):
        raise ValueError(
            f"Requested n_subjects={n_subjects}, but only "
            f"{len(unique_subjects)} unique subjects are available."
        )

    if ablation_seed is None:
        raise ValueError(
            "ablation_seed must be provided when n_subjects is used, "
            "so subject subset selection is explicit and reproducible."
        )

    rng = np.random.default_rng(ablation_seed)

    selected_subjects = rng.choice(
        unique_subjects,
        size=n_subjects,
        replace=False,
    )

    return np.sort(selected_subjects)

def run_experiment(
        model_name:str, 
        window:str, 
        train_val_path:Path, 
        device:str='cuda', 
        sfreq:int=250, 
        shuffle_labels:bool=False, 
        mode:str='max', 
        run_name:str=None, 
        checkpoint_dir:Path=None,
        output_dir:Path=None,
        n_subjects:int=None,
        ablation_seed:int=None,
): 
    if model_name == "eegnet" and window not in config.EEGNET_WINDOWS:
        raise ValueError(f"EEGNet is only configured for {config.EEGNET_WINDOWS}, got {window}")

    if model_name in ["cnn", "cnn_v2"] and window not in config.CNN_WINDOWS:
        raise ValueError(f"CNN window must be one of {config.CNN_WINDOWS}, got {window}")
    
    data_dir = train_val_path / f"{sfreq}hz"
    x_path = data_dir / f"X_{window}.npy"
    y_path = data_dir / "y.npy"
    subjects_path = data_dir / "subjects.npy"

    X_train_val = np.load(x_path)
    y_train_val = np.load(y_path)
    subjects_train_val = np.load(subjects_path)

    expected_timepoints = config.WINDOWS[sfreq][window].stop - config.WINDOWS[sfreq][window].start
    actual_timepoints = X_train_val.shape[-1]

    if actual_timepoints != expected_timepoints:
        raise ValueError(
            f"Sampling-rate/window mismatch: sfreq={sfreq}, window={window} expects "
            f"{expected_timepoints} timepoints, but loaded X has {actual_timepoints}. "
            f"Loaded X from: {x_path}"
        )

    seed = config.RANDOM_SEED
    n_folds = config.N_FOLDS

    if run_name is None:
        label_mode = "shuffle" if shuffle_labels else "main"

        if n_subjects is None:
            run_name = f"{model_name}_{window}_{sfreq}hz_{label_mode}"
        else:
            run_name = (
                f"{model_name}_{window}_{sfreq}hz_{label_mode}_"
                f"nsubj{n_subjects:02d}_seed{ablation_seed:03d}"
            )
    if checkpoint_dir is None:
        checkpoint_dir = config.CHECKPOINTS_PATH / run_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # out-of-fold probabilties and labels
    oof_probs_all = []
    oof_labels_all = []
    
    # other records
    epoch_history_all = []
    batch_history_all = []
    class_balance_records = []
    best_summary_records = []

    all_unique_subjects = np.unique(subjects_train_val)

    selected_subjects = select_subject_subset(
        unique_subjects=all_unique_subjects, 
        n_subjects=n_subjects, 
        ablation_seed=ablation_seed,
    )

    subject_pool_mask = np.isin(subjects_train_val, selected_subjects)

    X_train_val = X_train_val[subject_pool_mask]
    y_train_val = y_train_val[subject_pool_mask]
    subjects_train_val = subjects_train_val[subject_pool_mask]

    unique_subjects = np.unique(subjects_train_val)

    n_folds = min(config.N_FOLDS, len(unique_subjects))

    if n_folds < 2: 
        raise ValueError(
            f"Need at least 2 subjects for subject-disjoint CV, got {len(unique_subjects)}."
        )


    config_snapshot = make_config_snapshot(
        run_name=run_name,
        model_name=model_name,
        window=window,
        sfreq=sfreq,
        shuffle_labels=shuffle_labels,
        device=device,
        train_val_path=train_val_path,
        checkpoint_dir=checkpoint_dir,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        subjects_train_val=subjects_train_val,
        all_unique_subjects=all_unique_subjects,
        unique_subjects=unique_subjects,
        selected_subjects=selected_subjects,
        n_subjects=n_subjects, 
        ablation_seed=ablation_seed,
        n_folds=n_folds,
    )

    if output_dir is not None:
        pd.DataFrame([config_snapshot]).to_csv(
            output_dir / "config_snapshot.csv",
            index=False,
        )
        with open(output_dir / "config_snapshot.json", "w") as f:
            json.dump(config_snapshot, f, indent=2)
    
    print("\n" + "=" * 80, flush=True)
    print(
        f"[Run] model={model_name} | window={window} | sfreq={sfreq}Hz | "
        f"shuffle_labels={shuffle_labels} | n_subjects={n_subjects} | "
        f"ablation_seed={ablation_seed}",
        flush=True,
    )
    print(
        f"Loaded train/val data: X={X_train_val.shape}, "
        f"y={y_train_val.shape}, subjects={subjects_train_val.shape}, "
        f"selected_subjects={len(unique_subjects)}/{len(all_unique_subjects)} | "
        f"n_folds={n_folds}",
        flush=True,
    )
    print(f"Selected subjects: {selected_subjects}", flush=True)
    print("=" * 80, flush=True)

    kf = KFold(
        n_splits=n_folds, 
        shuffle=True, 
        random_state=seed
    )

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(unique_subjects)): 
        train_subjs = unique_subjects[train_idx]
        val_subjs = unique_subjects[val_idx]

        train_mask = np.isin(subjects_train_val, train_subjs)
        val_mask = np.isin(subjects_train_val, val_subjs)

        X_train = X_train_val[train_mask]
        y_train = y_train_val[train_mask]

        X_val = X_train_val[val_mask]
        y_val = y_train_val[val_mask]

        print(
            f"\nStarting fold {fold_idx + 1}/{n_folds} | "
            f"model={model_name} | window={window} | sfreq={sfreq}Hz | "
            f"train_n={len(y_train)} | val_n={len(y_val)} | "
            f"train_high={(y_train == 1).mean():.3f} | "
            f"val_high={(y_val == 1).mean():.3f}",
            flush=True,
        )

        class_balance_records.append(get_class_balance(
            y=y_train, 
            run_name=run_name,
            split_name="train", 
            model_name=model_name, 
            window=window,
            shuffle_labels=shuffle_labels,
            sfreq=sfreq,
            fold_idx=fold_idx
        ))
        class_balance_records.append(get_class_balance(
            y=y_val, 
            run_name=run_name,
            split_name="val", 
            model_name=model_name, 
            window=window,
            shuffle_labels=shuffle_labels,
            sfreq=sfreq,
            fold_idx=fold_idx
        ))
        
        y_train_used = y_train.copy()
        
        if shuffle_labels: 
            rng = np.random.default_rng(seed + fold_idx)
            rng.shuffle(y_train_used)
        
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True)

        X_train_z = (X_train - mean) / (std + 1e-6)
        X_val_z = (X_val - mean) / (std + 1e-6)

        train_dataset = EEGDataset(X_train_z, y_train_used)
        val_dataset = EEGDataset(X_val_z, y_val)

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=0
        )

        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=0
        )

        model = build_model(model_name=model_name, sfreq=sfreq, num_timepoints=actual_timepoints).to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, 
            mode=mode, 
            factor=config.SCHEDULER_FACTOR, 
            patience=config.SCHEDULER_PATIENCE
        )

        fold_checkpoint_path = checkpoint_dir / f"fold{fold_idx}_best.pt"

        checkpoint_extra = {
            "fold": fold_idx,
            "model_name": model_name,
            "window": window,
            "sfreq": sfreq,
            "shuffle_labels": shuffle_labels,
            "threshold": config.THRESHOLD,
            "zscore_mean": mean.astype(np.float32),
            "zscore_std": std.astype(np.float32),
            "train_subjects": train_subjs,
            "val_subjects": val_subjs,
            "n_subjects_requested": n_subjects,
            "ablation_seed": ablation_seed,
            "selected_subjects": selected_subjects,
        }

        epoch_history, batch_history_per_epoch, epoch_best_info = train_model(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            max_epochs=config.MAX_EPOCHS, 
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE, 
            device=device,
            grad_clip=config.GRAD_CLIP, 
            threshold=config.THRESHOLD, 
            progress_desc=f"Fold {fold_idx + 1}/{n_folds}",
            checkpoint_path=fold_checkpoint_path,
            checkpoint_extra=checkpoint_extra,
        )
        if not epoch_best_info:
            raise RuntimeError(
                f"No valid best model found for fold {fold_idx}. "
                "Check whether the validation fold contains both classes."
            )
        print(
            f"[Fold {fold_idx + 1}/{n_folds} done] "
            f"best_epoch={epoch_best_info['epoch']} | "
            f"best_val_roc_auc={epoch_best_info['best_val_roc_auc']:.4f} | "
            f"best_val_pr_auc={epoch_best_info['best_val_pr_auc']:.4f} | "
            f"best_val_bal_acc={epoch_best_info['best_val_balanced_accuracy']:.4f}",
            flush=True,
        )
        for row in epoch_history:
            row["run_name"] = run_name
            row["fold"] = fold_idx
            row["window"] = window
            row["sfreq"] = sfreq
            row["model"] = model_name
            row["shuffle_labels"] = shuffle_labels
            row["n_subjects_requested"] = n_subjects
            row["ablation_seed"] = ablation_seed
            row["n_selected_subjects"] = len(selected_subjects)

        for row in batch_history_per_epoch:
            row["run_name"] = run_name
            row["fold"] = fold_idx
            row["window"] = window
            row["sfreq"] = sfreq
            row["model"] = model_name
            row["shuffle_labels"] = shuffle_labels
            row["n_subjects_requested"] = n_subjects
            row["ablation_seed"] = ablation_seed
            row["n_selected_subjects"] = len(selected_subjects)

        epoch_best_info["zscore_mean"] = mean.astype(np.float32)
        epoch_best_info["zscore_std"] = std.astype(np.float32)
        epoch_best_info["fold"] = fold_idx
        epoch_best_info["train_subjects"] = train_subjs
        epoch_best_info["val_subjects"] = val_subjs
            
        epoch_history_all.append(epoch_history)
        batch_history_all.append(batch_history_per_epoch)

        oof_probs_all.append(epoch_best_info['best_val_probabilities'])
        oof_labels_all.append(epoch_best_info['best_val_labels'])
        best_summary_records.append({
            "run_name": run_name,
            "fold": fold_idx,
            "best_epoch": epoch_best_info["epoch"],
            "best_val_roc_auc": epoch_best_info["best_val_roc_auc"],
            "best_val_pr_auc": epoch_best_info["best_val_pr_auc"],
            "best_val_loss": epoch_best_info["best_val_loss"],
            "best_val_balanced_accuracy": epoch_best_info["best_val_balanced_accuracy"],
            "threshold": config.THRESHOLD,
            "n_train_subjects": len(train_subjs),
            "n_val_subjects": len(val_subjs),
            "n_subjects_requested": n_subjects,
            "ablation_seed": ablation_seed,
            "n_selected_subjects": len(selected_subjects),
            "window": window,
            "sfreq": sfreq,
            "model": model_name,
            "shuffle_labels": shuffle_labels,
            "checkpoint_path": str(fold_checkpoint_path),
        })

    epoch_log = [row for fold_hist in epoch_history_all for row in fold_hist]
    batch_log = [row for fold_hist in batch_history_all for row in fold_hist]
    oof_probs = np.concatenate(oof_probs_all)
    oof_labels = np.concatenate(oof_labels_all)

    global_threshold, bal_acc_at_global_threshold = find_best_threshold_bal_acc(
        y_true=oof_labels, 
        y_prob=oof_probs
    )
    print("\n" + "=" * 80, flush=True)
    print("[Cross-validation complete]", flush=True)
    print(
        f"model={model_name} | window={window} | sfreq={sfreq}Hz | "
        f"shuffle_labels={shuffle_labels} | n_subjects={n_subjects} | "
        f"ablation_seed={ablation_seed}",
        flush=True,
    )
    print(f"Global validation-derived threshold: {global_threshold:.4f}", flush=True)
    print(
        f"OOF CV balanced accuracy at global threshold: "
        f"{bal_acc_at_global_threshold:.4f}",
        flush=True,
    )
    print("=" * 80 + "\n", flush=True)

    run_summary = {
        "run_name": run_name,
        "model": model_name,
        "window": window,
        "sfreq": int(sfreq),
        "shuffle_labels": bool(shuffle_labels),
        "n_subjects_requested": None if n_subjects is None else int(n_subjects),
        "ablation_seed": None if ablation_seed is None else int(ablation_seed),
        "n_selected_subjects": int(len(selected_subjects)),
        "selected_subjects": [str(s) for s in selected_subjects],
        "n_folds": int(n_folds),
        "global_threshold": float(global_threshold),
        "cv_balanced_accuracy_at_global_threshold": float(bal_acc_at_global_threshold),
    }

    return {
        "config_snapshot": config_snapshot,
        "run_summary": run_summary,
        "epoch_log": epoch_log,
        "batch_log": batch_log,
        "best_summary": best_summary_records,
        "class_balance": class_balance_records,
        "oof_probs": oof_probs,
        "oof_labels": oof_labels,
    }
    