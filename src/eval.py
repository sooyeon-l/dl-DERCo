import numpy as np
from sklearn.metrics import balanced_accuracy_score
# NOTE: this requires that for the held-out test set: 
# 1. Average fold model probabilities on test.
# 2. Apply the one global validation-derived threshold.
# 3. Report test balanced accuracy.
def find_best_threshold_bal_acc(y_true, y_prob):
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


# 1. model.eval()
# 2. collect logits and labels across all batches
# 3. compute BCEWithLogitsLoss over full/weighted batches
# 4. convert logits to probabilities with sigmoid
# 5. compute AUC
# 6. compute balanced accuracy at threshold 0.5

# IMPORTANT: 
# criterion = nn.BCEWithLogitsLoss()
# probs = torch.sigmoid(logits)

# IMPORTANT NOTE: 
# 1. Load best checkpoint from each fold.
# 2. Use each fold’s own train z-score stats.
# 3. Predict probabilities on test.
# 4. Average probabilities across folds.
# 5. Report:
#    - test ROC AUC
#    - test PR AUC
#    - test balanced accuracy at 0.5
#    - test balanced accuracy at global validation-derived threshold
# Return: 
# metrics = {
#     "loss": val_loss,
#     "roc_auc": val_roc_auc,
#     "pr_auc": val_pr_auc,
#     "balanced_accuracy_05": bal_acc_05,
#     "probabilities": probabilities_np,
#     "labels": labels_np,
# }


# NOTE: Function 1 `predict_proba`
# def predict_proba(model, loader, device):
#     model.eval()
#     all_probs = []
#     all_labels = []

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             logits = model(x)
#             probs = torch.sigmoid(logits)

#             all_probs.append(probs.cpu().numpy())
#             all_labels.append(y.numpy())

#     return np.concatenate(all_probs), np.concatenate(all_labels)

# NOTE: Function 2 `compute_binary_metrics`
# def compute_binary_metrics(y_true, y_prob, threshold=0.5):
#     auc = roc_auc_score(y_true, y_prob)
#     preds = (y_prob >= threshold).astype(int)
#     bal_acc = balanced_accuracy_score(y_true, preds)

#     return {
#         "auc": auc,
#         "balanced_accuracy": bal_acc,
#     }

# NOTE: Function 3 `evaluate_cv_ensemble`
# # 1. Load fold checkpoints
# # 2. For each checkpoint:
# #    - rebuild the model
# #    - load model weights
# #    - load fold mean/std from checkpoint
# #    - z-score X_test with that fold's mean/std
# #    - predict test probabilities
# # 3. Average probabilities across folds
# # 4. Compute test AUC and balanced accuracy

# # It needs: 
# "zscore_mean": mean,
# "zscore_std": std,