"""
Evaluation metrics for imbalanced classification.
Computes Accuracy, Precision, Recall, F1 scores, ROC-AUC, Average Precision (PR-AUC), and Confusion Matrix.
"""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for binary imbalanced data.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Predicted probabilities for positive class (class 1).
        threshold: Decision boundary threshold (default: 0.5).

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # Core threshold-dependent metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Threshold-independent ranking metrics
    # If only one class present in true labels, handle gracefully
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        ap = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        ap = 0.0

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "threshold": threshold,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def compute_curves(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC and Precision-Recall curve points.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive class probabilities.

    Returns:
        Tuple of (fpr, tpr, roc_thresholds, precision, recall).
    """
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    return fpr, tpr, roc_thresh, prec, rec

