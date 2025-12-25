from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(y_true, y_pred) -> Tuple[Dict[str, float], str, "list[list[int]]"]:
    """
    Return metrics dict, classification report, and confusion matrix.
    Handles string labels (e.g., 'No'/'Yes') by setting pos_label='Yes'.
    """
    
    labels = sorted(set(list(y_true) + list(y_pred)))
    pos_label = "Yes" if "Yes" in labels else (labels[-1] if labels else 1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
    }

    report = classification_report(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics, report, cm


def save_confusion_matrix(cm, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
