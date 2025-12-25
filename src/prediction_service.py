from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import add_feature_engineering


def load_pipeline(model_path: Union[str, Path]) -> Any:
    """Load saved model artifact from disk."""
    return joblib.load(str(model_path))


def _get_pipeline(artifact: Any):
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"]
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


def _get_train_columns(artifact: Any, pipe: Any):
    """
    Prefer explicit train_columns if present.
    Otherwise recover expected columns from ColumnTransformer.
    """
    if isinstance(artifact, dict) and artifact.get("train_columns"):
        return list(artifact["train_columns"])

    # Recover from preprocess step
    try:
        preprocess = pipe.named_steps.get("preprocess")
        cols = []
        for _, _, col_list in preprocess.transformers_:
            if isinstance(col_list, list):
                cols.extend(col_list)
        return cols if cols else None
    except Exception:
        return None


def _align_columns(X: pd.DataFrame, train_cols: list[str]) -> pd.DataFrame:
    """
    Add missing columns with np.nan (sklearn-friendly), drop extras, reorder.
    """
    X = X.copy()
    for col in train_cols:
        if col not in X.columns:
            X[col] = np.nan  # ✅ critical: np.nan (NOT pd.NA)
    return X[train_cols]


def predict_from_dict(model_artifact: Any, employee: Dict[str, Any]) -> Tuple[str, float]:
    """
    Returns:
      pred_label: 'Yes'/'No'
      prob_yes: probability for class 'Yes' (NaN if not available)
    """
    pipe = _get_pipeline(model_artifact)

    # Build input frame
    X = pd.DataFrame([employee])

    # Feature engineering must happen before alignment
    X = add_feature_engineering(X)

    # Align columns to training expectation
    train_cols = _get_train_columns(model_artifact, pipe)
    if train_cols:
        X = _align_columns(X, train_cols)

    # Predict
    pred_label = pipe.predict(X)[0]

    # Probability for YES
    prob_yes = float("nan")
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        classes = list(getattr(pipe, "classes_", []))
        if "Yes" in classes:
            prob_yes = float(proba[classes.index("Yes")])
        elif 1 in classes:
            prob_yes = float(proba[classes.index(1)])
        else:
            prob_yes = float(proba[1]) if len(proba) > 1 else float(proba[0])

    return str(pred_label), prob_yes


def log_prediction(
    employee: Dict[str, Any],
    prob_yes: float,
    decision: str,
    log_path: Union[str, Path],
) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(employee)
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["prob_yes"] = "" if (prob_yes != prob_yes) else float(prob_yes)  # keep NaN safe
    row["decision"] = decision

    df_row = pd.DataFrame([row])

    if log_path.exists():
        df_row.to_csv(log_path, mode="a", index=False, header=False, encoding="utf-8")
    else:
        df_row.to_csv(log_path, index=False, encoding="utf-8")
