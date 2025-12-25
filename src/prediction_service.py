from __future__ import annotations
from datetime import datetime

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from pathlib import Path

from data_preprocessing import add_feature_engineering

def load_pipeline(model_path: str | Path) -> Dict[str, Any]:
    obj = joblib.load(str(model_path))

    if isinstance(obj, dict) and "pipeline" in obj:
        pipe = obj["pipeline"]
        cols = obj.get("train_columns", None)
        return {"pipeline": pipe, "train_columns": cols}

    pipe = obj


    cols = None
    try:
        preprocess = pipe.named_steps.get("preprocess")
        cols = []
        for _, _, col_list in preprocess.transformers_:
            if isinstance(col_list, list):
                cols.extend(col_list)
    except Exception:
        cols = None

    return {"pipeline": pipe, "train_columns": cols}


def _align_columns(X: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
    """
    Ensure X has exactly the same columns as training:
    - add missing with np.nan (handled by imputers)
    - drop unexpected extra columns
    - reorder columns
    """
    import numpy as np

    X = X.copy()

    for col in train_columns:
        if col not in X.columns:
            X[col] = np.nan

    X = X[train_columns]
    return X


def predict_from_dict(model_artifact: Dict[str, Any], employee: Dict[str, Any]) -> Tuple[str, float]:
    """
    Predict attrition from a dict of employee features.
    Returns:
      - predicted_label: 'Yes' or 'No'
      - probability_of_yes: float (0..1) or NaN if not available
    """
    pipe = model_artifact["pipeline"]
    train_columns = model_artifact["train_columns"]

    X = pd.DataFrame([employee])

    X = add_feature_engineering(X)

    if train_columns is not None:
        X = _align_columns(X, train_columns)

    pred_label = pipe.predict(X)[0]

    prob_yes = float("nan")
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        classes = list(pipe.classes_) if hasattr(pipe, "classes_") else None
        if classes and "Yes" in classes:
            prob_yes = float(proba[classes.index("Yes")])
        else:
            prob_yes = float(max(proba))

def log_prediction(employee: dict, prob_yes: float, decision: str, log_path: str | Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = employee.copy()
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["prob_yes"] = float(prob_yes)
    row["decision"] = decision

    df_row = pd.DataFrame([row])

    if log_path.exists():
        df_row.to_csv(log_path, mode="a", index=False, header=False, encoding="utf-8")
    else:
        df_row.to_csv(log_path, index=False, encoding="utf-8")

    return str(pred_label), prob_yes
