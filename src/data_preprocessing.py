from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

TARGET_COL = "Attrition"


def load_dataset(path: str) -> pd.DataFrame:
    """Load HR dataset from CSV."""
    return pd.read_csv(path)


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features used by the system (as described in the design doc).
    - TenureLevel: binned YearsAtCompany
    - WorkLifeBalanceScore: sum of satisfaction/balance signals
    """
    df = df.copy()

    if "YearsAtCompany" in df.columns:
        df["TenureLevel"] = pd.cut(
            df["YearsAtCompany"],
            bins=[-1, 2, 5, 10, 40],
            labels=["New", "Junior", "Mid", "Senior"],
            include_lowest=True,
        ).astype("object")

    score_cols = [c for c in ["WorkLifeBalance", "JobSatisfaction", "EnvironmentSatisfaction"] if c in df.columns]
    if score_cols:
        df["WorkLifeBalanceScore"] = df[score_cols].sum(axis=1)

    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing transformer:
    - Numeric: impute median
    - Categorical: impute most_frequent + ordinal encode (handles unseen categories)
    """
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
