from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

TARGET_COL = "Attrition"


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight feature engineering that is safe for BOTH training and single-row prediction.
    """
    df = df.copy()

    if "WorkLifeBalance" in df.columns and "WorkLifeBalanceScore" not in df.columns:
        df["WorkLifeBalanceScore"] = df["WorkLifeBalance"]

    if "YearsAtCompany" in df.columns and "TenureLevel" not in df.columns:
        bins = [-np.inf, 1, 3, 5, 10, np.inf]
        labels = ["0-1", "2-3", "4-5", "6-10", "10+"]
        df["TenureLevel"] = pd.cut(df["YearsAtCompany"], bins=bins, labels=labels)

    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
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


def build_preprocess_pipeline(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train/test split on engineered data.
    Returns raw X_train/X_test (DataFrames) + y.
    """
    df = add_feature_engineering(df)
    X, y = split_features_target(df, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
