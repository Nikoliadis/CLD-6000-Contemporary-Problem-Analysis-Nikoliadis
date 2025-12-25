from __future__ import annotations

from sklearn.feature_selection import SelectKBest, mutual_info_classif

import pandas as pd
import numpy as np

def build_selector(k: int = 12) -> SelectKBest:
    """Select top-k features using mutual information."""
    return SelectKBest(score_func=mutual_info_classif, k=k)

def select_top_k_features(X: pd.DataFrame, y: pd.Series, k: int):
    mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    selected_features = mi_scores.head(k).index.tolist()

    return X[selected_features], selected_features, mi_scores