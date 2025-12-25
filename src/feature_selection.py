from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def select_features(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    k: int = 12,
    must_keep: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Select top-k features using mutual information, BUT always keep 'must_keep' features.
    Returns:
      X_train_selected, X_test_selected, selected_feature_names
    """
    if must_keep is None:
        must_keep = []

    must_keep = [c for c in must_keep if c in X_train.columns]

    # If k is smaller than must_keep count, increase k
    if k < len(must_keep):
        k = len(must_keep)

    selector = SelectKBest(score_func=mutual_info_classif, k="all")
    selector.fit(X_train, y_train)

    scores = selector.scores_
    scores = np.nan_to_num(scores, nan=0.0)

    score_series = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)

    selected = list(dict.fromkeys(must_keep))

    for feat in score_series.index:
        if feat not in selected:
            selected.append(feat)
        if len(selected) >= k:
            break

    X_train_sel = X_train[selected].copy()
    X_test_sel = X_test[selected].copy()
    return X_train_sel, X_test_sel, selected
