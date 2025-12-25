from __future__ import annotations

from sklearn.feature_selection import SelectKBest, mutual_info_classif


def build_selector(k: int = 12) -> SelectKBest:
    """Select top-k features using mutual information."""
    return SelectKBest(score_func=mutual_info_classif, k=k)
