from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier


def build_model(random_state: int = 42) -> DecisionTreeClassifier:
    """
    Decision Tree classifier (interpretable baseline), aligned with the design document.
    """
    return DecisionTreeClassifier(
        max_depth=6,
        random_state=random_state,
        class_weight="balanced",
    )
