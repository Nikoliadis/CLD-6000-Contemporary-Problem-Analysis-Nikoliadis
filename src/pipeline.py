from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from data_preprocessing import load_dataset, build_preprocess_pipeline, build_preprocessor, add_feature_engineering
from feature_selection import select_features
from evaluation import evaluate_model, save_confusion_matrix, save_classification_report


class ColumnSelector:
    """
    Simple transformer that selects columns by integer indices (after preprocessing output).
    Works with numpy arrays.
    """
    def __init__(self, selected_indices):
        self.selected_indices = list(selected_indices)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_indices]


def run_pipeline(k_features=12):
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PATH = BASE_DIR / "data" / "hr_data.csv"
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_dataset(str(DATA_PATH))

    X_train, X_test, y_train, y_test = build_preprocess_pipeline(df)

    X_train = add_feature_engineering(X_train)
    X_test = add_feature_engineering(X_test)

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_p.shape[1])]

    X_train_df = (
        X_train_p if hasattr(X_train_p, "toarray") is False else X_train_p.toarray()
    )
    X_test_df = (
        X_test_p if hasattr(X_test_p, "toarray") is False else X_test_p.toarray()
    )

    import pandas as pd
    X_train_df = pd.DataFrame(X_train_df, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_df, columns=feature_names)

    must_keep = ["YearsAtCompany", "JobSatisfaction", "WorkLifeBalance", "OverTime"]

    must_keep_after = []
    for mk in must_keep:
        for fn in feature_names:
            if fn == mk or fn.endswith(mk) or mk in fn:
                must_keep_after.append(fn)

    X_train_sel, X_test_sel, selected_feature_names = select_features(
        X_train_df,
        y_train,
        X_test_df,
        k=k_features,
        must_keep=must_keep_after,
    )

    selected_indices = [feature_names.index(f) for f in selected_feature_names]

    candidates = {
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    best_name = None
    best_f1 = -1
    best_model = None

    for name, model in candidates.items():
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)

        metrics, report, cm = evaluate_model(y_test, y_pred, pos_label="Yes")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = model

    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(X_train_sel, y_train)

    y_pred_cal = calibrated.predict(X_test_sel)
    metrics, report, cm = evaluate_model(y_test, y_pred_cal, pos_label="Yes")

    save_confusion_matrix(cm, MODELS_DIR / "confusion_matrix.png")
    save_classification_report(report, MODELS_DIR / "classification_report.txt")

    full_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("select", ColumnSelector(selected_indices)),
            ("model", calibrated),
        ]
    )

    full_pipeline.fit(X_train, y_train)

    artifact = {
        "pipeline": full_pipeline,
        "selected_feature_names": selected_feature_names,
        "selected_indices": selected_indices,
        "classes_": list(getattr(calibrated, "classes_", [])),
        "best_candidate": best_name,
        "train_columns": list(X_train.columns),
    }

    out_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(artifact, out_path)

    return metrics, report, cm, out_path, selected_feature_names
