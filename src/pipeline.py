from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import load_dataset, add_feature_engineering, split_features_target, build_preprocessor
from feature_selection import build_selector
from model_training import build_model
from evaluation import evaluate_model, save_confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "hr_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run_pipeline(k_features: int = 12, test_size: float = 0.30, random_state: int = 42):
    # 1) Load
    df = load_dataset(str(DATA_PATH))

    # 2) Feature engineering
    df = add_feature_engineering(df)

    # 3) Split X/y
    X, y = split_features_target(df)

    # 4) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5) Build pipeline: preprocessing -> feature selection -> model
    preprocessor = build_preprocessor(X_train)
    selector = build_selector(k=k_features)
    model = build_model(random_state=random_state)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("select", selector),
            ("model", model),
        ]
    )

    # 6) Train
    pipe.fit(X_train, y_train)

    # 7) Evaluate
    y_pred = pipe.predict(X_test)
    metrics, report, cm = evaluate_model(y_test, y_pred)

    # 8) Save artifacts
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(
        {
            "pipeline": pipe,
            "train_columns": list(X_train.columns),
        },
        model_path,
    )


    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(cm, MODELS_DIR / "confusion_matrix.png")

    return metrics, report, cm, model_path


if __name__ == "__main__":
    m, _, _, mp = run_pipeline()
    print("🎯 FINAL METRICS:")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
    print(f"\nSaved model to: {mp}")
    print("Saved confusion matrix to: models/confusion_matrix.png")
