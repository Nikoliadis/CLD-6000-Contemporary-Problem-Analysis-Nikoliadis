import os
from pathlib import Path

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_preprocessing import load_data, preprocess_data

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hr_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_and_evaluate():
    # 1. Load data
    df = load_data(str(DATA_PATH))

    # 2. Preprocess
    X_train, X_test, y_train, y_test, encoders, target_encoder = preprocess_data(df)

    # 3. Define models to compare
    models = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
    }

    results = {}

    best_model_name = None
    best_f1 = 0.0

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        print("\nClassification report:\n")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # 4. Save best model
    best_model = results[best_model_name]["model"]
    best_model_path = MODELS_DIR / f"{best_model_name}_best_model.pkl"
    joblib.dump(
        {
            "model": best_model,
            "feature_encoders": encoders,
            "target_encoder": target_encoder,
        },
        best_model_path,
    )

    print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")
    print(f"Saved to: {best_model_path}")

if __name__ == "__main__":
    train_and_evaluate()
