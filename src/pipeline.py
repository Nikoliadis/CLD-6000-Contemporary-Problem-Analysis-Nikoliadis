from data_preprocessing import load_dataset, preprocess_data
from feature_selection import select_features
from model_training import train_decision_tree
from evaluation import evaluate_model
import joblib
from pathlib import Path

DATA_PATH = Path("data/hr_data.csv")

def pipeline():
    print("🔵 Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("🔵 Preprocessing & feature engineering...")
    X_train, X_test, y_train, y_test, encoders = preprocess_data(df)

    print("🔵 Selecting features...")
    X_train_sel, X_test_sel, selected_indices = select_features(X_train, y_train, X_test, k=12)

    print("🔵 Training model (Decision Tree)...")
    model = train_decision_tree(X_train_sel, y_train)

    print("🔵 Evaluating model...")
    metrics, cm = evaluate_model(model, X_test_sel, y_test)

    # Save model
    print("🔵 Saving model...")
    joblib.dump(
        {
            'model': model,
            'encoders': encoders,
            'selected_features': X_train_sel.columns.tolist()
        },
        "models/best_model.pkl"
    )

    print("\n🎯 FINAL METRICS:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n📊 Confusion Matrix saved as models/confusion_matrix.png")

if __name__ == "__main__":
    pipeline()
