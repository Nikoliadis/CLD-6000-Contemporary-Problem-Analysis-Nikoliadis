import joblib
import pandas as pd

def load_model(path="models/best_model.pkl"):
    saved = joblib.load(path)
    return saved['model'], saved['encoders'], saved['selected_features']

def predict(model, encoders, selected_features, data_dict):
    df = pd.DataFrame([data_dict])

    # encode using saved encoders
    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    df = df[selected_features]

    return model.predict(df)[0]
