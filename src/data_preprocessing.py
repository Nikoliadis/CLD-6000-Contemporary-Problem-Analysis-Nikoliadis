import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    """Load HR dataset from CSV."""
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame, target_column: str = "Attrition"):
    """
    Preprocess HR dataset:
    - drop rows with all NaNs
    - encode categorical variables
    - split into train / test
    """
    
    df = df.dropna(how="all")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    if y.dtype == "object":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    else:
        target_encoder = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoders, target_encoder