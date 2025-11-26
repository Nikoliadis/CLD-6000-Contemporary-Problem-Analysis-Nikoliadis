import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_dataset(path: str):
    return pd.read_csv(path)

def preprocess_data(df):
    # 1. Remove fully empty rows
    df = df.dropna(how='all')

    # 2. Drop rows with missing critical values
    df = df.dropna(how='any')

    # 3. FEATURE ENGINEERING
    df['TenureLevel'] = pd.cut(
        df['YearsAtCompany'],
        bins=[0, 2, 5, 10, 40],
        labels=['New', 'Junior', 'Mid', 'Senior'],
        include_lowest=True
    )

    # Convert categorical result TO STRING to avoid the NaN category problem
    df['TenureLevel'] = df['TenureLevel'].astype(str)

    df['WorkLifeBalanceScore'] = (
        df['WorkLifeBalance'] +
        df['JobSatisfaction'] +
        df['EnvironmentSatisfaction']
    )

    # 4. LABEL ENCODING
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 5. FINAL NA CLEAN
    df = df.fillna(0)  # now safe because no column is Categorical anymore

    # 6. TRAIN / TEST SPLIT
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoders
