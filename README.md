# Employee Attrition Prediction System (ML + Streamlit)

## Quick start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python src\pipeline.py
python -m streamlit run app\app.py
```

## What this project does
- Loads IBM HR Attrition dataset from `data/hr_data.csv`
- Preprocesses data (imputation + encoding)
- Selects top-K features (mutual information)
- Trains a Decision Tree classifier
- Evaluates model and saves:
  - `models/best_model.pkl` (full sklearn Pipeline)
  - `models/metrics.json`
  - `models/classification_report.txt`
  - `models/confusion_matrix.png`
- Streamlit app:
  - Train page (runs pipeline and shows metrics + confusion matrix)
  - Predict page (predicts attrition for a new employee profile)

