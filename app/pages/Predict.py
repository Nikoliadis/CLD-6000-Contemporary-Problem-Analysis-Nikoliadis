import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Add src to path
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from prediction_service import load_model, predict

st.title("🔍 Predict Employee Attrition")
st.write("Enter employee details to predict whether they are at risk of leaving.")
st.write("---")

model, encoders, selected_features = load_model(str(BASE_DIR / "models" / "best_model.pkl"))

# Input form
age = st.number_input("Age", min_value=18, max_value=70, value=30)
income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
years = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
overtime = st.selectbox("OverTime", ["Yes", "No"])

employee = {
    "Age": age,
    "MonthlyIncome": income,
    "YearsAtCompany": years,
    "OverTime": overtime
}

if st.button("🔮 Predict"):
    result = predict(model, encoders, selected_features, employee)
    if result == 1:
        st.error("⚠️ This employee is likely to ATTRITE.")
    else:
        st.success("✅ This employee is NOT likely to attrite.")
