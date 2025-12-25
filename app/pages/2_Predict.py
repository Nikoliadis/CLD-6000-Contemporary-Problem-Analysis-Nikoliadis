import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from prediction_service import load_pipeline, predict_from_dict

st.title("🔮 Predict Attrition")

model_path = BASE_DIR / "models" / "best_model.pkl"
if not model_path.exists():
    st.warning("Model not found. Please train the model first from the **Train Model** page.")
    st.stop()

model_artifact = load_pipeline(model_path)

st.write("Enter a subset of employee details. Missing fields will be handled by preprocessing.")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=3)
    overtime = st.selectbox("OverTime", ["Yes", "No"])

with col2:
    monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=50000, value=5000)
    job_satisfaction = st.slider("JobSatisfaction (1-4)", 1, 4, 3)
    work_life_balance = st.slider("WorkLifeBalance (1-4)", 1, 4, 3)

with col3:
    environment_satisfaction = st.slider("EnvironmentSatisfaction (1-4)", 1, 4, 3)
    job_role = st.text_input("JobRole", value="Sales Executive")
    department = st.text_input("Department", value="Sales")

employee = {
    "Age": age,
    "YearsAtCompany": years_at_company,
    "OverTime": overtime,
    "MonthlyIncome": monthly_income,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "EnvironmentSatisfaction": environment_satisfaction,
    "JobRole": job_role,
    "Department": department,
}

if st.button("Predict"):
    pred, prob = predict_from_dict(model_artifact, employee)
    if pred == "Yes":
        st.error(f"⚠️ Prediction: Attrition = YES (probability ~ {prob:.2f})")
    else:
        st.success(f"✅ Prediction: Attrition = NO (probability ~ {prob:.2f})")

st.caption("Note: For best results, provide as many fields as available from the dataset.")
