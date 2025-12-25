import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from prediction_service import load_pipeline, predict_from_dict, log_prediction

st.title("🔮 Predict Attrition")

model_path = BASE_DIR / "models" / "best_model.pkl"
if not model_path.exists():
    st.warning("Model not found. Please train the model first from the **Train Model** page.")
    st.stop()

model_artifact = load_pipeline(model_path)

st.write("Enter a subset of employee details. Missing fields will be handled by preprocessing.")

# ✅ Threshold slider to control sensitivity for predicting "Yes"
threshold = st.slider(
    "Decision threshold for 'Attrition = YES'",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.01,
)
st.caption("Lower threshold → more 'YES' predictions (higher recall). Higher threshold → stricter predictions.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=3)
    hours_per_day = st.number_input("Working hours per day", min_value=1, max_value=16, value=8)

with col2:
    # ✅ Greek baseline default, but UI stays English
    monthly_income = st.number_input("MonthlyIncome (€)", min_value=0, max_value=50000, value=880)
    job_satisfaction = st.slider("JobSatisfaction (1-4)", 1, 4, 3)
    work_life_balance = st.slider("WorkLifeBalance (1-4)", 1, 4, 3)

with col3:
    environment_satisfaction = st.slider("EnvironmentSatisfaction (1-4)", 1, 4, 3)
    job_role = st.text_input("JobRole", value="Sales Executive")
    department = st.text_input("Department", value="Sales")

# Auto-derive OverTime from working hours
overtime = "Yes" if hours_per_day > 8 else "No"
st.write(f"Derived OverTime: **{overtime}** (based on {hours_per_day} hours/day)")

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

log_path = BASE_DIR / "models" / "prediction_log.csv"

if st.button("Predict"):
    pred_label, prob_yes = predict_from_dict(model_artifact, employee)

    # ✅ Decision based on threshold (not just default model label)
    will_leave = prob_yes >= threshold
    decision = "Yes" if will_leave else "No"

    if will_leave:
        st.error(f"⚠️ Prediction: Attrition = YES (probability ~ {prob_yes:.2f})")
    else:
        st.success(f"✅ Prediction: Attrition = NO (probability ~ {prob_yes:.2f})")

    # ✅ Save prediction record
    log_prediction(employee, prob_yes, decision, log_path)

st.caption("Note: For best results, provide as many fields as available from the dataset.")

st.subheader("📁 Prediction Record")

if log_path.exists():
    df_log = pd.read_csv(log_path)
    st.dataframe(df_log.tail(20), use_container_width=True)
else:
    st.info("No Record Yet.")
