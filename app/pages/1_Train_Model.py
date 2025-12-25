import json
import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline import run_pipeline

st.title("🧠 Train Model")
st.write("Run the end-to-end ML pipeline and view evaluation outputs.")

k = st.slider("Number of selected features (k)", min_value=5, max_value=30, value=12, step=1)

if st.button("🚀 Run Training Pipeline"):
    with st.spinner("Training..."):
        metrics, _, _, model_path = run_pipeline(k_features=k)
    st.success(f"Training completed. Model saved to: {model_path}")

    st.subheader("📊 Metrics")
    st.json(metrics)

    cm_path = BASE_DIR / "models" / "confusion_matrix.png"
    if cm_path.exists():
        st.subheader("Confusion Matrix")
        st.image(str(cm_path))

    rep_path = BASE_DIR / "models" / "classification_report.txt"
    if rep_path.exists():
        st.subheader("Classification Report")
        st.code(rep_path.read_text(encoding="utf-8"))
