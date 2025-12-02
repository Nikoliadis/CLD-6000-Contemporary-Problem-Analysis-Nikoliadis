import streamlit as st
import sys
from pathlib import Path

# Add src folder to Python path
BASE_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from pipeline import pipeline

st.title("🧠 Train Machine Learning Model")
st.write("Run the ML pipeline and generate the Decision Tree model.")
st.write("---")

if st.button("🚀 Train Model Now"):
    st.write("Training model... please wait.")
    pipeline()
    st.success("Model training completed!")

    st.subheader("📊 Results")
    st.image(str(BASE_DIR / "models" / "confusion_matrix.png"), caption="Confusion Matrix")
