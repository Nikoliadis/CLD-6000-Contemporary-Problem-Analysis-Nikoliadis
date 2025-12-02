import streamlit as st

st.set_page_config(page_title="Employee Attrition Prediction System", layout="wide")

st.title("👔 Employee Attrition Prediction System")
st.write("A machine learning application for predicting employee turnover.")
st.write("---")

st.subheader("📌 What this app can do")
st.markdown("""
- Train a Decision Tree model on the IBM HR dataset  
- Display evaluation metrics  
- Show confusion matrix  
- Predict employee attrition based on input features  
""")

st.subheader("📁 App Pages")
st.markdown("""
- **Train Model** → Run the ML pipeline  
- **Predict** → Use the saved model to predict attrition for new employees  
""")

st.write("---")
st.info("Use the sidebar on the left to navigate between pages.")
