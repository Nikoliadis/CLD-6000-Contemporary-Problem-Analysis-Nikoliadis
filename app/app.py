import streamlit as st

st.set_page_config(page_title="Employee Attrition Prediction System", layout="wide")

st.title("👔 Employee Attrition Prediction System")
<<<<<<< HEAD
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
=======
st.write(
    "A local (localhost) HR analytics dashboard for training a Decision Tree model and predicting employee attrition."
)

st.markdown("""
**Pages (left sidebar):**
- **Train Model**: run the full ML pipeline (preprocessing → feature selection → training → evaluation)
- **Predict**: enter employee features and get an attrition prediction
""")

st.info("Tip: Start from **Train Model** to generate `models/best_model.pkl` and evaluation artifacts.")
>>>>>>> 57b6181 (Added And Fixed All The Functionalites Algorithms Needs To Be Fixed)
