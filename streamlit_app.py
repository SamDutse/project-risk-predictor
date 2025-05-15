import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="LGA Project Risk Predictor", layout="centered")

st.title("🔍 LGA Project Risk Prediction App")
st.markdown("Predict whether a project is likely to be **abandoned or delayed** based on key features.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("xgb_project_risk_model.pkl")  # or "rf_project_risk_model.pkl"

model = load_model()

# Sample options from synthetic data
lga_options = ['Jos North', 'Barkin Ladi', 'Langtang South', 'Mangu']
region_options = ['North Central', 'North West', 'South East']
sector_options = ['Health', 'Education', 'Agriculture', 'Infrastructure']
contractor_ids = ['C101', 'C102', 'C103', 'C104', 'C105']

# Sidebar inputs
st.sidebar.header("Enter Project Details")

LGA = st.sidebar.selectbox("Local Government Area", lga_options)
Region = st.sidebar.selectbox("Region", region_options)
Sector = st.sidebar.selectbox("Sector", sector_options)
Contractor_ID = st.sidebar.selectbox("Contractor ID", contractor_ids)

Contractor_Completed = st.sidebar.number_input("Completed Projects by Contractor", min_value=0, max_value=100, value=10)
Contractor_Abandoned = st.sidebar.number_input("Abandoned Projects by Contractor", min_value=0, max_value=20, value=1)
Contractor_Score = st.sidebar.slider("Contractor Score (1-5)", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
Amount = st.sidebar.number_input("Project Amount (₦)", min_value=100000, max_value=500000000, value=5000000, step=100000)
Duration_Days = st.sidebar.slider("Duration (Days)", min_value=30, max_value=720, value=180)

# Predict button
if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "LGA": LGA,
        "Region": Region,
        "Sector": Sector,
        "Contractor_ID": Contractor_ID,
        "Contractor_Completed": Contractor_Completed,
        "Contractor_Abandoned": Contractor_Abandoned,
        "Contractor_Score": Contractor_Score,
        "Amount": Amount,
        "Duration_Days": Duration_Days
    }])

    prediction = model.predict(input_df)[0]
    label = "🚨 High Risk (Delayed or Abandoned)" if prediction == 1 else "✅ On Time"
    st.success(f"**Prediction:** {label}")
