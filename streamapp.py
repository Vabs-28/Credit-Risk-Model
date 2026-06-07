import streamapp as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="EPS Predictor", layout="centered")

# Load models once to save memory and speed up the app
@st.cache_resource
def load_models():
    model = pickle.load(open('eps_v1.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    return model, scaler

model, scaler = load_models()

# App UI
st.title("EPS Prediction AI 📈")
st.write("Enter the bank's operational metrics below to predict its Earnings Per Share.")
st.markdown("---")

# Layout: 2 columns for a cleaner look
col1, col2 = st.columns(2)

# I've pre-filled your test values as defaults!
with col1:
    v1 = st.number_input("ROCE (%)", value=1.91, format="%.2f")
    v2 = st.number_input("CASA (%)", value=39.47, format="%.2f")
    v3 = st.number_input("Return on Equity / Networth (%)", value=14.36, format="%.2f")
    v4 = st.number_input("Non-Interest Income/Total Assets (%)", value=0.68, format="%.2f")

with col2:
    v5 = st.number_input("Operating Profit/Total Assets (%)", value=0.27, format="%.2f")
    v6 = st.number_input("Operating Expenses/Total Assets (%)", value=1.68, format="%.2f")
    v7 = st.number_input("Interest Expenses/Total Assets (%)", value=3.30, format="%.2f")
    v8 = st.number_input("Face value", value=2.00, format="%.2f")

st.markdown("---")

# Prediction Logic
if st.button("Predict EPS", type="primary"):
    # Format inputs into a 2D array
    raw_features = np.array([[v1, v2, v3, v4, v5, v6, v7, v8]])
    
    # Scale the inputs
    scaled_features = scaler.transform(raw_features)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    
    # Display Result
    st.success(f"### Predicted EPS: ₹{round(prediction, 2)}")