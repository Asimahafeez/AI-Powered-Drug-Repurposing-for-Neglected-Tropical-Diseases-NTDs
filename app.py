import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model_ntd.joblib")

st.title("AI-Powered Drug Repurposing for Neglected Tropical Diseases (NTDs)")
st.write("Upload data or input features manually to predict potential drug repurposing outcomes.")

# Option to upload CSV
uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())
    preds = model.predict(data)
    st.write("Predictions:")
    st.write(preds)
else:
    st.write("Or enter feature values manually:")
    input_data = []
    cols = [f"Feature_{i}" for i in range(1, 11)]
    for col in cols:
        val = st.number_input(f"Enter {col}", value=0.0)
        input_data.append(val)
    if st.button("Predict"):
        data = np.array(input_data).reshape(1, -1)
        pred = model.predict(data)
        st.write(f"Prediction: {pred[0]}")
