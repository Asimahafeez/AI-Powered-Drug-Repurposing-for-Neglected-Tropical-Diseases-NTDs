import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Title
st.title("AI-Powered Drug Repurposing for NTDs")

# File uploader
uploaded_file = st.file_uploader("Upload CSV (with features + target column)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    target_col = st.selectbox("Select target column", data.columns)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Detect if target is classification or regression
    if y.nunique() < 20 and y.dtype in [np.int64, np.int32, np.object_]:
        task = "classification"
    else:
        task = "regression"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "classification":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Classification Accuracy:** {acc:.2f}")
    else:
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"**Regression MSE:** {mse:.2f}")

    # User input for prediction
    st.subheader("Try prediction")
    user_input = []
    for col in X.columns:
        val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
        user_input.append(val)

    if st.button("Predict"):
        pred = model.predict([user_input])
        st.success(f"Predicted value: {pred[0]}")
