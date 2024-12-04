import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained Random Forest model
with open("rf_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

# Set up the Streamlit app
st.title("Fraud Detection System")
st.subheader("Created By Muzamil Hussain")
st.write(
    """
    This app analyzes credit card transactions and predicts whether they are fraudulent or legitimate.
    """
)

# Sidebar inputs
st.sidebar.header("Transaction Details")
time = st.sidebar.number_input("Transaction Time (seconds)", min_value=0, step=1)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=0.01)

# Add additional inputs for V1 to V28
v_features = {}
for i in range(1, 29):
    v_features[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0)

# Add new features: Hour, DayOfWeek, and LogAmount
hour = st.sidebar.number_input("Transaction Hour (0-23)", min_value=1, max_value=24, step=1)
day_of_week = st.sidebar.number_input("Day of the Week (1=Monday, ..., 7=Sunday)", min_value=1, max_value=7, step=1)
log_amount = st.sidebar.number_input("Logarithm of Transaction Amount", min_value=0.0, step=0.01)

# Combine inputs into a DataFrame
input_data = {
    "Time": [time],
    "Amount": [amount],
    **{f"V{i}": [v_features[f"V{i}"]] for i in range(1, 29)},
    "Hour": [hour],
    "DayOfWeek": [day_of_week],
    "LogAmount": [log_amount]
}
input_df = pd.DataFrame(input_data)

# Ensure input_df columns match the training data columns
expected_columns = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", 
    "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", 
    "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Hour", "DayOfWeek", "LogAmount"
]
input_df = input_df[expected_columns]

# Prediction button
if st.sidebar.button("Analyze Transaction"):
    try:
        # Predict and get probabilities
        prediction = rf_model.predict(input_df)
        probability = rf_model.predict_proba(input_df)[:, 1]

        # Display results
        if prediction[0] == 1:
            st.error(
                f"⚠️ Fraudulent Transaction Detected! (Probability: {probability[0] * 100:.2f}%)"
            )
        else:
            st.success(
                f"✔️ Transaction is Legitimate. (Probability: {100 - probability[0] * 100:.2f}%)"
            )
    except ValueError as e:
        st.error(f"Error: {str(e)}. Please ensure all required inputs are provided.")

# Display input data for debugging (optional)
st.write("Input Data:", input_df)
