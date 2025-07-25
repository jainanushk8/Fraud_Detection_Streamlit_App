import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- 1. Load the Trained Model ---
# Define the path to your saved model relative to the Streamlit app script
model_dir = 'models' # Assuming 'models' directory is at the same level as your streamlit_app.py
model_path = os.path.join(model_dir, 'lgbm_fraud_model.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Make sure 'models' directory and 'lgbm_fraud_model.pkl' exist.")
    st.stop() # Stop the app if the model cannot be loaded
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Streamlit App Title and Description ---
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💸 Online Payment Fraud Detection")
st.markdown("Enter transaction details to predict if it's a fraudulent activity.")

# --- 3. User Input Features ---
st.header("Transaction Details")

st.subheader("General Transaction Info")
amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=1000.00, step=100.0)
oldbalanceOrg = st.number_input("Old Balance Originator ($)", min_value=0.0, value=50000.0, step=1000.0)
oldbalanceDest = st.number_input("Old Balance Destination ($)", min_value=0.0, value=10000.0, step=1000.0)
# isFlaggedFraud removed from UI, as it was not a training feature
day_of_simulation = st.slider("Day of Simulation (1-30)", min_value=1, max_value=30, value=15)
hour_of_day = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=12)

st.subheader("Balance Status Indicators")
is_orig_bal_zero_before = st.checkbox("Originator Balance Zero Before Transaction", value=False)
is_orig_bal_zero_after = st.checkbox("Originator Balance Zero After Transaction", value=False)
is_dest_bal_zero_before = st.checkbox("Destination Balance Zero Before Transaction", value=False)
is_dest_bal_zero_after = st.checkbox("Destination Balance Zero After Transaction", value=False)

st.subheader("Transaction Type")
transaction_type = st.radio(
    "Select Transaction Type:",
    ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
)

# --- 4. Prepare Features for Prediction ---
def preprocess_inputs(amount, oldbalanceOrg, oldbalanceDest,
                      is_orig_bal_zero_before, is_orig_bal_zero_after,
                      is_dest_bal_zero_before, is_dest_bal_zero_after,
                      hour_of_day, day_of_simulation,
                      transaction_type):

    # Create a dictionary to hold the features
    features = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'oldbalanceDest': oldbalanceDest,
        'hour_of_day': hour_of_day,
        'day_of_simulation': day_of_simulation,
        'is_orig_bal_zero_before': 1 if is_orig_bal_zero_before else 0,
        'is_orig_bal_zero_after': 1 if is_orig_bal_zero_after else 0,
        'is_dest_bal_zero_before': 1 if is_dest_bal_zero_before else 0,
        'is_dest_bal_zero_after': 1 if is_dest_bal_zero_after else 0,
        'type_CASH_IN': 0,
        'type_CASH_OUT': 0,
        'type_DEBIT': 0,
        'type_PAYMENT': 0,
        'type_TRANSFER': 0
    }

    # Set the one-hot encoded type based on user selection
    features[f'type_{transaction_type}'] = 1

    # Define the exact 14 features in the order expected by the trained model
    feature_order = [
        'amount',
        'oldbalanceOrg',
        'oldbalanceDest',
        'hour_of_day',
        'day_of_simulation',
        'is_orig_bal_zero_before',
        'is_orig_bal_zero_after',
        'is_dest_bal_zero_before',
        'is_dest_bal_zero_after',
        'type_CASH_IN',
        'type_CASH_OUT',
        'type_DEBIT',
        'type_PAYMENT',
        'type_TRANSFER'
    ]

    # Convert to DataFrame in the correct order
    input_df = pd.DataFrame([features])[feature_order]

    return input_df

# --- 5. Make Prediction ---
if st.button("Predict Fraud"):
    input_df = preprocess_inputs(
        amount, oldbalanceOrg, oldbalanceDest,
        is_orig_bal_zero_before, is_orig_bal_zero_after,
        is_dest_bal_zero_before, is_dest_bal_zero_after,
        hour_of_day, day_of_simulation,
        transaction_type
    )

    prediction_proba = model.predict_proba(input_df)[:, 1] # Probability of being class 1 (fraud)
    prediction = (prediction_proba > 0.5).astype(int) # Default threshold 0.5

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ **This transaction is predicted as FRAUDULENT!**")
    else:
        st.success("✅ This transaction is predicted as LEGITIMATE.")

    st.write(f"Confidence Score (Probability of Fraud): **{prediction_proba[0]:.4f}**")
    st.write("---")
    st.markdown("Disclaimer: This is a predictive model based on historical data and should be used as an assistive tool.")
