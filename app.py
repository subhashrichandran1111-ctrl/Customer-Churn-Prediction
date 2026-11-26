
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# 1. Load the saved model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "churn_model.pkl"

model_package = joblib.load(model_path)

model = model_package["model"]
scaler = model_package["scaler"]
feature_columns = model_package["feature_columns"]
numerical_columns = model_package["numerical_columns"]

# --------------------------------
# 2. Streamlit page settings
# --------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether "
    "the customer is likely to leave the service."
)

# --------------------------------
# 3. Customer input form
# --------------------------------

with st.form("customer_form"):

    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    st.subheader("Service Information")

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )

    st.subheader("Contract and Payment Information")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    submitted = st.form_submit_button("Predict Churn")

# --------------------------------
# 4. Process input and predict
# --------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # One-hot encode categorical columns
    categorical_columns = input_data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    # Match the training feature columns

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Convert columns to float before scaling
    input_encoded = input_encoded.astype(float)

    # Apply the saved scaler
    input_encoded.loc[:, numerical_columns] = scaler.transform(
        input_encoded[numerical_columns]
    )

    # Make prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("⚠️ This customer is predicted to churn.")
    else:
        st.success("✅ This customer is predicted to stay.")

    st.caption(
        "This is a machine-learning estimate, not a guarantee "
        "of the customer's future behavior."
    )
