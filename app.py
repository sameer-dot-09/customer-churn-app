import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline model
model = joblib.load("churn_pipeline.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Title
st.title("Customer Churn Prediction 🚀")

# Sidebar for inputs
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Predict button
if st.sidebar.button("Predict"):

    # Create input dataframe
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment
    }])

    # Prediction
    result = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    # Display result
    st.subheader("Prediction Result")

    if result[0] == 1:
        st.error(f"⚠️ High Risk: Customer likely to churn ({prob:.2f})")
    else:
        st.success(f"✅ Customer likely to stay ({prob:.2f})")

    # Progress bar (visual probability)
    st.subheader("Churn Probability")
    st.progress(int(prob * 100))

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Customer Churn ML Project")
