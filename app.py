import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Early-Warning System",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 32px; font-weight: bold; color: #1E3A8A; }
    .sub-header { font-size: 18px; color: #4B5563; margin-bottom: 25px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    return joblib.load('churn_model.pkl')

try:
    model = load_trained_model()
except Exception as e:
    st.error("Model file 'churn_model.pkl' not found. Please run `python train_model.py` first.")
    st.stop()

# Header Section
st.markdown('<div class="main-header">📊 Customer Churn Early-Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict customer churn probability using Random Forest Ensemble Learning</div>', unsafe_allow_html=True)

st.sidebar.header("📋 Customer Profile Inputs")

# Sidebar Form Inputs
tenure = st.sidebar.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges (₹)", min_value=100.0, max_value=10000.0, value=650.0, step=50.0)
auto_total = tenure * monthly_charges
total_charges = st.sidebar.number_input("Total Charges (₹)", min_value=100.0, max_value=500000.0, value=float(auto_total), step=100.0)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

tech_support = st.sidebar.selectbox("Tech Support Subscribed", ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Online Security Subscribed", ["No", "Yes", "No internet service"])
paperless = st.sidebar.selectbox("Paperless Billing Enabled", ["Yes", "No"])

# Main Display Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Account Metadata")
    input_df = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'TechSupport': tech_support,
        'OnlineSecurity': online_security,
        'PaperlessBilling': paperless
    }])
    
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

with col2:
    st.subheader("Prediction & Retention Assessment")
    
    if st.button("Calculate Churn Probability", type="primary", use_container_width=True):
        churn_prob = model.predict_proba(input_df)[0][1] * 100
        
        st.write("---")
        st.metric(label="Calculated Churn Probability", value=f"{churn_prob:.1f}%")
        
        if churn_prob >= 60.0:
            st.error("🚨 HIGH CHURN RISK FLAG")
            st.markdown("""
            **Recommended Retention Protocol:**
            * Offer immediate 15% discount on upgrading to a 1-Year or 2-Year Contract.
            * Assign priority customer service handling.
            * Bundle complimentary Online Security package.
            """)
        elif 30.0 <= churn_prob < 60.0:
            st.warning("⚠️ MODERATE CHURN RISK FLAG")
            st.markdown("""
            **Recommended Retention Protocol:**
            * Send an automated check-in survey to identify service dissatisfaction.
            * Offer a 3-month free trial of Tech Support features.
            """)
        else:
            st.success("✅ LOW CHURN RISK FLAG")
            st.markdown("""
            **Recommended Retention Protocol:**
            * Account is stable. Target for upselling premium streaming packages or hardware add-ons.
            """)