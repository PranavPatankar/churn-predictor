import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Dark Design System (Custom CSS)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 30px;
        border-radius: 14px;
        border: 1px solid #334155;
        margin-bottom: 25px;
    }
    .header-title {
        color: #F8FAFC;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 6px;
    }
    
    /* Card Container */
    .custom-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-title {
        color: #CBD5E1;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
    }
    
    /* Grid Data Row */
    .data-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #334155;
        font-size: 14px;
    }
    .data-row:last-child {
        border-bottom: none;
    }
    .data-label { color: #94A3B8; font-weight: 500; }
    .data-val { color: #F8FAFC; font-weight: 600; }
    
    /* Risk Alert Cards */
    .risk-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-left: 4px solid #EF4444;
        color: #FCA5A5;
    }
    .risk-moderate {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-left: 4px solid #F59E0B;
        color: #FDE68A;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-left: 4px solid #10B981;
        color: #A7F3D0;
    }
    .risk-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .risk-desc {
        font-size: 13px;
        color: #E2E8F0;
        line-height: 1.6;
    }

    /* Metric Display */
    .prob-number {
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .prob-high { color: #EF4444; }
    .prob-mod { color: #F59E0B; }
    .prob-low { color: #10B981; }
    </style>
""", unsafe_allow_html=True)


# 3. Model Loading
@st.cache_resource
def load_trained_model():
    return joblib.load('churn_model.pkl')

try:
    model = load_trained_model()
except Exception:
    st.error("Model file 'churn_model.pkl' not found. Please run `python train_model.py` first.")
    st.stop()


# 4. Main Header Component
st.markdown("""
    <div class="header-container">
        <div class="header-title">Customer Retention Risk Engine</div>
        <div class="header-subtitle">Predictive Churn Scoring & Recommended Retention Strategies</div>
    </div>
""", unsafe_allow_html=True)


# 5. Sidebar Controls
st.sidebar.markdown("### Customer Parameters")

tenure = st.sidebar.slider("Account Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges (₹)", min_value=100.0, max_value=10000.0, value=650.0, step=50.0)
auto_total = tenure * monthly_charges
total_charges = st.sidebar.number_input("Total Lifetime Charges (₹)", min_value=100.0, max_value=500000.0, value=float(auto_total), step=100.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Service Details")

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])


# 6. Dashboard Layout
col_left, col_right = st.columns([1, 1], gap="large")

# Left Column: Input Data Summary Card
with col_left:
    st.markdown("""
        <div class="custom-card">
            <div class="card-title">Customer Account Profile</div>
            <div class="data-row"><span class="data-label">Tenure</span><span class="data-val">{} Months</span></div>
            <div class="data-row"><span class="data-label">Monthly Billed</span><span class="data-val">₹{:.2f}</span></div>
            <div class="data-row"><span class="data-label">Total Billed</span><span class="data-val">₹{:.2f}</span></div>
            <div class="data-row"><span class="data-label">Contract Type</span><span class="data-val">{}</span></div>
            <div class="data-row"><span class="data-label">Internet Service</span><span class="data-val">{}</span></div>
            <div class="data-row"><span class="data-label">Payment Method</span><span class="data-val">{}</span></div>
            <div class="data-row"><span class="data-label">Tech Support</span><span class="data-val">{}</span></div>
            <div class="data-row"><span class="data-label">Online Security</span><span class="data-val">{}</span></div>
            <div class="data-row"><span class="data-label">Paperless Billing</span><span class="data-val">{}</span></div>
        </div>
    """.format(
        tenure, monthly_charges, total_charges, contract, 
        internet_service, payment_method, tech_support, 
        online_security, paperless
    ), unsafe_allow_html=True)

# Right Column: Risk Assessment & Actions
with col_right:
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
    
    churn_prob = model.predict_proba(input_df)[0][1] * 100

    # Color mapping based on probability
    if churn_prob >= 60.0:
        prob_class = "prob-high"
        risk_class = "risk-high"
        status_text = "High Risk of Churn"
        action_plan = """
        • Offer 15% discount on converting to a 1-Year or 2-Year Contract.<br>
        • Route account calls to dedicated Senior Retention Specialist.<br>
        • Provide complimentary 6-month Online Security bundle upgrade.
        """
    elif 30.0 <= churn_prob < 60.0:
        prob_class = "prob-mod"
        risk_class = "risk-moderate"
        status_text = "Moderate Churn Risk"
        action_plan = """
        • Trigger automated feedback survey to identify service bottlenecks.<br>
        • Offer a 3-month free trial of Tech Support features.<br>
        • Highlight annual contract savings on next billing cycle.
        """
    else:
        prob_class = "prob-low"
        risk_class = "risk-low"
        status_text = "Low Risk / Account Healthy"
        action_plan = """
        • Account is stable. No retention discounts required.<br>
        • Cross-sell premium streaming or hardware add-on packages.
        """

    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Churn Probability Score</div>
            <div class="prob-number {prob_class}">{churn_prob:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.progress(int(churn_prob) / 100)

    st.markdown(f"""
        <div class="risk-box {risk_class}">
            <div class="risk-title">{status_text}</div>
            <div class="risk-desc">
                <strong>Recommended Retention Strategy:</strong><br>
                {action_plan}
            </div>
        </div>
    """, unsafe_allow_html=True)