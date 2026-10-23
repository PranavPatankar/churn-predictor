import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Zomato / Swiggy Delivery ETA Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #18181B 0%, #27272A 100%);
        padding: 24px 30px;
        border-radius: 14px;
        border: 1px solid #3F3F46;
        margin-bottom: 25px;
    }
    .header-title {
        color: #E11D48;
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #A1A1AA;
        font-size: 14px;
        margin-top: 6px;
    }
    
    .custom-card {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-title {
        color: #A1A1AA;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 16px;
    }
    
    .data-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #27272A;
        font-size: 14px;
    }
    .data-row:last-child { border-bottom: none; }
    .data-label { color: #A1A1AA; font-weight: 500; }
    .data-val { color: #FAFAFA; font-weight: 600; }
    
    .eta-number {
        font-size: 52px;
        font-weight: 800;
        color: #FB7185;
        letter-spacing: -1px;
    }
    
    .alert-box {
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 15px;
        font-size: 14px;
        line-height: 1.5;
    }
    .alert-warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-left: 4px solid #F59E0B;
        color: #FDE68A;
    }
    .alert-normal {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-left: 4px solid #10B981;
        color: #A7F3D0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    return joblib.load('delivery_model.pkl')

try:
    model = load_trained_model()
except Exception:
    st.error("Model file 'delivery_model.pkl' not found. Please run `python train_model.py` first.")
    st.stop()


st.markdown("""
    <div class="header-container">
        <div class="header-title">Zomato / Swiggy Delivery ETA Engine</div>
        <div class="header-subtitle">Predictive Order Delivery Time Estimation using Machine Learning</div>
    </div>
""", unsafe_allow_html=True)


st.sidebar.markdown("### Order & Route Inputs")

distance_km = st.sidebar.slider("Delivery Distance (km)", min_value=1.0, max_value=20.0, value=5.5, step=0.5)
prep_time_min = st.sidebar.slider("Kitchen Prep Time (mins)", min_value=5, max_value=45, value=15, step=1)
courier_exp_yrs = st.sidebar.slider("Rider Experience (Years)", min_value=0.5, max_value=8.0, value=2.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Environmental Factors")

traffic_level = st.sidebar.selectbox("Traffic Condition", ["Low", "Medium", "High", "Jam"])
weather = st.sidebar.selectbox("Weather Condition", ["Sunny", "Rainy", "Foggy", "Stormy"])
vehicle_type = st.sidebar.selectbox("Rider Vehicle Type", ["Scooter", "Motorcycle", "Electric Bike"])


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
        <div class="custom-card">
            <div class="card-title">Order Dispatch Metadata</div>
            <div class="data-row"><span class="data-label">Delivery Distance</span><span class="data-val">{} km</span></div>
            <div class="data-row"><span class="data-label">Restaurant Kitchen Prep</span><span class="data-val">{} mins</span></div>
            <div class="data-row"><span class="data-label">Rider Experience</span><span class="data-val">{} yrs</span></div>
            <div class="data-row"><span class="data-label">Traffic Condition</span><span class="data-val">{}</span></div>
            <div class="data-row"><span class="data-label">Weather State</span><span class="data-val">{}</span></div>
            <div class="data-row"><span class="data-label">Vehicle Type</span><span class="data-val">{}</span></div>
        </div>
    """.format(
        distance_km, prep_time_min, courier_exp_yrs, 
        traffic_level, weather, vehicle_type
    ), unsafe_allow_html=True)

with col_right:
    input_df = pd.DataFrame([{
        'distance_km': distance_km,
        'prep_time_min': prep_time_min,
        'courier_exp_yrs': courier_exp_yrs,
        'weather': weather,
        'traffic_level': traffic_level,
        'vehicle_type': vehicle_type
    }])
    
    predicted_eta = model.predict(input_df)[0]
    
    st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Predicted Estimated Time of Arrival (ETA)</div>
            <div class="eta-number">{int(np.round(predicted_eta))} Mins</div>
        </div>
    """, unsafe_allow_html=True)
    
    if traffic_level in ['High', 'Jam'] or weather in ['Rainy', 'Stormy']:
        st.markdown(f"""
            <div class="alert-box alert-warning">
                <strong>Delayed Delivery Alert:</strong> Adverse environmental conditions detected ({weather} weather / {traffic_level} traffic). ETA reflects extra buffer time.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="alert-box alert-normal">
                <strong>Smooth Route Detected:</strong> Clear transit pathways. Rider estimated to deliver on schedule.
            </div>
        """, unsafe_allow_html=True)