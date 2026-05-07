import streamlit as st
import pickle
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# =========================================================
# LOAD MODEL FILES
# =========================================================

@st.cache_resource
def load_files():

    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, scaler, label_encoder


model, scaler, le = load_files()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ---------------- MAIN BACKGROUND ---------------- */

.stApp {
    background: linear-gradient(
        135deg,
        #dbeafe 0%,
        #f0fdf4 50%,
        #dcfce7 100%
    );
    color: #111827;
}

/* ---------------- REMOVE DEFAULT STREAMLIT UI ---------------- */

header, footer {
    visibility: hidden;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none;
}

hr {
    display: none;
}

.block-container {
    padding-top: 2rem;
}

/* ---------------- TITLE ---------------- */

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #14532d;
    margin-bottom: 5px;
}

/* ---------------- SUBTITLE ---------------- */

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #374151;
    margin-bottom: 35px;
}

/* ---------------- SECTION TITLES ---------------- */

.section-title {
    color: #14532d;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* ---------------- METRIC CARDS ---------------- */

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #bbf7d0 0%,
        #dcfce7 100%
    );
}

/* ---------------- INPUT LABELS ---------------- */

label {
    font-weight: 600 !important;
    color: #111827 !important;
}

/* ---------------- INPUT BOXES ---------------- */

.stNumberInput input {
    border-radius: 12px !important;
}

/* ---------------- BUTTON ---------------- */

.stButton > button {
    width: 100%;
    height: 3.5em;
    border: none;
    border-radius: 16px;
    background: linear-gradient(
        to right,
        #22c55e,
        #16a34a
    );
    color: white;
    font-size: 22px;
    font-weight: bold;
    transition: 0.25s ease;
    box-shadow: 0 6px 16px rgba(34,197,94,0.35);
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(
        to right,
        #16a34a,
        #15803d
    );
    color: white;
}

/* ---------------- RESULT BOX ---------------- */

.prediction-box {
    background: linear-gradient(
        135deg,
        #22c55e,
        #16a34a
    );
    padding: 28px;
    border-radius: 22px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    margin-top: 25px;
    box-shadow: 0 10px 24px rgba(34,197,94,0.35);
}

/* ---------------- CONFIDENCE BOX ---------------- */

.conf-box {
    background: white;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    color: #2563eb;
    margin-top: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

/* ---------------- FOOTER ---------------- */

.stCaption {
    text-align: center;
    color: #4b5563;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌾 About Project")

st.sidebar.info("""
This ML application recommends the best crop
based on soil nutrients and environmental conditions.

### Models Used
- Random Forest
- ANN

### Best Model
🌲 Random Forest

### Accuracy
99.31%
""")

st.sidebar.success("Built using Streamlit + Scikit-learn")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🌱 Crop Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered crop prediction using Machine Learning</div>',
    unsafe_allow_html=True
)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric(
        label="Random Forest Accuracy",
        value="99.31%"
    )

with metric_col2:
    st.metric(
        label="ANN Accuracy",
        value="97.95%"
    )

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">🧪 Enter Soil & Weather Parameters</div>',
    unsafe_allow_html=True
)

input_col1, input_col2 = st.columns(2)

with input_col1:

    N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)

    P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)

    K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)

    temperature = st.number_input("Temperature (°C)")

with input_col2:

    humidity = st.number_input("Humidity (%)")

    ph = st.number_input("pH Value")

    rainfall = st.number_input("Rainfall (mm)")

# =========================================================
# PREDICTION
# =========================================================

if st.button("🌾 Predict Crop"):

    input_data = np.array([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    crop = le.inverse_transform(prediction)[0]

    confidence = np.max(
        model.predict_proba(scaled_data)
    ) * 100

    # Prediction Result
    st.markdown(
        f"""
        <div class="prediction-box">
            Recommended Crop: {crop} 🌾
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence
    st.markdown(
        f"""
        <div class="conf-box">
            🎯 Prediction Confidence: {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.caption(
    "Machine Learning Based Crop Recommendation System   |   Built using Streamlit"
)


