import streamlit as st
import pickle
import numpy as np
import pandas as pd

from locations import get_countries, get_states
from crop_history import get_historical_crops, get_districts_for_state
from crop_care import get_crop_care
from translation import LANGUAGES, translate_text, translate_list


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-Powered Crop Advisory System",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

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


# ============================================================
# DEFAULT SOIL VALUES (for State-Based mode)
# ============================================================
# We don't have a real soil sensor for State-Based mode, so N/P/K/pH
# are filled in using the training dataset's own average values.
# Computed live from the CSV (not hardcoded) so it stays correct if the
# dataset is ever updated. Cached so it only runs once per session.

@st.cache_data
def get_default_soil():

    df = pd.read_csv("dataset/Crop_recommendation.csv")

    return {
        "N": float(df["N"].mean()),
        "P": float(df["P"].mean()),
        "K": float(df["K"].mean()),
        "ph": float(df["ph"].mean()),
    }


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F8FAF8;
    color: #1C1C1E;
}

header,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    visibility: hidden;
    display: none;
}

.block-container {
    padding: 2.5rem 2rem 3rem;
    max-width: 1100px;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8EDE8;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem;
}

.sidebar-card {
    background: #F0F7F0;
    border: 1px solid #C8E0C8;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.sidebar-card h4 {
    margin: 0 0 5px;
    font-size: 11px;
    font-weight: 700;
    color: #5A8A5A;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.sidebar-card p {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #1C3A1C;
}


/* ============================================================
   HEADER
   ============================================================ */

.page-header {
    background: #FFFFFF;
    border: 1px solid #E4EBE4;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 26px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.header-icon {
    font-size: 48px;
    line-height: 1;
}

.header-text h1 {
    margin: 0 0 5px;
    font-size: 28px;
    font-weight: 800;
    color: #1C3A1C;
}

.header-text p {
    margin: 0;
    font-size: 14px;
    color: #6B8A6B;
}


/* ============================================================
   SECTION LABEL
   ============================================================ */

.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #6B8A6B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
}


/* ============================================================
   MODEL PERFORMANCE
   ============================================================ */

.acc-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 26px;
}

.acc-card {
    background: #FFFFFF;
    border: 1px solid #E4EBE4;
    border-radius: 16px;
    padding: 18px 20px;
}

.acc-card .model-name {
    font-size: 11px;
    font-weight: 700;
    color: #6B8A6B;
    margin-bottom: 5px;
}

.acc-card .acc-value {
    font-size: 30px;
    font-weight: 800;
    color: #1C3A1C;
    line-height: 1;
}

.acc-bar-bg {
    height: 6px;
    background: #E8F0E8;
    border-radius: 999px;
    margin-top: 10px;
    overflow: hidden;
}

.acc-bar-fill {
    height: 100%;
    border-radius: 999px;
}


/* ============================================================
   INPUT CARDS
   ============================================================ */

.field-wrap {
    border-radius: 14px;
    padding: 14px 16px;
    border: 2px solid;
    margin-bottom: 8px;
}

.field-n {
    background: #EEF3FF;
    border-color: #7B9EFF;
}

.field-p {
    background: #FFF0F6;
    border-color: #F070A8;
}

.field-k {
    background: #FFFBE6;
    border-color: #F5C842;
}

.field-t {
    background: #FFF3EE;
    border-color: #FF8C5A;
}

.field-h {
    background: #EDFAFF;
    border-color: #3EC6E8;
}

.field-ph {
    background: #F3EEFF;
    border-color: #A97EF5;
}

.field-r {
    background: #E8FFF4;
    border-color: #3DD68C;
}

.field-icon {
    font-size: 20px;
    margin-bottom: 6px;
    display: block;
}

.field-title {
    display: block;
    margin-bottom: 4px;
    font-size: 12px;
    font-weight: 700;
}

.field-n .field-title {
    color: #3050CC;
}

.field-p .field-title {
    color: #B02070;
}

.field-k .field-title {
    color: #8A6A00;
}

.field-t .field-title {
    color: #B04010;
}

.field-h .field-title {
    color: #006E8A;
}

.field-ph .field-title {
    color: #5A30A0;
}

.field-r .field-title {
    color: #0A8A4A;
}

.hint-text {
    font-size: 10px;
    font-weight: 500;
    margin-top: 5px;
    opacity: 0.72;
}


/* ============================================================
   STREAMLIT INPUT
   ============================================================ */

.stNumberInput input {
    border: none !important;
    border-radius: 9px !important;
    background: rgba(255, 255, 255, 0.75) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1C1C1E !important;
    padding: 9px 12px !important;
}

.stNumberInput input:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.15) !important;
}

label {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #3A3A3A !important;
}


/* ============================================================
   SELECT BOX
   ============================================================ */

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    background-color: #F0F2F6 !important;
    border: none !important;
    min-height: 52px !important;
}

div[data-baseweb="select"] span {
    font-size: 16px !important;
}


/* ============================================================
   PREDICT BUTTON
   ============================================================ */

.stButton > button {
    width: 100% !important;
    height: 62px !important;

    border: none !important;
    border-radius: 16px !important;

    background: #27AE60 !important;
    color: #FFFFFF !important;

    font-size: 18px !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;

    cursor: pointer !important;

    box-shadow:
        0 2px 0 #1E8449,
        0 4px 0 #196F3D,
        0 6px 0 #145A32,
        0 7px 0 rgba(0,0,0,0.25),
        0 9px 14px rgba(39,174,96,0.30) !important;

    transition:
        top 0.08s ease,
        box-shadow 0.08s ease,
        background 0.1s !important;
}

.stButton > button:hover {
    background: #2ECC71 !important;
    color: #FFFFFF !important;
    top: -2px !important;

    box-shadow:
        0 4px 0 #1E8449,
        0 6px 0 #196F3D,
        0 8px 0 #145A32,
        0 9px 0 rgba(0,0,0,0.22),
        0 12px 18px rgba(39,174,96,0.35) !important;
}

.stButton > button:active {
    top: 5px !important;

    box-shadow:
        0 1px 0 #196F3D,
        0 2px 6px rgba(39,174,96,0.18) !important;
}


/* ============================================================
   RESULT CARD
   ============================================================ */

.result-card {
    background: #FFFFFF;
    border: 2px solid #B8D8C0;
    border-radius: 20px;
    padding: 26px 30px;

    display: flex;
    align-items: center;

    gap: 22px;

    margin-top: 22px;
}

.result-icon-wrap {
    width: 70px;
    height: 70px;

    background: #EEF7F1;

    border-radius: 18px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 34px;

    flex-shrink: 0;
}

.result-label {
    font-size: 11px;
    font-weight: 700;

    color: #6B8A6B;

    text-transform: uppercase;
    letter-spacing: 0.07em;

    margin-bottom: 3px;
}

.result-crop {
    font-size: 32px;
    font-weight: 800;

    color: #1C3A1C;

    margin-bottom: 5px;
}

.result-conf {
    font-size: 14px;
    color: #5A8A6A;
    font-weight: 600;
}


/* ============================================================
   RANKED RESULTS (State-Based, top-3)
   ============================================================ */

.rank-card {
    background: #FFFFFF;
    border: 2px solid #E4EBE4;
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 12px;
}

.rank-card.rank-1 {
    border-color: #B8D8C0;
}

.rank-badge {
    width: 34px;
    height: 34px;
    border-radius: 999px;
    background: #EEF7F1;
    color: #1C3A1C;
    font-weight: 800;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.rank-emoji {
    font-size: 26px;
    flex-shrink: 0;
}

.rank-crop {
    font-size: 18px;
    font-weight: 800;
    color: #1C3A1C;
}

.rank-conf-track {
    height: 6px;
    background: #E4EDE6;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 6px;
    width: 180px;
}

.rank-conf-fill {
    height: 100%;
    border-radius: 999px;
    background: #27AE60;
}

.low-confidence-warning {
    background: #FFF8E6;
    border: 1px solid #F0D98C;
    border-radius: 14px;
    padding: 16px 18px;
    margin-top: 18px;
    font-size: 13px;
    color: #6B5A1C;
    line-height: 1.6;
}

.conf-track {
    height: 8px;

    background: #E4EDE6;

    border-radius: 999px;

    overflow: hidden;

    margin-top: 9px;

    width: 240px;
}

.conf-fill {
    height: 100%;

    border-radius: 999px;

    background: #27AE60;
}


/* ============================================================
   STATE BASED
   ============================================================ */

.state-card {
    background: #FFFFFF;

    border: 1px solid #E4EBE4;

    border-radius: 18px;

    padding: 24px;

    margin-top: 8px;
}

.state-card h3 {
    color: #1C3A1C;
    margin-bottom: 10px;
}

.state-card p {
    color: #6B8A6B;
    font-size: 14px;
}

.climate-card {
    background: #F0F7F0;

    border: 1px solid #C8E0C8;

    border-radius: 16px;

    padding: 20px;

    margin-top: 20px;
}

.climate-card h3 {
    color: #1C3A1C;
    margin-top: 0;
}

.climate-card p {
    color: #6B8A6B;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer-text {
    text-align: center;

    font-size: 13px;

    color: #9AAE9A;

    margin-top: 40px;

    padding-top: 18px;

    border-top: 1px solid #E8EDE8;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### 🌱 About")

    st.markdown(
        """
<div class="sidebar-card">

<h4>What it does</h4>

<p>
Recommends the best crop based on
soil nutrients and environmental conditions.
</p>

</div>


<div class="sidebar-card">

<h4>Best Model</h4>

<p>
🌲 Random Forest
</p>

</div>


<div class="sidebar-card">

<h4>Dataset</h4>

<p>
2,200 samples · 22 crops
</p>

</div>


<div class="sidebar-card">

<h4>Input Features</h4>

<p>
N · P · K · Temp · Humidity · pH · Rainfall
</p>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
<div class="page-header">

<div class="header-icon">
🌾
</div>

<div class="header-text">

<h1>
Krishi Sahayak — AI-Powered Crop Advisory System
</h1>

<p>
Enter soil and climate parameters or select a location
for crop recommendation
</p>

</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# RECOMMENDATION METHOD
# ============================================================

st.markdown(
    '<div class="section-label">Advisory Method</div>',
    unsafe_allow_html=True
)


# IMPORTANT:
# No HTML wrapper around radio.
# This prevents the empty white box.

mode = st.radio(
    "Choose recommendation method",

    [
        "🌾 Parameter-Based",
        "🌍 State-Based",
        "🌱 Crop Care Guide"
    ],

    horizontal=True,

    label_visibility="collapsed"
)


# ============================================================
# CROP EMOJIS
# ============================================================

CROP_EMOJI = {

    "rice": "🌾",

    "maize": "🌽",

    "chickpea": "🫘",

    "kidneybeans": "🫘",

    "pigeonpeas": "🫘",

    "mothbeans": "🌿",

    "mungbean": "🫘",

    "blackgram": "🫘",

    "lentil": "🫘",

    "pomegranate": "🍎",

    "banana": "🍌",

    "mango": "🥭",

    "grapes": "🍇",

    "watermelon": "🍉",

    "muskmelon": "🍈",

    "apple": "🍎",

    "orange": "🍊",

    "papaya": "🍈",

    "coconut": "🥥",

    "cotton": "🌸",

    "jute": "🌿",

    "coffee": "☕"
}


# ============================================================
# ============================================================
#
# PARAMETER-BASED MODE
#
# ============================================================
# ============================================================

if mode == "🌾 Parameter-Based":


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-label">Model Performance</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
<div class="acc-grid">


<div class="acc-card">

<div class="model-name">
🌲 Random Forest
</div>

<div class="acc-value">
99.31%
</div>

<div class="acc-bar-bg">

<div
class="acc-bar-fill"
style="width:99.31%;background:#4CAF7D;">
</div>

</div>

</div>


<div class="acc-card">

<div class="model-name">
🧠 ANN (Neural Network)
</div>

<div class="acc-value">
97.95%
</div>

<div class="acc-bar-bg">

<div
class="acc-bar-fill"
style="width:97.95%;background:#6DBFA0;">
</div>

</div>

</div>


</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # SOIL & CLIMATE PARAMETERS
    # ========================================================

    st.markdown(
        '<div class="section-label">Soil & Climate Parameters</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    # ========================================================
    # COLUMN 1
    # ========================================================

    with col1:


        # ----------------------------------------------------
        # NITROGEN
        # ----------------------------------------------------

        st.markdown(
            """
<div class="field-wrap field-n">

<span class="field-icon">
🟦
</span>

<span class="field-title">
Nitrogen (N)
</span>
""",
            unsafe_allow_html=True
        )


        N = st.number_input(
            "N",
            min_value=0.0,
            max_value=140.0,
            step=1.0,
            label_visibility="collapsed",
            key="N_input"
        )


        st.markdown(
            """
<div class="hint-text">
Range: 0 – 140
</div>

</div>
""",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # PHOSPHORUS
        # ----------------------------------------------------

        st.markdown(
            """
<div class="field-wrap field-p">

<span class="field-icon">
🟪
</span>

<span class="field-title">
Phosphorus (P)
</span>
""",
            unsafe_allow_html=True
        )


        P = st.number_input(
            "P",
            min_value=0.0,
            max_value=145.0,
            step=1.0,
            label_visibility="collapsed",
            key="P_input"
        )


        st.markdown(
            """
<div class="hint-text">
Range: 5 – 145
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # COLUMN 2
    # ========================================================

    with col2:


        # ----------------------------------------------------
        # POTASSIUM
        # ----------------------------------------------------

        st.markdown(
            """
<div class="field-wrap field-k">

<span class="field-icon">
🟨
</span>

<span class="field-title">
Potassium (K)
</span>
""",
            unsafe_allow_html=True
        )


        K = st.number_input(
            "K",
            min_value=0.0,
            max_value=205.0,
            step=1.0,
            label_visibility="collapsed",
            key="K_input"
        )


        st.markdown(
            """
<div class="hint-text">
Range: 5 – 205
</div>

</div>
""",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        st.markdown(
            """
<div class="field-wrap field-t">

<span class="field-icon">
🌡️
</span>

<span class="field-title">
Temperature (°C)
</span>
""",
            unsafe_allow_html=True
        )


        temperature = st.number_input(
            "Temp",
            min_value=0.0,
            max_value=50.0,
            step=0.1,
            label_visibility="collapsed",
            key="temperature_input"
        )


        st.markdown(
            """
<div class="hint-text">
Range: 8 – 44 °C
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # COLUMN 3
    # ========================================================

    with col3:


        # ----------------------------------------------------
        # HUMIDITY
        # ----------------------------------------------------

        st.markdown(
            """
<div class="field-wrap field-h">

<span class="field-icon">
💧
</span>

<span class="field-title">
Humidity (%)
</span>
""",
            unsafe_allow_html=True
        )


        humidity = st.number_input(
            "Hum",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            label_visibility="collapsed",
            key="humidity_input"
        )


        st.markdown(
            """
<div class="hint-text">
Range: 14 – 100 %
</div>

</div>
""",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # PH
        # ----------------------------------------------------

        st.markdown(
            """
<div class="field-wrap field-ph">

<span class="field-icon">
🧪
</span>

<span class="field-title">
Soil pH
</span>
""",
            unsafe_allow_html=True
        )


        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            step=0.1,
            label_visibility="collapsed",
            key="ph_input"
        )


        st.markdown(
            """
<div class="hint-text">
Range: 3.5 – 10.0
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # RAINFALL
    # ========================================================

    col_rain, col_empty = st.columns([1, 2])


    with col_rain:

        st.markdown(
            """
<div class="field-wrap field-r">

<span class="field-icon">
🌧️
</span>

<span class="field-title">
Rainfall (mm)
</span>
""",
            unsafe_allow_html=True
        )


        rainfall = st.number_input(
            "Rain",
            min_value=0.0,
            max_value=400.0,
            step=1.0,
            label_visibility="collapsed",
            key="rainfall_input"
        )


        st.markdown(
            """
<div class="hint-text">
Range: 20 – 299 mm
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    col_button, col_empty = st.columns([1, 2])


    with col_button:

        predict_clicked = st.button(
            "🌾   Predict Best Crop",
            key="parameter_predict"
        )


    # ========================================================
    # PARAMETER PREDICTION
    # ========================================================

    if predict_clicked:

        try:

            # ------------------------------------------------
            # CREATE INPUT
            # ------------------------------------------------

            input_data = np.array(
                [
                    [
                        N,
                        P,
                        K,
                        temperature,
                        humidity,
                        ph,
                        rainfall
                    ]
                ]
            )


            # ------------------------------------------------
            # SCALE
            # ------------------------------------------------

            scaled_data = scaler.transform(
                input_data
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                scaled_data
            )


            # ------------------------------------------------
            # DECODE
            # ------------------------------------------------

            crop = le.inverse_transform(
                prediction
            )[0]


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    scaled_data
                )

                confidence = float(
                    np.max(probabilities) * 100
                )


            # ------------------------------------------------
            # EMOJI
            # ------------------------------------------------

            emoji = CROP_EMOJI.get(
                str(crop).lower(),
                "🌱"
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if confidence is not None:

                st.markdown(
                    f"""
<div class="result-card">

<div class="result-icon-wrap">
{emoji}
</div>

<div>

<div class="result-label">
Recommended Crop
</div>

<div class="result-crop">
{str(crop).capitalize()}
</div>

<div class="result-conf">
Prediction confidence:
<strong>
{confidence:.2f}%
</strong>
</div>

<div class="conf-track">

<div
class="conf-fill"
style="width:{min(confidence,100):.1f}%;">
</div>

</div>

</div>

</div>
""",
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
<div class="result-card">

<div class="result-icon-wrap">
{emoji}
</div>

<div>

<div class="result-label">
Recommended Crop
</div>

<div class="result-crop">
{str(crop).capitalize()}
</div>

</div>

</div>
""",
                    unsafe_allow_html=True
                )


        except Exception as e:

            st.error(
                f"Prediction failed: {str(e)}"
            )


# ============================================================
# ============================================================
#
# STATE-BASED MODE
#
# ============================================================
# ============================================================

elif mode == "🌍 State-Based":


    # ========================================================
    # LOCATION
    # ========================================================

    st.markdown(
        '<div class="section-label">Location</div>',
        unsafe_allow_html=True
    )


    col_country, col_state = st.columns(2)


    # ========================================================
    # COUNTRY
    # ========================================================

    with col_country:

        country = st.selectbox(
            "Country",

            options=get_countries(),

            key="state_country"
        )


    # ========================================================
    # STATE
    # ========================================================

    with col_state:

        state_options = get_states(
            country
        )

        state = st.selectbox(
            "State",

            options=state_options,

            key="state_state"
        )


    # ========================================================
    # DISTRICT
    # ========================================================
    # Populated from the REAL district names in the downloaded
    # dataset (exact spelling as the government uses it) — no typos,
    # no guessing. Falls back to free-text only if the dataset hasn't
    # been downloaded yet, so the app doesn't hard-block before setup
    # is complete.

    available_districts = get_districts_for_state(state)

    if available_districts:

        district = st.selectbox(
            "District ",
            options=available_districts,
            key="state_district_select"
        )

    else:

        st.warning(
            "Local crop history dataset not found yet (or no districts "
            "matched this state) — showing a text field instead. Download "
            "it from Kaggle (search 'Crop Production in India' by "
            "abhinand05) or data.gov.in, save it as "
            "dataset/district_season_crop_production.csv, then reload."
        )

        district = st.text_input(
            "District ",
            placeholder="e.g. Bardhaman, Bankura, Nadia",
            key="state_district_text"
        )


    # ========================================================
    # SEASON
    # ========================================================

    season = st.radio(
        "Season ",
        options=["Kharif", "Rabi"],
        horizontal=True,
        key="state_season"
    )


    # ========================================================
    # SELECTED LOCATION
    # ========================================================

    st.markdown(
        f"""
<div class="state-card">

<h3>
🌾 Season-Based Recommendation
</h3>

<p>
Selected location: <strong>{country}</strong> → <strong>{state}</strong>
→ <strong>{district or "..."}</strong> · Season: <strong>{season}</strong>
</p>

<p>
This mode does NOT use the trained ML model. It ranks crops using
real historical government production statistics (data.gov.in,
downloaded locally) for this district and season — the same records
used to plan actual agricultural policy.
</p>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # BUTTON
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    col_button, col_empty = st.columns([1, 2])


    with col_button:

        history_button = st.button(
            "🌾   Get Season-Based Recommendation",
            key="state_predict"
        )


    # ========================================================
    # SEASON-BASED HISTORICAL LOOKUP
    # ========================================================

    if history_button:

        if not district.strip():

            st.error(
                "Please enter a district — it's required to look up "
                "real crop history for a specific farming region."
            )

        else:

            try:

                with st.spinner(f"Looking up {season} crop history for {district}, {state}..."):
                    results = get_historical_crops(
                        state=state,
                        district=district.strip(),
                        season=season,
                    )

                top3 = results[:3]
                total_all = sum(r["total"] for r in results)

                st.markdown(
                    '<div class="section-label" style="margin-top:22px;">Top 3 Historically Dominant Crops</div>',
                    unsafe_allow_html=True
                )

                for rank, r in enumerate(top3, start=1):
                    crop = r["crop"]
                    share = (r["total"] / total_all * 100) if total_all else 0
                    emoji = CROP_EMOJI.get(str(crop).lower(), "🌱")
                    rank_class = "rank-card rank-1" if rank == 1 else "rank-card"

                    st.markdown(
                        f"""
<div class="{rank_class}">
<div class="rank-badge">#{rank}</div>
<div class="rank-emoji">{emoji}</div>
<div>
<div class="rank-crop">{str(crop).capitalize()}</div>
<div class="result-conf">
{share:.1f}% of recorded {season} production &nbsp;·&nbsp;
{r['total']:,.0f} {r['unit']} total &nbsp;·&nbsp;
{r['years']} year(s) of data
</div>
<div class="rank-conf-track">
<div class="rank-conf-fill" style="width:{min(share,100):.1f}%;"></div>
</div>
</div>
</div>
""",
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""
<div class="low-confidence-warning">
ℹ️ This ranks crops by their actual historical share of recorded
{season} production/area in {district}, {state}, per data.gov.in.
It reflects what has really been grown there — not a soil/climate
prediction from the ML model.
</div>
""",
                    unsafe_allow_html=True
                )

            except RuntimeError as e:
                st.error(f"Couldn't fetch crop history: {e}")

            except Exception as e:
                st.error(f"Lookup failed: {str(e)}")


# ============================================================
# ============================================================
#
# CROP CARE GUIDE MODE
#
# ============================================================
# ============================================================

elif mode == "🌱 Crop Care Guide":


    # ========================================================
    # ABOUT THIS MODE
    # ========================================================

    st.markdown(
        """
<div class="state-card">

<h3>🌱 Crop Care Guide</h3>

<p>
Pick a crop to see its real required N, P, K, and temperature — the
same averages the ML model itself learned from — plus fertilizer
guidance derived from those numbers, and the general steps to farm it.
</p>

<p>
This is general reference information, not a soil-test-based
recommendation, and doesn't include pesticide/chemical dosage guidance.
</p>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # CROP + LANGUAGE SELECTION
    # ========================================================

    st.markdown(
        '<div class="section-label" style="margin-top:22px;">Choose a Crop</div>',
        unsafe_allow_html=True
    )

    care_crop_options = sorted(CROP_EMOJI.keys())

    col_crop, col_lang = st.columns(2)

    with col_crop:
        care_crop = st.selectbox(
            "Crop",
            options=care_crop_options,
            format_func=lambda c: f"{CROP_EMOJI.get(c,'🌱')} {c.capitalize()}",
            key="care_crop"
        )

    with col_lang:
        care_language = st.selectbox(
            "Language",
            options=list(LANGUAGES.keys()),
            key="care_language"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_button, col_empty = st.columns([1, 2])

    with col_button:
        care_button = st.button(
            "🌱   Show Care Guide",
            key="care_predict"
        )


    # ========================================================
    # CARE GUIDE RESULT
    # ========================================================

    if care_button:

        try:

            result = get_crop_care(care_crop)
            emoji = CROP_EMOJI.get(result["crop"], "🌱")
            lang_code = LANGUAGES[care_language]

            # Translate the descriptive text — not the crop name (a
            # dataset label / proper noun) and not the N/P/K chemical
            # symbols, which are standard notation in every language.
            t_required_conditions = translate_text("Required Conditions", lang_code)
            t_temperature = translate_text("Temperature", lang_code)
            t_levels_note = translate_text(
                "Levels (Low/Medium/High) are relative to the other 21 crops in the dataset, not fixed thresholds.",
                lang_code
            )
            t_n_level = translate_text(result["n_level"], lang_code)
            t_p_level = translate_text(result["p_level"], lang_code)
            t_k_level = translate_text(result["k_level"], lang_code)
            t_fertilizer_header = translate_text("Fertilizer Guidance", lang_code)
            t_process_header = translate_text("Farming Process", lang_code)
            t_fertilizer_lines = translate_list(result["fertilizer_guidance"], lang_code)
            t_process_lines = translate_list(result["process"], lang_code)
            t_disclaimer = translate_text(
                "N/P/K/temperature are real dataset averages, but your actual field will vary. "
                "Fertilizer guidance here is a general nutrient-to-fertilizer mapping, not a "
                "precise dose — a soil test from your local Krishi Vigyan Kendra (KVK) will "
                "always be more accurate.",
                lang_code
            )

            if lang_code != "en":
                st.caption(
                    "🌐 Machine-translated — cross-check against the English version if "
                    "anything looks unclear."
                )

            st.markdown(
                f"""
<div class="climate-card">

<h3>{emoji} {result['crop'].capitalize()} — {t_required_conditions}</h3>

<p>
<strong>Nitrogen (N):</strong> {result['N']:.1f} &nbsp;({t_n_level})
&nbsp;·&nbsp;
<strong>Phosphorus (P):</strong> {result['P']:.1f} &nbsp;({t_p_level})
&nbsp;·&nbsp;
<strong>Potassium (K):</strong> {result['K']:.1f} &nbsp;({t_k_level})
&nbsp;·&nbsp;
<strong>{t_temperature}:</strong> {result['temperature']:.1f} °C
</p>

<p style="font-size:12px;color:#6B8A6B;">
{t_levels_note}
</p>

</div>
""",
                unsafe_allow_html=True
            )

            col_fert, col_proc = st.columns(2)

            with col_fert:
                fert_html = "".join(f"<li>{line}</li>" for line in t_fertilizer_lines)
                st.markdown(
                    f"""
<div class="acc-card" style="min-height:220px;">
<div class="model-name">🌾 {t_fertilizer_header}</div>
<ul style="font-size:14px;color:#1C3A1C;line-height:1.7;margin-top:8px;padding-left:18px;">
{fert_html}
</ul>
</div>
""",
                    unsafe_allow_html=True
                )

            with col_proc:
                proc_html = "".join(f"<li>{step}</li>" for step in t_process_lines)
                st.markdown(
                    f"""
<div class="acc-card" style="min-height:220px;">
<div class="model-name">📋 {t_process_header}</div>
<ol style="font-size:14px;color:#1C3A1C;line-height:1.7;margin-top:8px;padding-left:18px;">
{proc_html}
</ol>
</div>
""",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
<div class="low-confidence-warning">
ℹ️ {t_disclaimer}
</div>
""",
                unsafe_allow_html=True
            )

        except RuntimeError as e:
            st.error(f"Couldn't load crop care info: {e}")

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")




# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer-text">

Crop Recommendation System
&nbsp;·&nbsp;
Built with Streamlit

</div>
""",
    unsafe_allow_html=True
)
