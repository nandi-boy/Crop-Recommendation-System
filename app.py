import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

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

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #F8FAF8; color: #1C1C1E; }

header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden; display: none; }

.block-container { padding: 2.5rem 2rem 3rem; max-width: 1100px; }

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8EDE8;
}
section[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

.sidebar-card {
    background: #F0F7F0;
    border: 1px solid #C8E0C8;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.sidebar-card h4 {
    margin: 0 0 5px;
    font-size: 11px; font-weight: 700; color: #5A8A5A;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.sidebar-card p { margin: 0; font-size: 14px; font-weight: 600; color: #1C3A1C; }

.page-header {
    background: #FFFFFF;
    border: 1px solid #E4EBE4;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 26px;
    display: flex; align-items: center; gap: 20px;
}
.header-icon { font-size: 48px; line-height: 1; }
.header-text h1 { margin: 0 0 5px; font-size: 28px; font-weight: 800; color: #1C3A1C; }
.header-text p  { margin: 0; font-size: 14px; color: #6B8A6B; }

.section-label {
    font-size: 11px; font-weight: 700; color: #6B8A6B;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 14px;
}

.acc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 26px; }
.acc-card { background: #fff; border: 1px solid #E4EBE4; border-radius: 16px; padding: 18px 20px; }
.acc-card .model-name { font-size: 11px; font-weight: 700; color: #6B8A6B; margin-bottom: 5px; }
.acc-card .acc-value  { font-size: 30px; font-weight: 800; color: #1C3A1C; line-height: 1; }
.acc-bar-bg  { height: 6px; background: #E8F0E8; border-radius: 999px; margin-top: 10px; overflow: hidden; }
.acc-bar-fill{ height: 100%; border-radius: 999px; }

/* ── Colorful field wrappers ── */
.field-wrap {
    border-radius: 14px;
    padding: 14px 16px;
    border: 2px solid;
    margin-bottom: 8px;
}
.field-n  { background: #EEF3FF; border-color: #7B9EFF; }
.field-p  { background: #FFF0F6; border-color: #F070A8; }
.field-k  { background: #FFFBE6; border-color: #F5C842; }
.field-t  { background: #FFF3EE; border-color: #FF8C5A; }
.field-h  { background: #EDFAFF; border-color: #3EC6E8; }
.field-ph { background: #F3EEFF; border-color: #A97EF5; }
.field-r  { background: #E8FFF4; border-color: #3DD68C; }

.field-icon  { font-size: 20px; margin-bottom: 6px; display: block; }
.field-title {
    display: block; margin-bottom: 4px;
    font-size: 12px; font-weight: 700;
}
.field-n  .field-title { color: #3050CC; }
.field-p  .field-title { color: #B02070; }
.field-k  .field-title { color: #8A6A00; }
.field-t  .field-title { color: #B04010; }
.field-h  .field-title { color: #006E8A; }
.field-ph .field-title { color: #5A30A0; }
.field-r  .field-title { color: #0A8A4A; }

.hint-text { font-size: 10px; font-weight: 500; margin-top: 5px; opacity: 0.72; }
.field-n  .hint-text { color: #3050CC; }
.field-p  .hint-text { color: #B02070; }
.field-k  .hint-text { color: #8A6A00; }
.field-t  .hint-text { color: #B04010; }
.field-h  .hint-text { color: #006E8A; }
.field-ph .hint-text { color: #5A30A0; }
.field-r  .hint-text { color: #0A8A4A; }

/* Override Streamlit number inputs inside colored wrappers */
.stNumberInput input {
    border: none !important;
    border-radius: 9px !important;
    background: rgba(255,255,255,0.75) !important;
    font-size: 15px !important; font-weight: 600 !important;
    color: #1C1C1E !important;
    padding: 9px 12px !important;
}
.stNumberInput input:focus { outline: none !important; box-shadow: 0 0 0 2px rgba(0,0,0,0.15) !important; }

label { font-size: 13px !important; font-weight: 700 !important; color: #3A3A3A !important; }

/* ── 3D Predict Button ── */
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
    position: relative !important;
    top: 0px !important;
    box-shadow:
        0 2px 0 #1e8449,
        0 4px 0 #196f3d,
        0 6px 0 #145a32,
        0 7px 0 rgba(0,0,0,0.25),
        0 9px 14px rgba(39,174,96,0.30) !important;
    transition: top 0.08s ease, box-shadow 0.08s ease, background 0.1s !important;
}
.stButton > button:hover {
    background: #2ecc71 !important;
    color: #FFFFFF !important;
    top: -2px !important;
    box-shadow:
        0 4px 0 #1e8449,
        0 6px 0 #196f3d,
        0 8px 0 #145a32,
        0 9px 0 rgba(0,0,0,0.22),
        0 12px 18px rgba(39,174,96,0.35) !important;
}
.stButton > button:active {
    top: 5px !important;
    box-shadow:
        0 1px 0 #196f3d,
        0 2px 6px rgba(39,174,96,0.18) !important;
}

/* ── Result card ── */
.result-card {
    background: #fff;
    border: 2px solid #B8D8C0;
    border-radius: 20px;
    padding: 26px 30px;
    display: flex; align-items: center; gap: 22px;
    margin-top: 22px;
}
.result-icon-wrap {
    width: 70px; height: 70px; background: #EEF7F1;
    border-radius: 18px; display: flex; align-items: center;
    justify-content: center; font-size: 34px; flex-shrink: 0;
}
.result-label { font-size: 11px; font-weight: 700; color: #6B8A6B; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 3px; }
.result-crop  { font-size: 32px; font-weight: 800; color: #1C3A1C; margin-bottom: 5px; }
.result-conf  { font-size: 14px; color: #5A8A6A; font-weight: 600; }
.conf-track   { height: 8px; background: #E4EDE6; border-radius: 999px; overflow: hidden; margin-top: 9px; width: 240px; }
.conf-fill    { height: 100%; border-radius: 999px; background: #27AE60; }

.footer-text {
    text-align: center; font-size: 13px; color: #9AAE9A;
    margin-top: 40px; padding-top: 18px; border-top: 1px solid #E8EDE8;
}

</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🌱 About")
    st.markdown("""
    <div class="sidebar-card"><h4>What it does</h4><p>Recommends the best crop based on soil nutrients and environment.</p></div>
    <div class="sidebar-card"><h4>Best Model</h4><p>🌲 Random Forest</p></div>
    <div class="sidebar-card"><h4>Dataset</h4><p>2,200 samples · 22 crops</p></div>
    <div class="sidebar-card"><h4>Input Features</h4><p>N · P · K · Temp · Humidity · pH · Rainfall</p></div>
    """, unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="page-header">
  <div class="header-icon">🌾</div>
  <div class="header-text">
    <h1>Crop Recommendation System</h1>
    <p>Enter soil and climate parameters to get an AI-powered crop recommendation</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Model accuracy ──
st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
st.markdown("""
<div class="acc-grid">
  <div class="acc-card">
    <div class="model-name">🌲 Random Forest</div>
    <div class="acc-value">99.31%</div>
    <div class="acc-bar-bg"><div class="acc-bar-fill" style="width:99.31%;background:#4CAF7D;"></div></div>
  </div>
  <div class="acc-card">
    <div class="model-name">🧠 ANN (Neural Network)</div>
    <div class="acc-value">97.95%</div>
    <div class="acc-bar-bg"><div class="acc-bar-fill" style="width:97.95%;background:#6DBFA0;"></div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ──
st.markdown('<div class="section-label">Soil & Climate Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="field-wrap field-n"><span class="field-icon">🟦</span><span class="field-title">Nitrogen (N)</span>', unsafe_allow_html=True)
    N = st.number_input("N", min_value=0.0, max_value=140.0, step=1.0, label_visibility="collapsed")
    st.markdown('<div class="hint-text">Range: 0 – 140</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="field-wrap field-p"><span class="field-icon">🟪</span><span class="field-title">Phosphorus (P)</span>', unsafe_allow_html=True)
    P = st.number_input("P", min_value=0.0, max_value=145.0, step=1.0, label_visibility="collapsed")
    st.markdown('<div class="hint-text">Range: 5 – 145</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="field-wrap field-k"><span class="field-icon">🟨</span><span class="field-title">Potassium (K)</span>', unsafe_allow_html=True)
    K = st.number_input("K", min_value=0.0, max_value=205.0, step=1.0, label_visibility="collapsed")
    st.markdown('<div class="hint-text">Range: 5 – 205</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="field-wrap field-t"><span class="field-icon">🌡️</span><span class="field-title">Temperature (°C)</span>', unsafe_allow_html=True)
    temperature = st.number_input("Temp", min_value=0.0, max_value=50.0, step=0.1, label_visibility="collapsed")
    st.markdown('<div class="hint-text">Range: 8 – 44 °C</div></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="field-wrap field-h"><span class="field-icon">💧</span><span class="field-title">Humidity (%)</span>', unsafe_allow_html=True)
    humidity = st.number_input("Hum", min_value=0.0, max_value=100.0, step=0.1, label_visibility="collapsed")
    st.markdown('<div class="hint-text">Range: 14 – 100 %</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="field-wrap field-ph"><span class="field-icon">🧪</span><span class="field-title">Soil pH</span>', unsafe_allow_html=True)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, label_visibility="collapsed")
    st.markdown('<div class="hint-text">Range: 3.5 – 10.0</div></div>', unsafe_allow_html=True)

col_r, col_gap = st.columns([1, 2])
with col_r:
    st.markdown('<div class="field-wrap field-r"><span class="field-icon">🌧️</span><span class="field-title">Rainfall (mm)</span>', unsafe_allow_html=True)
    rainfall = st.number_input("Rain", min_value=0.0, max_value=400.0, step=1.0, label_visibility="collapsed")
    st.markdown('<div class="hint-text">Range: 20 – 299 mm</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_btn, _ = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("🌾   Predict Best Crop")

# ── Result ──
CROP_EMOJI = {
    "rice":"🌾","maize":"🌽","chickpea":"🫘","kidneybeans":"🫘",
    "pigeonpeas":"🫘","mothbeans":"🌿","mungbean":"🫘","blackgram":"🫘",
    "lentil":"🫘","pomegranate":"🍎","banana":"🍌","mango":"🥭",
    "grapes":"🍇","watermelon":"🍉","muskmelon":"🍈","apple":"🍎",
    "orange":"🍊","papaya":"🍈","coconut":"🥥","cotton":"🌸",
    "jute":"🌿","coffee":"☕",
}

if predict_clicked:
    input_data  = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_data = scaler.transform(input_data)
    prediction  = model.predict(scaled_data)
    crop        = le.inverse_transform(prediction)[0]
    confidence  = float(np.max(model.predict_proba(scaled_data)) * 100)
    emoji       = CROP_EMOJI.get(crop.lower(), "🌱")

    st.markdown(f"""
    <div class="result-card">
      <div class="result-icon-wrap">{emoji}</div>
      <div>
        <div class="result-label">Recommended Crop</div>
        <div class="result-crop">{crop.capitalize()}</div>
        <div class="result-conf">Prediction confidence: <strong>{confidence:.2f}%</strong></div>
        <div class="conf-track"><div class="conf-fill" style="width:{confidence:.1f}%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
  Crop Recommendation System &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
