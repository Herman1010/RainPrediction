import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# -------------------------
# CONFIG PAGE
# -------------------------
st.set_page_config(
    page_title="Weather AI Pro",
    page_icon="🌦️",
    layout="wide"
)

# -------------------------
# LOAD MODEL 
# -------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("data/weatherAUS_ForModel.csv")
    X = df.drop(columns=["RainTomorrow", "Date", "SeasonName"], errors="ignore")
    y = df["RainTomorrow"]

    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)

    return model, X

model, X = load_model()

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("🌡️ Paramètres météo")

humidity = st.sidebar.slider("💧 Humidité (%)", 0.0, 100.0, 50.0)
wind = st.sidebar.slider("🌬️ Vent (km/h)", 0.0, 100.0, 30.0)
pressure = st.sidebar.slider("🌡️ Pression (hPa)", 980.0, 1050.0, 1015.0)

# -------------------------
# INPUT USER
# -------------------------
input_data = X.iloc[0:1].copy()
input_data["Humidity3pm"] = humidity
input_data["WindGustSpeed"] = wind
input_data["Pressure9am"] = pressure

# -------------------------
# PREDICTION
# -------------------------
proba = model.predict_proba(input_data)[0][1]

# -------------------------
# BACKGROUND DYNAMIQUE
# -------------------------
if proba > 0.6:
    bg = "https://images.unsplash.com/photo-1501696461415-6bd6660c6742"
elif proba > 0.3:
    bg = "https://images.unsplash.com/photo-1504608524841-42fe6f032b4b"
else:
    bg = "https://images.unsplash.com/photo-1502082553048-f009c37129b9"

# -------------------------
# CSS 
# -------------------------
st.markdown(f"""
<style>

.stApp {{
    background: url("{bg}");
    background-size: cover;
    background-attachment: fixed;
    color: white;
}}

/* TITRE */
.title {{
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    text-shadow: 2px 2px 10px black;
}}

/* SUBTITLE */
.subtitle {{
    text-align: center;
    font-size: 18px;
    color: #e0e0e0;
}}

/* SIDEBAR BACKGROUND */
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.7);
}}

/* TEXTE SIDEBAR BLANC */
[data-testid="stSidebar"] * {{
    color: white !important;
}}

/* SLIDERS STYLE */
.stSlider > div > div {{
    background-color: rgba(255,255,255,0.3);
}}

.stSlider > div > div > div {{
    background-color: #ff4b4b;
}}

/* CARDS */
.card {{
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
    text-align: center;
    transition: transform 0.3s;
}}

.card:hover {{
    transform: scale(1.05);
}}

/* RESULT CARD */
.result {{
    padding: 35px;
    border-radius: 25px;
    background: rgba(0,0,0,0.5);
    text-align: center;
    backdrop-filter: blur(10px);
}}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown('<p class="title">🌦️ Weather AI Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prédiction intelligente de la pluie</p>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# CARDS METEO
# -------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="card">💧 Humidité<br><h2>{humidity}%</h2></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="card">🌬️ Vent<br><h2>{wind} km/h</h2></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="card">🌡️ Pression<br><h2>{pressure} hPa</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# RESULTAT
# -------------------------
st.markdown("## 🎯 Prévision météo")

st.markdown('<div class="result">', unsafe_allow_html=True)

st.markdown(f"<h1>{proba:.2%}</h1>", unsafe_allow_html=True)

if proba > 0.7:
    st.markdown("🌧️ <b>Forte pluie attendue</b>", unsafe_allow_html=True)
elif proba > 0.4:
    st.markdown("🌥️ <b>Risque de pluie</b>", unsafe_allow_html=True)
else:
    st.markdown("☀️ <b>Temps clair</b>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PROGRESS BAR
# -------------------------
st.progress(float(proba))

# -------------------------
# ANALYSE IA
# -------------------------
st.markdown("## 🧠 Analyse météo")

if humidity > 70:
    st.info("💧 Humidité élevée → favorise la pluie")
if pressure < 1005:
    st.warning("🌡️ Pression basse → conditions instables")
if wind > 50:
    st.warning("🌬️ Vent fort → perturbations possibles")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown(" Projet IA Application- CY Tech | Nour Ben Miled|Herman Sessou")