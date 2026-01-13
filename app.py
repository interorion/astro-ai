import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Astro-AI Research Dashboard",
    layout="wide"
)

st.title("🌌 Astro-AI Research Dashboard")
st.caption("Real Telescope AI • Exoplanets • Galaxies • SETI")

# =========================================================
# 🔄 LAZY LOAD MODELS (CRITICAL FOR STABILITY)
# =========================================================

@st.cache_resource
def load_exoplanet_model():
    return load_model("models/exoplanet_kepler_model.h5", compile=False)

@st.cache_resource
def load_galaxy_model():
    return load_model("models/galaxy_sdss_model.h5", compile=False)

@st.cache_resource
def load_seti_model():
    return load_model("models/seti_autoencoder_real.h5", compile=False)

# =========================================================
# 🧠 AI FUNCTIONS
# =========================================================

def exoplanet_ai(lightcurve):
    model = load_exoplanet_model()

    lc = lightcurve / np.median(lightcurve)
    lc = lc[:model.input_shape[1]]  # trim if longer
    lc = lc.reshape(1, -1, 1)

    prob = float(model.predict(lc)[0][0])
    return {
        "planet_probability": prob,
        "detected": prob > 0.5
    }

def galaxy_ai(image):
    model = load_galaxy_model()

    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)

    prob = float(model.predict(img)[0][0])

    if prob > 0.5:
        return {"type": "Spiral", "confidence": prob}
    else:
        return {"type": "Elliptical", "confidence": 1 - prob}

def seti_ai(signal):
    model = load_seti_model()

    signal = (signal - signal.mean()) / signal.std()
    signal = signal.reshape(1, -1)

    recon = model.predict(signal)
    error = float(np.mean((signal - recon) ** 2))

    return {
        "anomaly_score": error,
        "candidate": error > 0.05
    }

def asteroid_ai(times):
    velocity = np.gradient(times)
    stable = np.std(velocity) < 1.0

    return {
        "orbit_points": len(times),
        "orbit_stable": stable,
        "impact_risk": "Low" if stable else "Medium"
    }

# =========================================================
# UI TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["🪐 Exoplanets", "🌌 Galaxies", "☄️ Asteroids", "📡 SETI"]
)

# =========================================================
# 🪐 EXOPLANET TAB
# =========================================================
with tab1:
    st.subheader("Exoplanet Detection (Kepler CNN)")
    lc_text = st.text_area(
        "Paste normalized light curve (comma separated)",
        "1.0,0.99,0.98,0.97,0.99,1.0"
    )

    if st.button("Run Exoplanet AI"):
        try:
            lc = np.array([float(x.strip()) for x in lc_text.split(",") if x.strip()])
            result = exoplanet_ai(lc)

            st.json(result)
            st.line_chart(lc)
        except Exception as e:
            st.error("Invalid light curve format.")

# =========================================================
# 🌌 GALAXY TAB
# =========================================================
with tab2:
    st.subheader("Galaxy Classification (SDSS CNN)")
    uploaded = st.file_uploader(
        "Upload a galaxy image (JPG / PNG)",
        type=["jpg", "png"]
    )

    if uploaded:
        image = cv2.imdecode(
            np.frombuffer(uploaded.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        st.image(image, caption="Uploaded Galaxy", use_column_width=True)

        if st.button("Run Galaxy AI"):
            result = galaxy_ai(image)
            st.json(result)

# =========================================================
# ☄️ ASTEROID TAB
# =========================================================
with tab3:
    st.subheader("Asteroid Orbit Analysis")
    t_text = st.text_area(
        "Paste time values (comma separated)",
        "0,1,2,3,4,5,6"
    )

    if st.button("Run Asteroid AI"):
        try:
            times = np.array([float(x.strip()) for x in t_text.split(",") if x.strip()])
            result = asteroid_ai(times)

            st.json(result)
            st.line_chart(times)
        except:
            st.error("Invalid time values.")

# =========================================================
# 📡 SETI TAB
# =========================================================
with tab4:
    st.subheader("SETI Signal Anomaly Detection")
    s_text = st.text_area(
        "Paste signal vector (comma separated)",
        "0.1,0.2,0.1,6.5,6.7,0.1"
    )

    if st.button("Run SETI AI"):
        try:
            signal = np.array([float(x.strip()) for x in s_text.split(",") if x.strip()])
            result = seti_ai(signal)

            st.json(result)
            st.line_chart(signal)
        except:
            st.error("Invalid signal vector.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("🔬 Real telescope-trained AI • Kepler • SDSS • Breakthrough Listen")
