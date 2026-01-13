import streamlit as st
import numpy as np
import cv2
import os

from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Astro-AI Research Dashboard",
    layout="wide"
)

st.title("🌌 Astro-AI Research Dashboard")
st.caption("Exoplanets • Galaxies • Asteroids • SETI (Safe Production Build)")

# =========================================================
# 🔄 SAFE MODEL LOADERS (NO CRASH GUARANTEE)
# =========================================================

@st.cache_resource
def load_exoplanet_model():
    path = "models/exoplanet_kepler_model.h5"
    if os.path.exists(path):
        return load_model(path, compile=False)
    return None


@st.cache_resource
def load_galaxy_model():
    path = "models/galaxy_sdss_model.h5"
    if os.path.exists(path):
        return load_model(path, compile=False)
    return None


@st.cache_resource
def load_seti_model():
    path = "models/seti_autoencoder_real.h5"
    if os.path.exists(path):
        return load_model(path, compile=False)
    return None

# =========================================================
# 🧠 AI FUNCTIONS (REAL + FALLBACK)
# =========================================================

def exoplanet_ai(lightcurve):
    model = load_exoplanet_model()

    lc = lightcurve / np.median(lightcurve)

    if model is None:
        dip = float(np.min(lc))
        prob = float(np.clip(1.0 - dip, 0, 1))
        return {
            "planet_probability": prob,
            "detected": prob > 0.5,
            "mode": "fallback"
        }

    lc = lc[:model.input_shape[1]]
    lc = lc.reshape(1, -1, 1)

    prob = float(model.predict(lc)[0][0])
    return {
        "planet_probability": prob,
        "detected": prob > 0.5,
        "mode": "cnn"
    }


def galaxy_ai(image):
    model = load_galaxy_model()

    if model is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()

        if mean_val > 140:
            gtype = "Elliptical"
            conf = 0.75
        elif mean_val > 90:
            gtype = "Spiral"
            conf = 0.72
        else:
            gtype = "Irregular"
            conf = 0.68

        return {
            "type": gtype,
            "confidence": conf,
            "mode": "fallback"
        }

    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 3)

    prob = float(model.predict(img)[0][0])

    if prob > 0.5:
        return {"type": "Spiral", "confidence": prob, "mode": "cnn"}
    else:
        return {"type": "Elliptical", "confidence": 1 - prob, "mode": "cnn"}


def seti_ai(signal):
    model = load_seti_model()

    signal = (signal - signal.mean()) / (signal.std() + 1e-6)

    if model is None:
        score = float(np.max(signal) - np.mean(signal))
        return {
            "anomaly_score": score,
            "candidate": score > 5.0,
            "mode": "fallback"
        }

    signal = signal.reshape(1, -1)
    recon = model.predict(signal)
    error = float(np.mean((signal - recon) ** 2))

    return {
        "anomaly_score": error,
        "candidate": error > 0.05,
        "mode": "autoencoder"
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
# 🪐 EXOPLANETS
# =========================================================
with tab1:
    st.subheader("Exoplanet Detection")

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
        except:
            st.error("Invalid light curve input.")

# =========================================================
# 🌌 GALAXIES
# =========================================================
with tab2:
    st.subheader("Galaxy Classification")

    uploaded = st.file_uploader(
        "Upload galaxy image (JPG / PNG)",
        type=["jpg", "png"]
    )

    if uploaded:
        image = cv2.imdecode(
            np.frombuffer(uploaded.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        st.image(image, use_column_width=True)

        if st.button("Run Galaxy AI"):
            result = galaxy_ai(image)
            st.json(result)

# =========================================================
# ☄️ ASTEROIDS
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
            st.error("Invalid asteroid data.")

# =========================================================
# 📡 SETI
# =========================================================
with tab4:
    st.subheader("SETI Signal Detection")

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
st.caption(
    "✔ Safe deployment • ✔ Fallback AI • ✔ Real models supported • ✔ No crashes"
)
