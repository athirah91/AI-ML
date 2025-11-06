# =========================================================
# app.py — GSAD Data Science Hub (GitHub-Ready Version)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path

# Forecasting models
from prophet import Prophet
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Page config ---
st.set_page_config(page_title="GSAD Data Science Hub", layout="wide")

from streamlit_option_menu import option_menu
import base64

# =========================================================
# Path setup (Relative to repo)
# =========================================================
APP_DIR = Path(__file__).parent
ASSET_DIR = APP_DIR / "assets"
MODEL_DIR = APP_DIR / "models"
ART_DIR = APP_DIR / "artifacts" / "mkbf_prophet"

# Auto-create folders if missing (safeguard)
for folder in [ASSET_DIR, MODEL_DIR, ART_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================
# Shared helpers / loaders
# =========================================================
def get_base64_of_bin_file(bin_file):
    """Convert image to base64 for Streamlit background."""
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data(show_spinner=False)
def load_df_csv(path: Path, filename: str):
    f = path / filename
    if f.exists():
        return pd.read_csv(f, parse_dates=["ds"])
    return None

@st.cache_data(show_spinner=False)
def load_metrics_json(path: Path):
    f = path / "metrics.json"
    if f.exists():
        return json.loads(f.read_text())
    return None


# =========================================================
# Page: Home
# =========================================================
def page_home():
    file_path = ASSET_DIR / "qw3.jpg"
    if not file_path.exists():
        st.warning("⚠️ Background image not found in 'assets/qw3.jpg'")
        return

    bin_str = get_base64_of_bin_file(file_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-color: #0e0f14;
        }}
        .gradient-text {{
            font-size: 2.8rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(90deg, #a855f7, #ec4899, #fbbf24);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subheading {{
            text-align: center;
            font-size: 1.2rem;
            color: #cbd5e1;
            margin-top: -10px;
            margin-bottom: 30px;
        }}
        .frost-box {{
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            max-width: 900px;
            margin: auto;
            box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="frost-box">
            <h1 class="gradient-text">GSAD Data Science Hub</h1>
            <p class="subheading">Your central hub for forecasting and analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="frost-box" style="margin-top:30px; padding:15px;">
            <div style="font-size:1.1rem; font-weight:700; color:#f8fafc; margin-bottom:8px; margin-left:20px;">
                Module Available
            </div>
            <ul style="color:#cbd5e1; font-size:1rem; list-style-type:disc; margin-left:20px;">
                <li><b>MKBF Forecast</b> → View predictive charts and trends.</li>
                <li><b>Root Cause Classification</b> → Classify incident descriptions instantly.</li>
                <li><b>Forecasting App</b> → Upload any time series to explore & forecast.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Page: MKBF Forecast
# =========================================================
def page_mkbf():
    st.markdown("## <div style='text-align: center;'>MKBF Forecast</div>", unsafe_allow_html=True)

    try:
        actual = pd.read_csv(ART_DIR / "actual.csv")
    except FileNotFoundError:
        st.error(f"❌ Cannot find 'actual.csv' in {ART_DIR}. Make sure it exists in your repo.")
        st.stop()

    # Clean columns
    actual.columns = [c.strip() for c in actual.columns]
    if set(["ds", "y"]).issubset(actual.columns):
        actual = actual.rename(columns={"ds": "Date", "y": "Value"})
    elif not set(["Date", "Value"]).issubset(actual.columns):
        st.error("❌ 'actual.csv' must contain either ['Date','Value'] or ['ds','y'] columns.")
        return

    actual["Date"] = pd.to_datetime(actual["Date"], errors="coerce")
    actual = actual.dropna(subset=["Date", "Value"]).sort_values("Date")

    # Train Prophet
    st.info("🔁 Training Prophet model on the latest actual data...")
    df_train = actual.rename(columns={"Date": "ds", "Value": "y"})
    m = Prophet(
        changepoint_prior_scale=0.1,
        changepoint_range=0.95,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_prior_scale=10.0,
    )
    m.fit(df_train)

    horizon = st.slider("Forecast horizon (months ahead)", 6, 36, 12)
    future = m.make_future_dataframe(periods=horizon, freq="MS")
    forecast = m.predict(future)

    # Combine
    combined = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(df_train, on="ds", how="left")
    combined = combined.rename(
        columns={
            "ds": "Date",
            "y": "Actual",
            "yhat": "Forecast",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound",
        }
    )

    # Plot
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined["Date"], y=combined["Forecast"], mode="lines", name="Forecast", line=dict(width=3)))
    fig.add_trace(
        go.Scatter(
            x=pd.concat([combined["Date"], combined["Date"][::-1]]),
            y=pd.concat([combined["Upper Bound"], combined["Lower Bound"][::-1]]),
            fill="toself",
            fillcolor="rgba(135,206,235,0.25)",
            line=dict(width=0),
            name="Confidence",
        )
    )
    fig.add_trace(go.Scatter(x=combined["Date"], y=combined["Actual"], mode="lines+markers", name="Actual", line=dict(width=2)))
    fig.update_layout(title="MKBF Actual vs Forecast", xaxis_title="Date", yaxis_title="MKBF", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Editable table
    st.subheader("✏️ Update MKBF Actual Data")
    editable_df = actual.copy()
    editable_df["Date"] = editable_df["Date"].dt.strftime("%Y-%m-%d")

    edited_df = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True, hide_index=True)

    if st.button("💾 Save & Retrain Forecast"):
        try:
            edited_df["Date"] = pd.to_datetime(edited_df["Date"], errors="coerce")
            if edited_df["Date"].isna().any() or edited_df["Value"].isna().any():
                st.warning("⚠️ Invalid dates or missing values detected.")
            else:
                edited_df.to_csv(ART_DIR / "actual.csv", index=False)
                st.cache_data.clear()
                st.success("✅ Saved successfully! Retraining Prophet...")
                st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Error saving: {e}")

    # Downloads
    with st.expander("📂 Download Data"):
        st.download_button("Download Combined (Actual + Forecast)", data=combined.to_csv(index=False),
                           file_name="combined_forecast.csv", mime="text/csv")
        st.download_button("Download actual.csv", data=edited_df.to_csv(index=False),
                           file_name="actual.csv", mime="text/csv")


# =========================================================
# Root Cause Classification
# =========================================================
def page_text_classification():
    st.header("Root Cause Classification")
    model_path = MODEL_DIR / "model.pkl"
    vec_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    le_path = MODEL_DIR / "label_encoder.pkl"

    if not (model_path.exists() and vec_path.exists() and le_path.exists()):
        st.error("⚠️ Model files not found in 'models/'. Place model.pkl, tfidf_vectorizer.pkl, and label_encoder.pkl inside /models.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    user_text = st.text_area("Enter incident description:", height=120)
    if st.button("Predict Root Cause"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            features = vectorizer.transform([user_text])
            proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else np.ones(len(model.classes_)) / len(model.classes_)
            encoded_pred = model.predict(features)[0]
            predicted_label = label_encoder.inverse_transform([encoded_pred])[0]
            label_names = label_encoder.inverse_transform(model.classes_)

            st.success(f"Predicted: **{predicted_label}** — Confidence: {max(proba):.1%}")
            top_idx = np.argsort(proba)[::-1][:3]
            top_df = pd.DataFrame({"Class": label_names[top_idx], "Confidence": [f"{p:.1%}" for p in proba[top_idx]]})
            st.dataframe(top_df, use_container_width=True)


# =========================================================
# Sidebar Navigation
# =========================================================
with st.sidebar:
    selected = option_menu(
        None,
        ["Home", "MKBF Forecast", "Root Cause Classification"],
        icons=["house", "graph-up", "chat-text"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0.5rem", "background-color": "#161a1d", "border-radius": "16px"},
            "icon": {"color": "#FAFAFA", "font-size": "1.1rem"},
            "nav-link": {"font-size": "1rem", "color": "#FAFAFA", "padding": "0.6rem 0.8rem", "border-radius": "12px", "margin": "6px 8px"},
            "nav-link-selected": {"background-color": "#1f77b4", "color": "white"},
        },
    )

# =========================================================
# Router
# =========================================================
if selected == "Home":
    page_home()
elif selected == "MKBF Forecast":
    page_mkbf()
elif selected == "Root Cause Classification":
    page_text_classification()
