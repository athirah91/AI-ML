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
from statsmodels.tsa.stattools import adfuller

# --- Page config ---
st.set_page_config(page_title="GSAD Data Science Hub", layout="wide")

from streamlit_option_menu import option_menu
import base64


# =========================
# Helper: Base64 encode image
# =========================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# =========================
# Shared helpers / loaders
# =========================
APP_DIR = Path(__file__).parent
ART_DIR = APP_DIR / "artifacts" / "mkbf_prophet"


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


# =========================
# Page: Home
# =========================
def page_home():
    file_path = "assets/qw3.jpg"
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

# =========================
# Page: MKBF Forecast
# =========================
def page_mkbf():
    st.title("📈 MKBF Forecast Dashboard")

    # --- Load actual data ---
    try:
        actual = pd.read_csv(ART_DIR / "actual.csv")
    except Exception:
        st.error("❌ Cannot find 'actual.csv' in artifacts folder.")
        return

    actual["Date"] = pd.to_datetime(actual["Date"])
    actual["Value"] = pd.to_numeric(actual["Value"], errors="coerce")

    # --- Sidebar options ---
    model_choice = st.selectbox(
        "Select Forecast Model",
        ["Prophet", "ARIMA", "SARIMA", "Holt-Winters"],
    )

    forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)

    # --- Show actual data preview ---
    with st.expander("🔍 Show raw actual data"):
        st.dataframe(actual.tail(10))

    st.markdown("---")

    # --- Prophet Model ---
    if model_choice == "Prophet":
        st.info("🔁 Training Prophet model on the latest actual data...")

        df_train = actual.rename(columns={"Date": "ds", "Value": "y"})
        m = Prophet(
            changepoint_prior_scale=0.1,
            changepoint_range=0.95,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )

        try:
            m.fit(df_train)
        except Exception as e:
            st.error(f"⚠️ Prophet model training failed: {e}")
            return

        future = m.make_future_dataframe(periods=forecast_horizon, freq="M")
        forecast = m.predict(future)

        # Plot
        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

        st.success("✅ Prophet forecast generated successfully!")

        # Save outputs
        (ART_DIR / "output").mkdir(exist_ok=True)
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
            ART_DIR / "output" / "prophet_forecast.csv", index=False
        )

    # --- ARIMA Model ---
    elif model_choice == "ARIMA":
        st.info("🔁 Training ARIMA model...")
        df = actual.set_index("Date")["Value"]

        try:
            model = ARIMA(df, order=(1, 1, 1))
            fitted = model.fit()
            pred = fitted.get_forecast(steps=forecast_horizon)
            pred_ci = pred.conf_int()
        except Exception as e:
            st.error(f"⚠️ ARIMA model failed: {e}")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        df.plot(ax=ax, label="Observed")
        pred.predicted_mean.plot(ax=ax, label="Forecast", color="orange")
        ax.fill_between(
            pred_ci.index,
            pred_ci.iloc[:, 0],
            pred_ci.iloc[:, 1],
            color="gray",
            alpha=0.3,
        )
        ax.legend()
        st.pyplot(fig)
        st.success("✅ ARIMA forecast generated successfully!")

    # --- SARIMA Model ---
    elif model_choice == "SARIMA":
        st.info("🔁 Training SARIMA model...")
        df = actual.set_index("Date")["Value"]

        try:
            model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
            fitted = model.fit(disp=False)
            pred = fitted.get_forecast(steps=forecast_horizon)
            pred_ci = pred.conf_int()
        except Exception as e:
            st.error(f"⚠️ SARIMA model failed: {e}")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        df.plot(ax=ax, label="Observed")
        pred.predicted_mean.plot(ax=ax, label="Forecast", color="orange")
        ax.fill_between(
            pred_ci.index,
            pred_ci.iloc[:, 0],
            pred_ci.iloc[:, 1],
            color="gray",
            alpha=0.3,
        )
        ax.legend()
        st.pyplot(fig)
        st.success("✅ SARIMA forecast generated successfully!")

    # --- Holt-Winters Model ---
    elif model_choice == "Holt-Winters":
        st.info("🔁 Training Holt-Winters model...")
        df = actual.set_index("Date")["Value"]

        try:
            model = ExponentialSmoothing(
                df, trend="add", seasonal="add", seasonal_periods=12
            )
            fitted = model.fit()
            forecast = fitted.forecast(forecast_horizon)
        except Exception as e:
            st.error(f"⚠️ Holt-Winters model failed: {e}")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        df.plot(ax=ax, label="Observed")
        forecast.plot(ax=ax, label="Forecast", color="orange")
        ax.legend()
        st.pyplot(fig)
        st.success("✅ Holt-Winters forecast generated successfully!")

