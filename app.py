# app.py
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


def get_base64_of_bin_file(bin_file: Path) -> str:
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# =========================
# =========================
# Recommended folder layout (matches your repo structure)
# =========================
from pathlib import Path

# Always use the same folder where app.py is located
HUB_ROOT = Path(__file__).resolve().parent

# Subfolders relative to app.py
ASSETS_DIR = HUB_ROOT / "assets"
ART_DIR = HUB_ROOT / "artifacts" / "mkbf_prophet"
MODELS_DIR = HUB_ROOT / "models"

HUB_ROOT.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    file_path = ASSETS_DIR / "qw3.jpg"
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
# Page: MKBF Forecast (auto-update + combined)
# =========================
def page_mkbf():
    st.markdown("## <div style='text-align: center;'>MKBF Forecast</div>", unsafe_allow_html=True)

    try:
        actual = pd.read_csv(ART_DIR / "actual.csv")
    except Exception:
        st.error("❌ Cannot find 'actual.csv' in artifacts folder.")
        return

    # Auto-detect column names
    actual.columns = [c.strip() for c in actual.columns]
    if set(["ds", "y"]).issubset(actual.columns):
        actual = actual.rename(columns={"ds": "Date", "y": "Value"})
    elif not set(["Date", "Value"]).issubset(actual.columns):
        st.error("❌ 'actual.csv' must contain either ['Date','Value'] or ['ds','y'] columns.")
        return

    actual["Date"] = pd.to_datetime(actual["Date"], errors="coerce")
    actual = actual.dropna(subset=["Date", "Value"]).sort_values("Date")

    # Retrain Prophet
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

    # Forecast
    horizon = st.slider("Forecast horizon (months ahead)", 6, 36, 12)
    future = m.make_future_dataframe(periods=horizon, freq="MS")
    forecast = m.predict(future)

    # Combine into one table
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
    fig.add_trace(
        go.Scatter(x=combined["Date"], y=combined["Forecast"], mode="lines", name="Forecast", line=dict(width=3))
    )
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
    fig.add_trace(
        go.Scatter(x=combined["Date"], y=combined["Actual"], mode="lines+markers", name="Actual", line=dict(width=2))
    )
    fig.update_layout(title="MKBF Actual vs Forecast", xaxis_title="Date", yaxis_title="MKBF", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Combined table
    st.subheader("📊 Actual + Forecast Combined Table")
    st.dataframe(combined.tail(24), use_container_width=True)

    # Editable table
    st.subheader("✏️ Update MKBF Actual Data")
    editable_df = actual.copy()
    editable_df["Date"] = editable_df["Date"].dt.strftime("%Y-%m-%d")

    edited_df = st.data_editor(
        editable_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="mkbf_editor"
    )

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
        st.download_button(
            "Download Combined (Actual + Forecast)",
            data=combined.to_csv(index=False),
            file_name="combined_forecast.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download actual.csv",
            data=edited_df.to_csv(index=False),
            file_name="actual.csv",
            mime="text/csv",
        )


# =========================
# Root Cause Classification
# =========================
def page_text_classification():
    st.header("Root Cause Classification")

    model_path = MODELS_DIR / "model.pkl"
    vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    le_path = MODELS_DIR / "label_encoder.pkl"

    if not (model_path.exists() and vec_path.exists() and le_path.exists()):
        st.error(
            "Model files not found. Place model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl in the 'models' folder."
        )
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
            proba = (
                model.predict_proba(features)[0]
                if hasattr(model, "predict_proba")
                else np.ones(len(model.classes_)) / len(model.classes_)
            )
            encoded_pred = model.predict(features)[0]
            predicted_label = label_encoder.inverse_transform([encoded_pred])[0]
            label_names = label_encoder.inverse_transform(model.classes_)

            st.success(f"Predicted: **{predicted_label}** — Confidence: {max(proba):.1%}")

            top_idx = np.argsort(proba)[::-1][:3]
            top_df = pd.DataFrame(
                {"Class": label_names[top_idx], "Confidence": [f"{p:.1%}" for p in proba[top_idx]]}
            )
            st.dataframe(top_df, use_container_width=True)


# =========================
# Forecasting App (Full)
# =========================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def page_forecast_app():
    st.title("Time Series Forecasting App")
    st.write("Upload or use dummy data, explore, clean, model, evaluate, and forecast — with guidance throughout.")

    # -----------------------------
    # Step 1: Data source
    # -----------------------------
    st.subheader("Step 1: Upload your data source")
    use_dummy = st.checkbox("Use dummy data", value=False)
    uploaded_file = None if use_dummy else st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Your CSV should include a date column and a numeric **target** column.")

    if use_dummy:
        df_raw = make_dummy_monthly()
        st.info("Using dummy monthly data with trend + yearly seasonality.")
    else:
        if uploaded_file is None:
            st.stop()
        df_raw = pd.read_csv(uploaded_file)

    # -----------------------------
    # Step 2: Columns & quick data info
    # -----------------------------
    st.subheader("Step 2: Choose columns")
    cols = list(df_raw.columns)
    date_col = st.selectbox("Date column", cols, index=0)
    target_col = st.selectbox("Target column (numeric)", cols, index=min(1, len(cols) - 1))
    dim_candidates = [c for c in cols if c not in (date_col, target_col)]
    dims_selected = st.multiselect("Optional dimensions (for filtering/aggregation)", dim_candidates)

    with st.expander("Raw data preview & info"):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df_raw.head(10), use_container_width=True)
        with c2:
            st.markdown("**Raw df.info() summary**")
            st.dataframe(df_info_table(df_raw), use_container_width=True)

    # -----------------------------
    # Step 3: Cleaning (selectors, not checkboxes)
    # -----------------------------
    st.subheader("Step 3: Data cleaning")
    st.caption("Pick which columns to cast to datetime / numeric. Choose how to handle duplicates and sorting.")

    colA, colB = st.columns(2)
    with colA:
        to_datetime_cols = st.multiselect("Convert these columns to datetime", options=cols, default=[date_col])
        dup_policy = st.selectbox("Duplicate handling on date", ["Keep first", "Keep last", "Don't drop"], index=0)
    with colB:
        to_numeric_cols = st.multiselect("Convert these columns to numeric", options=cols, default=[target_col])
        sort_choice = st.selectbox("Sort by date", ["Ascending", "Descending", "No sort"], index=0)

    # Apply cleaning (do not mutate original raw)
    df = df_raw.copy()

    # Cast types
    for c in to_datetime_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in to_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing required columns
    df = df.dropna(subset=[date_col, target_col])

    # Duplicate handling on date
    if dup_policy != "Don't drop":
        keep = "first" if dup_policy == "Keep first" else "last"
        df = df.drop_duplicates(subset=[date_col], keep=keep)

    # Sort
    if sort_choice != "No sort":
        ascending = sort_choice == "Ascending"
        df = df.sort_values(date_col, ascending=ascending)

    # Rename to ds/y for modeling
    df = df[[date_col, target_col] + dims_selected].rename(columns={date_col: "ds", target_col: "y"})

    with st.expander("Before vs After (types & head)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before cleaning**")
            st.dataframe(df_info_table(df_raw), use_container_width=True)
            st.dataframe(df_raw.head(5), use_container_width=True)
        with c2:
            st.markdown("**After cleaning**")
            st.dataframe(df_info_table(df), use_container_width=True)
            st.dataframe(df.head(5), use_container_width=True)

    # -----------------------------
    # Step 4: Resampling (optional)
    # -----------------------------
    st.subheader("Step 4: Resampling")
    enable_resample = st.checkbox("Enable resampling", value=False)
    freq_for_future = "D"

    if enable_resample:
        st.caption("Convert your calendar to Daily / Weekly / Monthly.")
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}

        colr1, colr2 = st.columns(2)
        with colr1:
            new_freq_label = st.selectbox("New frequency", ["Weekly", "Monthly", "Daily"], index=1)
        with colr2:
            agg_res = st.selectbox("Aggregation when resampling", ["Mean", "Sum", "Median", "Max", "Min"], index=0)

        df = (
            df.set_index("ds")
            .resample(freq_map[new_freq_label])
            .agg({"y": agg_fn(agg_res)})
            .dropna()
            .reset_index()
        )
        freq_for_future = "MS" if new_freq_label == "Monthly" else ("W" if new_freq_label == "Weekly" else "D")

    # -----------------------------
    # Step 6: EDA
    # -----------------------------
    st.subheader("Step 5: EDA")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["ds"], df["y"])
    ax.set_title("Observed series")
    st.pyplot(fig)

    with st.expander("Seasonal decomposition"):
        model_decomp = st.selectbox("Decomposition model", ["additive", "multiplicative"], index=0)
        st.caption(
            "• **Additive** = level + season + noise (use when seasonal swings are constant)\n"
            "• **Multiplicative** = level × season × noise (use when seasonal swings grow with the level)\n"
            "Examples: Additive → temperature by month; Multiplicative → revenue with % seasonal lift"
        )
        try:
            decomp = seasonal_decompose(df.set_index("ds")["y"], model=model_decomp, period=12)
            fig2 = decomp.plot()
            fig2.set_size_inches(8, 6)
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Decomposition skipped: {e}")

    # -----------------------------
    # Step 7: Train window & metrics
    # -----------------------------
    st.subheader("Step 6: Training window & metrics")
    min_dt, max_dt = df["ds"].min(), df["ds"].max()
    c1, c2 = st.columns(2)
    with c1:
        train_start = st.date_input(
            "Training start",
            value=min_dt.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
        )
    with c2:
        train_end = st.date_input(
            "Training end",
            value=(max_dt - pd.DateOffset(months=6)).date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
        )
    metrics_wanted = st.multiselect("Evaluation metrics", ["MAE", "RMSE", "MAPE", "R²"], default=["MAE", "RMSE", "MAPE"])

    # -----------------------------
    # Step 8: Model
    # -----------------------------
    st.subheader("Step 7: Model")
    model_choice = st.radio("Choose model", ["Prophet", "Exponential Smoothing", "ARIMA", "SARIMA"], horizontal=True)
    holidays_df = None

    if model_choice == "Prophet":
        # --- pretty sections with color ---
        st.markdown(
            "<div style='border-left:6px solid #3b82f6;background:#f0f7ff;padding:10px 12px;margin:8px 0;'>"
            "<b>Changepoints</b><br>Where the trend is allowed to change.</div>",
            unsafe_allow_html=True,
        )
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            cps = st.number_input(
                "changepoint_prior_scale", min_value=0.001, max_value=0.5, step=0.001, value=0.1, format="%.3f"
            )
        with cp2:
            ncp = st.slider("n_changepoints", min_value=0, max_value=50, value=25, step=1)
        with cp3:
            cpr = st.slider("changepoint_range", min_value=0.8, max_value=1.0, value=0.95, step=0.05)

        st.markdown(
            "<div style='border-left:6px solid #16a34a;background:#effaf3;padding:10px 12px;margin:8px 0;'>"
            "<b>Seasonality</b><br>Use multiplicative if seasonal amplitude grows with level.</div>",
            unsafe_allow_html=True,
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            seasonality_mode = st.selectbox("seasonality_mode", ["additive", "multiplicative"], index=0)
        with s2:
            seas_prior = st.number_input("seasonality_prior_scale", min_value=0.01, max_value=10.0, value=10.0, step=0.5)
        with s3:
            yearly = st.selectbox("yearly_seasonality", ["auto", "True", "False"], index=0)

        monthly_on = st.checkbox("Add monthly seasonality (period≈30.5)", value=False)
        monthly_fourier = st.slider(
            "monthly fourier_order", min_value=3, max_value=20, value=5, step=1, disabled=not monthly_on
        )

        weekly = st.selectbox("weekly_seasonality", ["auto", "True", "False"], index=0)

        st.markdown(
            "<div style='border-left:6px solid #ef4444;background:#fff5f5;padding:10px 12px;margin:8px 0;'>"
            "<b>Holidays</b><br>Add public/school holidays that affect demand.</div>",
            unsafe_allow_html=True,
        )
        h1, h2 = st.columns([2, 1])
        with h1:
            built_in_country = st.selectbox(
                "Built-in country holidays", ["None", "Malaysia", "France", "UnitedStates", "UnitedKingdom", "India"], index=0
            )
        with h2:
            holiday_prior = st.number_input("holidays_prior_scale", min_value=0.01, max_value=10.0, value=10.0, step=0.5)

        up_h = st.file_uploader(
            "Or upload holidays CSV (columns: ds, holiday, [lower_window, upper_window])", type=["csv"]
        )
        if up_h is not None:
            holidays_df = pd.read_csv(up_h)
            if "ds" in holidays_df.columns:
                holidays_df["ds"] = pd.to_datetime(holidays_df["ds"], errors="coerce")
            else:
                st.warning("Holidays CSV must include 'ds' column.")

    elif model_choice == "Exponential Smoothing":
        st.info(
            "**Simple**: level only (good for stable series)\n\n"
            "**Holt’s Linear**: level + trend (when you see a clear slope)\n\n"
            "**Holt-Winters**: level + trend + seasonality (needs seasonal period)"
        )

    # -----------------------------
    # Train & evaluate
    # -----------------------------
    if st.button("Run training & evaluation"):
        train = df[(df["ds"] >= pd.to_datetime(train_start)) & (df["ds"] <= pd.to_datetime(train_end))].copy()
        test = df[df["ds"] > pd.to_datetime(train_end)].copy()

        if train.empty or test.empty:
            st.error("Training or test set is empty. Adjust your training window.")
            return

        if model_choice == "Prophet":
            m = Prophet(
                changepoint_prior_scale=cps,
                n_changepoints=ncp,
                changepoint_range=cpr,
                seasonality_mode=seasonality_mode,
                seasonality_prior_scale=seas_prior,
                yearly_seasonality=None if yearly == "auto" else (yearly == "True"),
                weekly_seasonality=None if weekly == "auto" else (weekly == "True"),
                holidays=holidays_df,
                holidays_prior_scale=holiday_prior,
            )
            if monthly_on:
                m.add_seasonality(name="monthly", period=30.5, fourier_order=monthly_fourier)
            if built_in_country != "None":
                try:
                    m.add_country_holidays(country_name=built_in_country)
                except Exception as e:
                    st.warning(f"Could not add built-in holidays '{built_in_country}': {e}")
            m.fit(train[["ds", "y"]])
            future = m.make_future_dataframe(periods=len(test), freq=freq_for_future)
            fcst = m.predict(future)
            y_pred = fcst.set_index("ds").loc[test["ds"], "yhat"].values

        elif model_choice == "Exponential Smoothing":
            ts = train.set_index("ds")["y"]
            smoothing_type = st.radio("Type", ["Simple", "Holt’s Linear", "Holt-Winters"], horizontal=True)
            if smoothing_type == "Simple":
                fit = SimpleExpSmoothing(ts).fit()
            elif smoothing_type == "Holt’s Linear":
                fit = ExponentialSmoothing(ts, trend="add").fit()
            else:
                fit = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
            y_pred = fit.predict(start=test["ds"].min(), end=test["ds"].max())

        elif model_choice == "ARIMA":
            ts = train.set_index("ds")["y"]
            fit = ARIMA(ts, order=(1, 1, 1)).fit()
            y_pred = fit.predict(start=test["ds"].min(), end=test["ds"].max(), typ="levels")

        else:  # SARIMA
            ts = train.set_index("ds")["y"]
            m_seas = 12 if freq_for_future in ["MS", "M"] else 7 if freq_for_future == "W" else 0
            order = (1, 1, 1)
            seas_order = (1, 1, 1, m_seas) if m_seas else (0, 0, 0, 0)
            fit = SARIMAX(
                ts,
                order=order,
                seasonal_order=seas_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            y_pred = fit.predict(start=test["ds"].min(), end=test["ds"].max())

        y_true = test.set_index("ds")["y"].values
        met = compute_metrics(y_true, np.asarray(y_pred), which=metrics_wanted)

        # Evaluation chart
        st.subheader("Evaluation")
        fig_eval, ax = plt.subplots(figsize=(7, 3))
        ax.plot(train["ds"], train["y"], label="Train")
        ax.plot(test["ds"], test["y"], label="Test (actual)")
        ax.plot(test["ds"], y_pred, label="Predicted", linestyle="--")
        ax.axvline(pd.to_datetime(train_end), color="red", linestyle="--", label="Train/Test split")
        ax.legend()
        st.pyplot(fig_eval)

        cols = st.columns(len(metrics_wanted))
        for i, k in enumerate(metrics_wanted):
            v = met.get(k, None)
            if v is not None:
                cols[i].metric(k, f"{v * 100:.2f}%" if k == "MAPE" else f"{v:,.2f}")

    # -----------------------------
    # Future forecast
    # -----------------------------
    st.subheader("Step 8: Forecast into the future")
    horizon = st.slider("Forecast horizon", min_value=1, max_value=36, value=12)

    if st.button("Generate future forecast"):
        if model_choice == "Prophet":
            m2 = Prophet(
                changepoint_prior_scale=cps,
                n_changepoints=ncp,
                changepoint_range=cpr,
                seasonality_mode=seasonality_mode,
                seasonality_prior_scale=seas_prior,
                yearly_seasonality=None if yearly == "auto" else (yearly == "True"),
                weekly_seasonality=None if weekly == "auto" else (weekly == "True"),
                holidays=holidays_df,
                holidays_prior_scale=holiday_prior,
            )
            if monthly_on:
                m2.add_seasonality(name="monthly", period=30.5, fourier_order=monthly_fourier)
            if built_in_country != "None":
                try:
                    m2.add_country_holidays(country_name=built_in_country)
                except Exception as e:
                    st.warning(f"Could not add built-in holidays '{built_in_country}': {e}")

            m2.fit(df[["ds", "y"]])
            future2 = m2.make_future_dataframe(periods=horizon, freq=freq_for_future)
            fcst2 = m2.predict(future2)
            fig_f = m2.plot(fcst2)
            fig_f.set_size_inches(8, 4)
            st.pyplot(fig_f)
            out_df = fcst2[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)

        else:
            ts_all = df.set_index("ds")["y"]

            if model_choice == "Exponential Smoothing":
                model = ExponentialSmoothing(
                    ts_all,
                    trend="add",
                    seasonal=("add" if freq_for_future in ["MS", "M"] else None),
                    seasonal_periods=(12 if freq_for_future in ["MS", "M"] else None),
                )
                fit = model.fit(optimized=True)
                future_index = pd.date_range(
                    ts_all.index[-1] + (pd.offsets.MonthBegin() if freq_for_future in ["MS", "M"] else pd.Timedelta(days=1)),
                    periods=horizon,
                    freq=freq_for_future,
                )
                yhat = fit.forecast(horizon)

            elif model_choice == "ARIMA":
                fit = ARIMA(ts_all, order=(1, 1, 1)).fit()
                future_index = pd.date_range(
                    ts_all.index[-1] + (pd.offsets.MonthBegin() if freq_for_future in ["MS", "M"] else pd.Timedelta(days=1)),
                    periods=horizon,
                    freq=freq_for_future,
                )
                yhat = fit.forecast(horizon)

            else:  # SARIMA
                m_seas = 12 if freq_for_future in ["MS", "M"] else 7 if freq_for_future == "W" else 0
                fit = SARIMAX(
                    ts_all,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, m_seas) if m_seas else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                future_index = pd.date_range(
                    ts_all.index[-1] + (pd.offsets.MonthBegin() if freq_for_future in ["MS", "M"] else pd.Timedelta(days=1)),
                    periods=horizon,
                    freq=freq_for_future,
                )
                yhat = fit.forecast(horizon)

            # Plot forecast
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(ts_all.index, ts_all.values, label="Observed")
            ax.plot(future_index, yhat, label="Forecast", linestyle="--")
            ax.legend()
            st.pyplot(fig)
            out_df = pd.DataFrame({"ds": future_index, "yhat": yhat})

        st.download_button(
            "Download forecast results in CSV",
            out_df.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv",
        )


# =========================
# Sidebar Navigation
# =========================
with st.sidebar:
    selected = option_menu(
        None,
        ["Home", "MKBF Forecast", "Root Cause Classification", "Forecasting App"],
        icons=["house", "graph-up", "chat-text", "bar-chart-line"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0.5rem", "background-color": "#161a1d", "border-radius": "16px"},
            "icon": {"color": "#FAFAFA", "font-size": "1.1rem"},
            "nav-link": {
                "font-size": "1rem",
                "color": "#FAFAFA",
                "padding": "0.6rem 0.8rem",
                "border-radius": "12px",
                "margin": "6px 8px",
            },
            "nav-link-selected": {"background-color": "#1f77b4", "color": "white"},
        },
    )

# =========================
# Router
# =========================
if selected == "Home":
    page_home()
elif selected == "MKBF Forecast":
    page_mkbf()
elif selected == "Root Cause Classification":
    page_text_classification()
elif selected == "Forecasting App":
    page_forecast_app()

