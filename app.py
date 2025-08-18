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

# --- Page config (light/original look) ---
st.set_page_config(page_title="GSAD Data Science Hub", layout="wide")

from streamlit_option_menu import option_menu

# =========================
# Shared helpers / loaders
# =========================
APP_DIR = Path(__file__).parent
ART_DIR = Path(r"C:\AIML\Final\artifacts\mkbf_prophet")

@st.cache_data(show_spinner=False)
def load_df_csv(path: Path, filename: str):
    f = path / filename
    if f.exists():
        return pd.read_csv(f, parse_dates=['ds'])
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
    st.title("GSAD Data Science Hub")
    st.write("Hello Athirah 👋")
    st.markdown("Your central hub for forecasting, analytics, and AI-powered insights.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MKBF Forecast Models", "3", "SARIMA / Holt / Prophet")
        st.caption("Visualize and monitor Mean Kilometers Between Failures trends.")
    with col2:
        st.metric("Root Cause Categories", "X", "trained model")
        st.caption("Classify incident descriptions into root cause categories with AI.")

    st.markdown("""
    ### 📌 How to Use
    - **MKBF Forecast** → View predictive charts and trends from your saved artifacts.
    - **Root Cause Classification** → Classify incident descriptions instantly.
    - **Forecast App** → Upload any time series (or use dummy data) to explore & forecast.
    """)

# =========================
# Page: MKBF Forecast (uses artifacts)
# =========================
def page_mkbf():
    st.markdown("## <div style='text-align: center;'>MKBF Forecast</div>", unsafe_allow_html=True)

    actual = load_df_csv(ART_DIR, "actual.csv")                 # ds, y
    fcst   = load_df_csv(ART_DIR, "forecast_full.csv")          # ds, yhat, yhat_lower, yhat_upper
    metrics = load_metrics_json(ART_DIR)

    if (actual is None) or (fcst is None):
        st.error(f"Could not find actual/forecast files in: {ART_DIR}")
        st.info("Ensure your notebook saved 'actual.csv' and 'forecast_full.csv'.")
        return

    default_cutoff = (metrics.get("cutoff") if metrics else "2024-06-30")
    cutoff_date = st.text_input("Train/Test cutoff (for split line)", value=default_cutoff)

    actual_use = actual.copy()
    actual_use = actual_use[actual_use["ds"] <= fcst["ds"].max()]

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fcst["ds"], y=fcst["yhat"], mode="lines", name="Forecast", line=dict(width=3)))
    fig.add_trace(go.Scatter(
        x=pd.concat([fcst["ds"], fcst["ds"][::-1]]),
        y=pd.concat([fcst["yhat_upper"], fcst["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(135, 206, 235, 0.25)", line=dict(width=0), name="Confidence"
    ))
    fig.add_trace(go.Scatter(x=actual_use["ds"], y=actual_use["y"], mode="lines", name="Actual", line=dict(width=2)))

    try:
        cdt = pd.to_datetime(cutoff_date)
        ymin = float(min(actual_use["y"].min(), fcst["yhat_lower"].min()))
        ymax = float(max(actual_use["y"].max(), fcst["yhat_upper"].max()))
        fig.add_trace(go.Scatter(x=[cdt, cdt], y=[ymin, ymax], mode="lines", name="Train/Test Split", line=dict(dash="dash")))
    except Exception:
        pass

    mil = fcst[fcst["yhat"] >= 1_000_000]
    if not mil.empty:
        mil_date = mil.iloc[0]["ds"]; mil_val  = mil.iloc[0]["yhat"]
        fig.add_trace(go.Scatter(x=[mil_date], y=[mil_val], mode="markers+text",
                                 text=[f"1M MKBF: {pd.to_datetime(mil_date).strftime('%b %Y')}"],
                                 textposition="top left", name="1M Milestone"))
    fig.update_layout(title="Prophet Forecast vs Actual", xaxis_title="Date", yaxis_title="MKBF", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Metric cards (unified look)
    st.markdown("""
    <style>
    div[data-testid="metric-container"]{background-color:white;border:1px solid #ddd;border-radius:8px;padding:10px;}
    </style>
    """, unsafe_allow_html=True)

    if metrics:
        st.subheader("Evaluation on Test Split")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"{metrics.get('MAE', 0):,.0f}")
        c2.metric("RMSE", f"{metrics.get('RMSE', 0):,.0f}")
        mape_val = metrics.get('MAPE')
        c3.metric("MAPE", f"{mape_val*100:,.2f}%" if mape_val is not None else "-")
        r2_val = metrics.get('R2')
        c4.metric("R²", f"{r2_val:.3f}" if r2_val is not None else "-")
        c5.metric("Test points", f"{metrics.get('n_test', 0)}")

    with st.expander("Show data tables"):
        st.markdown("**Actuals (trimmed to forecast range)**")
        st.dataframe(actual_use.tail(24), use_container_width=True)
        st.markdown("**Forecast (tail)**")
        st.dataframe(fcst.tail(24), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download forecast_full.csv", data=fcst.to_csv(index=False),
                           file_name="forecast_full.csv", mime="text/csv")
    with col2:
        if metrics:
            st.download_button("Download metrics.json", data=json.dumps(metrics, indent=2),
                               file_name="metrics.json", mime="application/json")

# =========================
# Page: Root Cause Classification (as you had)
# =========================
def page_text_classification():
    st.header("Root Cause Classification")
    model_path = APP_DIR / "model.pkl"
    vec_path   = APP_DIR / "tfidf_vectorizer.pkl"
    le_path    = APP_DIR / "label_encoder.pkl"

    if not (model_path.exists() and vec_path.exists() and le_path.exists()):
        st.error("Model files not found. Put model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl next to app.py.")
        return

    with open(model_path, "rb") as f: model = pickle.load(f)
    with open(vec_path, "rb") as f: vectorizer = pickle.load(f)
    with open(le_path, "rb") as f: label_encoder = pickle.load(f)

    user_text = st.text_area("Enter incident description to classify:", height=120, placeholder="Type your incident description…")
    if st.button("Predict Root Cause"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            features = vectorizer.transform([user_text])
            classes = np.array(model.classes_)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(features)
                scores = np.asarray(scores)
                if scores.ndim == 0:
                    score = float(scores); p_pos = 1.0/(1.0+np.exp(-score))
                    proba = np.array([1 - p_pos, p_pos])
                else:
                    if scores.ndim > 1: scores = scores[0]
                    ex = np.exp(scores - np.max(scores)); proba = ex / ex.sum()
            else:
                proba = np.ones(len(classes)) / len(classes)

            encoded_pred = model.predict(features)[0]
            predicted_label = label_encoder.inverse_transform([encoded_pred])[0]
            label_names = label_encoder.inverse_transform(classes)
            pred_idx = np.where(classes == encoded_pred)[0][0]
            pred_conf = float(proba[pred_idx])

            st.success(f"Predicted Class: {predicted_label}  —  Confidence: {pred_conf:.1%}")

            top_idx = np.argsort(proba)[::-1][:3]
            top_df = pd.DataFrame({"Class": label_names[top_idx],
                                   "Confidence": [f"{p:.1%}" for p in proba[top_idx]]})
            st.subheader("Top predictions")
            st.dataframe(top_df, use_container_width=True)

            import plotly.graph_objects as go
            fig_top = go.Figure(go.Bar(x=label_names[top_idx], y=proba[top_idx]))
            fig_top.update_layout(title="Top-3 class probabilities", xaxis_title="Class",
                                  yaxis_title="Confidence", yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig_top, use_container_width=True)

# =========================
# Page: Forecast App (new, guided)
# =========================
def make_dummy_monthly(n_years=5, seed=42):
    rng = pd.date_range("2018-01-01", periods=n_years*12, freq="MS")
    rnd = np.random.RandomState(seed)
    trend = np.linspace(100, 300, len(rng))
    season = 40 * np.sin(2*np.pi*(rng.month/12.0))
    noise = rnd.normal(0, 12, len(rng))
    y = trend + season + noise
    return pd.DataFrame({"ds": rng, "y": y})

def agg_fn(name):
    return {"Mean":"mean","Sum":"sum","Median":"median","Max":"max","Min":"min"}[name]

def compute_metrics(y_true, y_pred, which):
    out = {}
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
    if "MAE" in which:  out["MAE"]  = float(mean_absolute_error(y_true, y_pred))
    if "RMSE" in which: out["RMSE"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if "MAPE" in which: out["MAPE"] = float(mean_absolute_percentage_error(y_true, y_pred))
    if "R²" in which and len(y_true) > 1: out["R²"] = float(r2_score(y_true, y_pred))
    return out

def page_forecast_app():
    st.title("📈 Time Series Forecasting App")
    st.write("Upload a time series or use dummy data, explore seasonality, configure models, evaluate, and forecast.")

    # Data source
    st.subheader("Step 0: Pick your data source")
    use_dummy = st.checkbox("Use dummy monthly data (untick to upload your own CSV)", value=True)
    uploaded_file = None if use_dummy else st.file_uploader("Upload CSV", type=["csv"])
    st.caption("CSV must include a date column and a numeric **target** column.")

    if use_dummy:
        df_raw = make_dummy_monthly()
        st.info("Using dummy monthly data with trend + yearly seasonality.")
    else:
        if uploaded_file is None:
            st.stop()
        df_raw = pd.read_csv(uploaded_file)

    # Column selection & cleaning
    st.subheader("Step 1: Choose columns & clean")
    cols = list(df_raw.columns)
    date_col = st.selectbox("Date column", cols, index=0)
    target_col = st.selectbox("Target (numeric)", cols, index=min(1, len(cols)-1))
    dim_candidates = [c for c in cols if c not in (date_col, target_col)]
    dims_selected = st.multiselect("Select dataset dimensions (optional)", dim_candidates)
    with st.expander("Preview raw data"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: apply_datetime = st.checkbox("Parse dates", value=True)
    with c2: drop_invalid  = st.checkbox("Drop invalid dates", value=True)
    with c3: drop_dupes    = st.checkbox("Drop duplicate dates", value=True)
    with c4: sort_dates    = st.checkbox("Sort by date", value=True)

    df = df_raw.copy()
    if apply_datetime: df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if drop_invalid:   df = df.dropna(subset=[date_col])
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Filtering over dimensions + aggregation
    st.subheader("Step 2: Filter & aggregate (optional)")
    for d in dims_selected:
        unique_vals = sorted(df[d].dropna().unique().tolist())
        keep_all = st.checkbox(f"Keep all values for {d}", value=True, key=f"keep_{d}")
        if not keep_all:
            vals = st.multiselect(f"Values to keep for {d}", unique_vals, default=unique_vals[:1], key=f"vals_{d}")
            df = df[df[d].isin(vals)]

    agg_choice = st.selectbox("Target aggregation over dimensions", ["Mean","Sum","Median","Max","Min"], index=0)
    df = df[[date_col, target_col] + dims_selected]
    if dims_selected:
        df = df.groupby(date_col, as_index=False)[target_col].agg(agg_fn(agg_choice))
    df = df.rename(columns={date_col: "ds", target_col: "y"})
    if drop_dupes: df = df.drop_duplicates(subset=["ds"])
    if sort_dates: df = df.sort_values("ds")

    # Resampling
    st.subheader("Step 3: Resampling (optional)")
    st.caption("Convert your calendar to Daily, Weekly, or Monthly.")
    do_resample = st.checkbox("Resample my dataset", value=False)
    freq_map = {"Daily":"D", "Weekly":"W", "Monthly":"MS"}
    if do_resample:
        colr1, colr2 = st.columns(2)
        with colr1: new_freq_label = st.selectbox("New frequency", ["Weekly","Monthly","Daily"], index=1)
        with colr2: agg_choice_resample = st.selectbox("Aggregation when resampling", ["Mean","Sum","Median","Max","Min"], index=0)
        df = (df.set_index("ds").resample(freq_map[new_freq_label]).agg({"y": agg_fn(agg_choice_resample)}).dropna().reset_index())
    freq_for_future = "MS" if (do_resample and new_freq_label=="Monthly") else ("W" if (do_resample and new_freq_label=="Weekly") else "D")

    # EDA
    st.subheader("Step 4: EDA")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df["ds"], df["y"]); ax.set_title("Observed series"); st.pyplot(fig)
    with st.expander("Seasonal decomposition (additive, period=12)"):
        try:
            decomp = seasonal_decompose(df.set_index("ds")["y"], model="additive", period=12)
            fig2 = decomp.plot(); fig2.set_size_inches(8,6); st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Decomposition skipped: {e}")

    # Train window & metrics
    st.subheader("Step 5: Training window & metrics")
    min_dt, max_dt = df["ds"].min(), df["ds"].max()
    c1, c2 = st.columns(2)
    with c1: train_start = st.date_input("Training start", value=min_dt.date(), min_value=min_dt.date(), max_value=max_dt.date())
    with c2: train_end   = st.date_input("Training end",   value=(max_dt - pd.DateOffset(months=6)).date(), min_value=min_dt.date(), max_value=max_dt.date())
    metrics_wanted = st.multiselect("Evaluation metrics", ["MAE","RMSE","MAPE","R²"], default=["MAE","RMSE","MAPE"])

    # Model selection
    st.subheader("Step 6: Model")
    model_choice = st.radio("Choose model", ["Prophet","Exponential Smoothing","ARIMA","SARIMA"], horizontal=True)

    # Prophet controls
    holidays_df = None
    if model_choice == "Prophet":
        st.markdown("**Prophet hyperparameters**")
        cols_cp = st.columns(3)
        with cols_cp[0]:
            cps = st.number_input("Changepoint prior scale (higher = more flexible trend)", min_value=0.001, max_value=1.0, step=0.001, value=0.1, format="%.3f")
        with cols_cp[1]:
            ncp = st.slider("Number of changepoints", min_value=0, max_value=50, value=25, step=1)
        with cols_cp[2]:
            cpr = st.slider("Changepoint range (fraction of history)", min_value=0.1, max_value=1.0, value=0.95, step=0.05)

        cols_seas = st.columns(3)
        with cols_seas[0]: seasonality_mode = st.selectbox("Seasonality mode", ["additive","multiplicative"], index=0)
        with cols_seas[1]: seas_prior = st.number_input("Seasonality prior scale", min_value=0.01, max_value=50.0, value=10.0, step=0.5)
        with cols_seas[2]: yearly = st.selectbox("Yearly seasonality", ["auto", "True", "False"], index=0)
        monthly_on = st.checkbox("Add monthly seasonality", value=False)
        monthly_fourier = st.slider("Monthly fourier order", min_value=3, max_value=20, value=5, step=1, disabled=not monthly_on)
        weekly = st.selectbox("Weekly seasonality", ["auto", "True", "False"], index=0)
        st.caption("Use **multiplicative** if seasonal amplitude grows with level.")

        st.markdown("**Holidays**")
        colh1, colh2 = st.columns([2,1])
        with colh1: built_in_country = st.selectbox("Built-in country holidays (optional)", ["None","France","UnitedStates","UnitedKingdom","India"], index=0)
        with colh2: holiday_prior = st.number_input("Holiday prior scale", min_value=0.01, max_value=50.0, value=10.0, step=0.5)
        up_h = st.file_uploader("Or upload holidays CSV (columns: ds, holiday, [lower_window, upper_window])", type=["csv"])
        if up_h is not None:
            holidays_df = pd.read_csv(up_h)
            if "ds" in holidays_df.columns: holidays_df["ds"] = pd.to_datetime(holidays_df["ds"], errors="coerce")
            else: st.warning("Holidays CSV must include 'ds' column.")

    # Train & evaluate
    if st.button("Run training & evaluation"):
        train = df[(df["ds"] >= pd.to_datetime(train_start)) & (df["ds"] <= pd.to_datetime(train_end))].copy()
        test  = df[df["ds"] > pd.to_datetime(train_end)].copy()
        if train.empty or test.empty:
            st.error("Training or test set is empty. Adjust your training window."); return

        if model_choice == "Prophet":
            m = Prophet(
                changepoint_prior_scale=cps, n_changepoints=ncp, changepoint_range=cpr,
                seasonality_mode=seasonality_mode, seasonality_prior_scale=seas_prior,
                yearly_seasonality=None if yearly=="auto" else (yearly=="True"),
                weekly_seasonality=None if weekly=="auto" else (weekly=="True"),
                holidays=holidays_df, holidays_prior_scale=holiday_prior
            )
            if monthly_on: m.add_seasonality(name="monthly", period=30.5, fourier_order=monthly_fourier)
            if built_in_country != "None":
                try: m.add_country_holidays(country_name=built_in_country)
                except Exception as e: st.warning(f"Could not add built-in holidays: {e}")
            m.fit(train[["ds","y"]])
            future = m.make_future_dataframe(periods=len(test), freq=freq_for_future)
            fcst = m.predict(future); y_pred = fcst.set_index("ds").loc[test["ds"], "yhat"].values

        elif model_choice == "Exponential Smoothing":
            ts = train.set_index("ds")["y"]
            model = ExponentialSmoothing(ts, trend="add",
                                         seasonal=("add" if freq_for_future in ["MS","M"] else None),
                                         seasonal_periods=(12 if freq_for_future in ["MS","M"] else None))
            fit = model.fit(optimized=True)
            y_pred = fit.predict(start=test["ds"].min(), end=test["ds"].max())

        elif model_choice == "ARIMA":
            ts = train.set_index("ds")["y"]; fit = ARIMA(ts, order=(1,1,1)).fit()
            y_pred = fit.predict(start=test["ds"].min(), end=test["ds"].max(), typ="levels")

        else:  # SARIMA
            ts = train.set_index("ds")["y"]
            m_seas = 12 if freq_for_future in ["MS","M"] else 7 if freq_for_future=="W" else 0
            order = (1,1,1); seas_order = (1,1,1,m_seas) if m_seas else (0,0,0,0)
            fit = SARIMAX(ts, order=order, seasonal_order=seas_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            y_pred = fit.predict(start=test["ds"].min(), end=test["ds"].max())

        y_true = test.set_index("ds")["y"].values
        met = compute_metrics(y_true, np.asarray(y_pred), which=metrics_wanted)

        st.subheader("Evaluation")
        fig_eval, ax = plt.subplots(figsize=(7,3))
        ax.plot(train["ds"], train["y"], label="Train")
        ax.plot(test["ds"], test["y"], label="Test (actual)")
        ax.plot(test["ds"], y_pred, label="Predicted", linestyle="--")
        ax.axvline(pd.to_datetime(train_end), color="red", linestyle="--", label="Train/Test split")
        ax.legend(); st.pyplot(fig_eval)

        cols = st.columns(len(metrics_wanted))
        for i, k in enumerate(metrics_wanted):
            v = met.get(k, None)
            if v is not None:
                cols[i].metric(k, f"{v*100:.2f}%" if k=="MAPE" else f"{v:,.2f}")

    # Future forecast
    st.subheader("Step 7: Forecast into the future")
    horizon = st.slider("Forecast horizon", min_value=1, max_value=36, value=12)
    if st.button("Generate future forecast"):
        if model_choice == "Prophet":
            m2 = Prophet(
                changepoint_prior_scale=cps, n_changepoints=ncp, changepoint_range=cpr,
                seasonality_mode=seasonality_mode, seasonality_prior_scale=seas_prior,
                yearly_seasonality=None if yearly=="auto" else (yearly=="True"),
                weekly_seasonality=None if weekly=="auto" else (weekly=="True"),
                holidays=holidays_df, holidays_prior_scale=holiday_prior
            )
            if monthly_on: m2.add_seasonality(name="monthly", period=30.5, fourier_order=monthly_fourier)
            if built_in_country != "None":
                try: m2.add_country_holidays(country_name=built_in_country)
                except Exception as e: st.warning(f"Could not add built-in holidays: {e}")
            m2.fit(df[["ds","y"]])
            future2 = m2.make_future_dataframe(periods=horizon, freq=freq_for_future)
            fcst2 = m2.predict(future2)
            fig_f = m2.plot(fcst2); fig_f.set_size_inches(8,4); st.pyplot(fig_f)
            out_df = fcst2[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon)
        else:
            ts_all = df.set_index("ds")["y"]
            if model_choice == "Exponential Smoothing":
                model = ExponentialSmoothing(ts_all, trend="add",
                                             seasonal=("add" if freq_for_future in ["MS","M"] else None),
                                             seasonal_periods=(12 if freq_for_future in ["MS","M"] else None))
                fit = model.fit(optimized=True)
                future_index = pd.date_range(ts_all.index[-1] + pd.offsets.MonthBegin() if freq_for_future in ["MS","M"] else ts_all.index[-1] + pd.Timedelta(days=1),
                                             periods=horizon, freq=freq_for_future)
                yhat = fit.forecast(horizon)
            elif model_choice == "ARIMA":
                fit = ARIMA(ts_all, order=(1,1,1)).fit()
                future_index = pd.date_range(ts_all.index[-1] + pd.offsets.MonthBegin() if freq_for_future in ["MS","M"] else ts_all.index[-1] + pd.Timedelta(days=1),
                                             periods=horizon, freq=freq_for_future)
                yhat = fit.forecast(horizon)
            else:  # SARIMA
                m_seas = 12 if freq_for_future in ["MS","M"] else 7 if freq_for_future=="W" else 0
                fit = SARIMAX(ts_all, order=(1,1,1),
                              seasonal_order=((1,1,1,m_seas) if m_seas else (0,0,0,0)),
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                future_index = pd.date_range(ts_all.index[-1] + pd.offsets.MonthBegin() if freq_for_future in ["MS","M"] else ts_all.index[-1] + pd.Timedelta(days=1),
                                             periods=horizon, freq=freq_for_future)
                yhat = fit.forecast(horizon)
            fig, ax = plt.subplots(figsize=(7,3))
            ax.plot(ts_all.index, ts_all.values, label="Observed")
            ax.plot(future_index, yhat, label="Forecast", linestyle="--")
            ax.legend(); st.pyplot(fig)
            out_df = pd.DataFrame({"ds": future_index, "yhat": yhat})

        st.download_button("Download forecast CSV", out_df.to_csv(index=False),
                           file_name="forecast.csv", mime="text/csv")

    with st.expander("Guidance & notes"):
        st.markdown("""
- **Filtering & aggregation:** Choose dimensions to keep; aggregated to one value per date.
- **Resampling:** Use **Mean** for rates/averages; **Sum** for counts/volumes.
- **Prophet changepoints:** Higher prior scale / more changepoints → more flexible trend.
- **Seasonality mode:** *additive* when seasonal amplitude is constant; *multiplicative* when it grows with the level.
- **Holidays:** Use built-in country calendars or upload your own CSV (`ds`, `holiday`, optional windows); tune **holiday prior scale** to control influence.
- **Training window & metrics:** Everything after the end date is test. Pick MAE/RMSE/MAPE/R² as needed.
""")

# =========================
# Sidebar Navigation
# =========================
with st.sidebar:
    selected = option_menu(
        None,
        ["Home", "MKBF Forecast", "Root Cause Classification", "Forecast App"],
        icons=["house", "graph-up", "chat-text", "bar-chart-line"],
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "0.5rem", "background-color": "#f4f4f4", "border-radius": "16px"},
            "icon": {"color": "black", "font-size": "1.1rem"},
            "nav-link": {
                "font-size": "1rem", "color": "black",
                "padding": "0.6rem 0.8rem", "border-radius": "12px", "margin": "6px 8px",
            },
            "nav-link-selected": {"background-color": "#ffffff", "color": "black"},
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
elif selected == "Forecast App":
    page_forecast_app()
