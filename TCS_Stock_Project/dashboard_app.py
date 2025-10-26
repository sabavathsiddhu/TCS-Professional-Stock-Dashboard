# dashboard_app.py
"""
Professional TCS Stock Dashboard
By: Sabavath Siddhu

Features:
- Polished gradient UI with glass cards
- File fallback: local CSV or live Yahoo Finance fetch
- Interactive filters (date range, aggregate frequency, ticker)
- Daily / Monthly / Yearly / Forecast / Insights / Model tabs
- Many charts: line, candlestick, bar, pie, heatmap, box, scatter
- Forecast generation and model training (linear/regression)
- Export: CSV / PNG / PDF for charts (kaleido recommended)
- Downloadable report snippets
- Logging and helpful messages
"""

import os
import io
import json
from datetime import datetime, timedelta
import tempfile
import math

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import joblib
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image, ImageOps

# -------------------------
# CONFIG / PATHS / GLOBALS
# -------------------------
BASE_DIR = r"C:\Users\palla shanmithaReddy\OneDrive\Documents\projects\TCS_Stock_Project\data"
STOCK_CSV = os.path.join(BASE_DIR, "TCS_stock_clean_final.csv")
FORECAST_CSV = os.path.join(BASE_DIR, "TCS_future_30days_forecast_with_CI.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "tcs_model.pkl")
DEFAULT_TICKER = "TCS.NS"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# STREAMLIT PAGE SETUP & THEME
# -------------------------
st.set_page_config(page_title="TCS Professional Dashboard — Sabavath Siddhu",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 50%, #9be7ff 100%);
        color: #0b1220;
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    }

    /* Glass card */
    .glass {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 8px 30px rgba(10, 15, 25, 0.3);
        border: 1px solid rgba(255,255,255,0.06);
        backdrop-filter: blur(6px);
    }

    /* KPI style */
    .kpi-title { color: rgba(255,255,255,0.85); font-size:14px; }
    .kpi-value { color:#fff; font-size:20px; font-weight:700; }

    /* footer */
    .footer { text-align:center; color: rgba(255,255,255,0.95); padding-top:8px; font-size:0.9rem; }

    /* smaller fonts for sidebars */
    .small { font-size:0.9rem; color: rgba(255,255,255,0.95); }
    </style>
    """, unsafe_allow_html=True
)

st.title("✨ TCS Professional Stock Dashboard")
st.markdown("**By — Sabavath Siddhu** · Interactive analytics with visualization, forecasting, and model training")

# -------------------------
# HELPERS
# -------------------------
@st.cache_data(ttl=300)
def load_local_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Basic expectations: Date, Open, High, Low, Close, Volume
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # try common alternatives
        for col in df.columns:
            if 'date' in col.lower():
                df.rename(columns={col:'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                break
    return df.sort_values('Date').reset_index(drop=True)

@st.cache_data(ttl=180)
def fetch_yfinance(ticker_symbol, period="1y", interval="1d"):
    try:
        df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        if 'Adj Close' in df.columns and 'Close' not in df.columns:
            df['Close'] = df['Adj Close']
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return None


def add_features(df):
    df = df.copy()

    # Ensure Close exists
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    # Convert numeric columns safely
    def to_numeric(series):
        return (
            series.astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)  # remove non-numeric chars
            .replace("", np.nan)
            .astype(float)
        )

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    # Sort and calculate indicators
    df = df.sort_values('Date').reset_index(drop=True)
    df['MA_7'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA_30'] = df['Close'].rolling(window=30, min_periods=1).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    if {'High', 'Low'}.issubset(df.columns):
        df['Volatility'] = df['High'] - df['Low']

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    return df


def ensure_forecast_columns(forecast_df):
    if forecast_df is None or forecast_df.empty:
        return forecast_df
    if 'Upper_Bound' not in forecast_df.columns or 'Lower_Bound' not in forecast_df.columns:
        forecast_df['Upper_Bound'] = forecast_df['Predicted_Close'] * 1.05
        forecast_df['Lower_Bound'] = forecast_df['Predicted_Close'] * 0.95
    return forecast_df

def compute_simple_forecast(df, days=30):
    df = df.copy().dropna().reset_index(drop=True)
    last_date = df['Date'].max()
    mu = df['Close'].pct_change().mean()
    val = df['Close'].iloc[-1]
    preds = []
    future_dates = []
    for i in range(1, days+1):
        val = val * (1 + (mu or 0))
        preds.append(val)
        future_dates.append(last_date + timedelta(days=i))
    fdf = pd.DataFrame({'Date': future_dates, 'Predicted_Close': preds})
    fdf['Upper_Bound'] = fdf['Predicted_Close'] * 1.05
    fdf['Lower_Bound'] = fdf['Predicted_Close'] * 0.95
    return fdf

def plotly_to_png_bytes(fig, width=1200, height=600, scale=2):
    try:
        return pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    except Exception:
        return None

def png_to_pdf_bytes(png_bytes):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

# -------------------------
# SIDEBAR: Controls and Upload
# -------------------------
st.sidebar.header("Controls")
data_source = st.sidebar.selectbox("Data source", options=["Local CSV", "Yahoo Finance (Live)"])
if data_source == "Local CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
    if uploaded is not None:
        try:
            df_uploaded = pd.read_csv(uploaded)
            df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'])
            st.sidebar.success("Uploaded CSV loaded")
            DATA_DF = df_uploaded
        except Exception as e:
            st.sidebar.error("Upload failed: " + str(e))
            DATA_DF = None
    else:
        DATA_DF = None
else:
    DATA_DF = None

use_live = (data_source == "Yahoo Finance (Live)")
ticker = st.sidebar.text_input("Ticker (for Yahoo)", value=DEFAULT_TICKER if use_live else DEFAULT_TICKER)
history_days = st.sidebar.slider("History (days) for live fetch", min_value=90, max_value=3650, value=365)
date_min_default = datetime.now().date() - timedelta(days=history_days)
date_range = st.sidebar.date_input("Date range", value=(date_min_default, datetime.now().date()))
show_ma = st.sidebar.checkbox("Show moving averages (MA 7/30)", value=True)
export_png = st.sidebar.checkbox("Enable PNG export (kaleido)", value=True)
train_model_flag = st.sidebar.checkbox("Enable model training in Insights tab", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Put local CSV as:\n`data/TCS_stock_clean_final.csv` or upload here.")

# -------------------------
# LOAD DATA
# -------------------------
if DATA_DF is not None:
    df = DATA_DF.copy()
else:
    if use_live:
        df = fetch_yfinance(ticker, period=f"{history_days}d", interval="1d")
    else:
        df = load_local_csv(STOCK_CSV)

if df is None or df.empty:
    st.error(f"Data not found. Put CSV at: {STOCK_CSV} or upload a CSV or enable Yahoo Live.")
    st.stop()

# Add features
try:
    df = add_features(df)
except Exception as e:
    st.error("Failed to add features: " + str(e))
    st.stop()

# Apply date range filter
start_d, end_d = date_range
mask = (df['Date'].dt.date >= start_d) & (df['Date'].dt.date <= end_d)
df_view = df.loc[mask].copy()
if df_view.empty:
    st.warning("No data in selected date range — showing full dataset.")
    df_view = df.copy()
# -------------------------
# LOAD DATA
# -------------------------
if DATA_DF is not None:
    df = DATA_DF.copy()
else:
    df = load_local_csv(STOCK_CSV)

# Fallback to Yahoo Finance if no CSV found
if df is None or df.empty:
    st.warning("⚠️ Local CSV not found — fetching live data from Yahoo Finance.")
    df = fetch_yfinance(DEFAULT_TICKER, period="1y", interval="1d")

if df is None or df.empty:
    st.error("❌ Unable to load any data — please check network or CSV file.")
    st.stop()

# -------------------------
# HEADER KPIs (glass cards)
# -------------------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns([1.4,1,1,1,1])
with k1:
    st.markdown("<div class='kpi-title'>Date Range</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>{df_view['Date'].dt.date.min()} → {df_view['Date'].dt.date.max()}</div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='kpi-title'>Avg Close</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>₹{df_view['Close'].mean():,.2f}</div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='kpi-title'>Max Close</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>₹{df_view['Close'].max():,.2f}</div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='kpi-title'>Latest Close</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>₹{df_view['Close'].iloc[-1]:,.2f}</div>", unsafe_allow_html=True)
with k5:
    st.markdown("<div class='kpi-title'>Total Volume</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-value'>{int(df_view['Volume'].sum()):,}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# MAIN TABS
# -------------------------
tabs = st.tabs(["Daily", "Monthly", "Yearly", "Forecast", "Insights", "Model"])

# ---------- DAILY TAB ----------
with tabs[0]:
    st.header("📅 Daily Analysis")
    c1, c2 = st.columns((3,1))

    with c1:
        st.subheader("Close Price (interactive)")
        fig = px.line(df_view, x='Date', y='Close', title="Daily Close Price", labels={'Close':"Price (₹)"}, template='plotly_dark')
        if show_ma and 'MA_7' in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['MA_7'], name='MA 7', line=dict(dash='dash', color='orange')))
        if show_ma and 'MA_30' in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['MA_30'], name='MA 30', line=dict(dash='dot', color='magenta')))
        fig.update_layout(legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Candlestick (last 120 days)")
        cs = df_view.tail(120)
        fig_c = go.Figure(data=[go.Candlestick(x=cs['Date'], open=cs['Open'], high=cs['High'], low=cs['Low'], close=cs['Close'])])
        fig_c.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        st.subheader("Quick Stats")
        st.write("Distribution & recent values")
        st.metric("Latest Close", f"₹{df_view['Close'].iloc[-1]:,.2f}", delta=f"{df_view['Close'].pct_change().iloc[-1]*100:.2f}%")
        st.write("Close — descriptive stats")
        st.dataframe(df_view['Close'].describe().round(2))

# ---------- MONTHLY TAB ----------
with tabs[1]:
    st.header("📆 Monthly Analysis")
    monthly = df_view.groupby(df_view['Date'].dt.to_period('M')).agg({'Close':'mean','Volume':'sum','Daily_Return':'mean'}).reset_index()
    monthly['Month'] = monthly['Date'].dt.to_timestamp()

    m1, m2 = st.columns(2)
    with m1:
        st.subheader("Average Close by Month")
        fig_m = px.bar(monthly, x='Month', y='Close', title="Monthly Average Close", labels={'Close':'Avg Close (₹)'}, template='plotly_dark', color='Close')
        st.plotly_chart(fig_m, use_container_width=True)
    with m2:
        st.subheader("Monthly Volume (stacked)")
        fig_mv = px.bar(monthly, x='Month', y='Volume', title="Monthly Total Volume", labels={'Volume':'Volume'}, template='plotly_dark')
        st.plotly_chart(fig_mv, use_container_width=True)

    st.subheader("Month-on-Month Growth")
    monthly['MoM_pct'] = monthly['Close'].pct_change() * 100
    fig_mm = px.line(monthly, x='Month', y='MoM_pct', title="Month-on-Month % Growth", markers=True, template='plotly_dark')
    st.plotly_chart(fig_mm, use_container_width=True)

# ---------- YEARLY TAB ----------
with tabs[2]:
    st.header("📈 Yearly Analysis & Pie Charts")
    yearly = df_view.groupby(df_view['Date'].dt.year).agg({'Close':'mean','Volume':'sum','Daily_Return':'mean'}).reset_index().rename(columns={'Date':'Year'})

    y1, y2 = st.columns(2)
    with y1:
        st.subheader("Yearly Avg Close")
        fig_y = px.bar(yearly, x='Year', y='Close', title="Yearly Average Close", template='plotly_dark', color='Close')
        st.plotly_chart(fig_y, use_container_width=True)
    with y2:
        st.subheader("Yearly Volume Share (Pie)")
        fig_p = px.pie(yearly, names='Year', values='Volume', title="Yearly Volume Contribution", template='plotly_dark')
        st.plotly_chart(fig_p, use_container_width=True)

    st.subheader("Yearly Return Distribution (Box)")
    fig_box = px.box(df_view, x=df_view['Date'].dt.year, y='Daily_Return', title="Daily Return Distribution by Year", template='plotly_dark')
    st.plotly_chart(fig_box, use_container_width=True)

# ---------- FORECAST TAB ----------
with tabs[3]:
    st.header("🔮 Forecast")
    st.write("Simple drift forecast, and model-based forecast (if trained).")

    # compute simple forecast
    forecast_simple = compute_simple_forecast(df_view, days=30)
    forecast_simple = ensure_forecast_columns(forecast_simple)

    colf1, colf2 = st.columns([3,1])
    with colf1:
        figf = go.Figure()
        figf.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Close'], name='Historical', line=dict(color='#7bdff6')))
        figf.add_trace(go.Scatter(x=forecast_simple['Date'], y=forecast_simple['Predicted_Close'], name='Simple Forecast', line=dict(color='#ffd56b', dash='dot')))
        figf.add_trace(go.Scatter(
            x=list(forecast_simple['Date']) + list(forecast_simple['Date'][::-1]),
            y=list(forecast_simple['Upper_Bound']) + list(forecast_simple['Lower_Bound'][::-1]),
            fill='toself', fillcolor='rgba(255,213,110,0.12)', line=dict(color='rgba(255,255,255,0)'), name='CI'
        ))
        figf.update_layout(template='plotly_dark', height=520)
        st.plotly_chart(figf, use_container_width=True)
    with colf2:
        st.subheader("Forecast Metrics")
        st.metric("Avg Predicted (30d)", f"₹{forecast_simple['Predicted_Close'].mean():,.2f}")
        st.metric("Max Predicted", f"₹{forecast_simple['Predicted_Close'].max():,.2f}")
        st.metric("Min Predicted", f"₹{forecast_simple['Predicted_Close'].min():,.2f}")
        st.download_button("📥 Download Forecast CSV", data=forecast_simple.to_csv(index=False).encode('utf-8'),
                           file_name="TCS_forecast_simple_30d.csv", mime="text/csv")

    # If trained model exists, allow model forecast
    trained_model = load_model(MODEL_FILE)
    if trained_model is not None:
        st.success("Loaded trained model — showing model-based forecast")
        # attempt model-based forecast using last row features if possible
        try:
            X_like = None
            feat_names = getattr(trained_model, "feature_names_in_", None)
            if feat_names is not None:
                # if these features exist in df
                if set(feat_names).issubset(set(df_view.columns)):
                    X_like = df_view[feat_names].iloc[-1:].values
            # do iterative predict if X_like available
            preds = []
            if X_like is not None:
                base = X_like.copy()
                for i in range(30):
                    p = float(trained_model.predict(base)[0])
                    preds.append(p)
                    # if 'Close' present in features, update it
                    if 'Close' in feat_names:
                        idx = list(feat_names).index('Close')
                        base[0, idx] = p
                future_dates = [df_view['Date'].max() + timedelta(days=i) for i in range(1,31)]
                model_forecast = pd.DataFrame({'Date': future_dates, 'Predicted_Close': preds})
                model_forecast['Upper_Bound'] = model_forecast['Predicted_Close'] * 1.05
                model_forecast['Lower_Bound'] = model_forecast['Predicted_Close'] * 0.95

                figf2 = go.Figure()
                figf2.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Close'], name='Historical', line=dict(color='#7bdff6')))
                figf2.add_trace(go.Scatter(x=model_forecast['Date'], y=model_forecast['Predicted_Close'], name='Model Forecast', line=dict(color='#ff7b7b')))
                figf2.update_layout(template='plotly_dark', height=420)
                st.plotly_chart(figf2, use_container_width=True)
            else:
                st.info("Model exists but required features not available for iterative forecasting.")
        except Exception as e:
            st.error("Model-based forecast failed: " + str(e))

# ---------- INSIGHTS TAB ----------
with tabs[4]:
    st.header("💡 Insights")
    st.markdown("Automated insights and correlations")

    # Correlation heatmap
    numeric = df_view.select_dtypes(include=[np.number]).drop(columns=['Year'], errors='ignore')
    if not numeric.empty:
        corr = numeric.corr()
        st.subheader("Correlation Matrix")
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title="Feature Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

    # PCA projection for dimensionality check
    if numeric.shape[1] >= 2:
        st.subheader("PCA (2 components)")
        features = numeric.dropna().drop(columns=[], errors='ignore')
        try:
            scaler = StandardScaler()
            Z = scaler.fit_transform(features.fillna(0))
            pca = PCA(n_components=2)
            comp = pca.fit_transform(Z)
            pca_df = pd.DataFrame(comp, columns=['PC1','PC2'])
            pca_df['Date'] = df_view['Date'].values[-len(pca_df):]
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', hover_data=['Date'], title="PCA Projection")
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.info("PCA failed: " + str(e))

    # Top movers
    st.subheader("Top daily movers (by absolute return)")
    movers = df_view.copy()
    movers['abs_ret'] = movers['Daily_Return'].abs()
    top = movers.nlargest(10, 'abs_ret')[['Date','Daily_Return']].round(2)
    st.dataframe(top)

# ---------- MODEL TAB ----------
with tabs[5]:
    st.header("🧠 Model Training & Feature Importance")
    st.markdown("Train simple regression models and inspect performance & feature importance.")

    st.subheader("Model selection & options")
    model_type = st.selectbox("Choose model", options=["LinearRegression", "Ridge", "RandomForest"])
    test_size = st.slider("Test size (fraction)", min_value=0.05, max_value=0.5, value=0.2)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42)
    features_to_use = st.multiselect("Features to use (default: Open, High, Low, Volume, MA_7, MA_30)",
                                     options=[c for c in df_view.columns if c not in ['Date','Year','Month']],
                                     default=[c for c in ['Open','High','Low','Volume','MA_7','MA_30'] if c in df_view.columns])

    target_col = st.selectbox("Target column", options=[c for c in df_view.columns if c not in ['Date'] and df_view[c].dtype != 'O'], index= df_view.columns.get_loc('Close') if 'Close' in df_view.columns else 0)

    if st.button("Train model"):
        if not features_to_use:
            st.warning("Select at least one feature.")
        else:
            X = df_view[features_to_use].fillna(method='ffill').fillna(0)
            y = df_view[target_col].fillna(method='ffill').fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            if model_type == "LinearRegression":
                model = LinearRegression()
            elif model_type == "Ridge":
                model = Ridge()
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.success(f"Trained {model_type} — MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
            save_model(model, MODEL_FILE)
            st.info(f"Model saved to: {MODEL_FILE}")

            # Feature importance (for tree) or coefficients
            st.subheader("Feature importance / coefficients")
            if model_type in ["RandomForest"]:
                try:
                    importances = model.feature_importances_
                    fi = pd.DataFrame({'feature':features_to_use, 'importance': importances}).sort_values('importance', ascending=False)
                    fig_fi = px.bar(fi, x='feature', y='importance', title='Feature Importance (RandomForest)', template='plotly_dark')
                    st.plotly_chart(fig_fi, use_container_width=True)
                except Exception as e:
                    st.info("Failed to compute feature importances: " + str(e))
            else:
                coefs = None
                try:
                    coefs = pd.Series(model.coef_, index=features_to_use).sort_values(key=abs, ascending=False)
                    fig_coef = px.bar(coefs.reset_index().rename(columns={'index':'feature',0:'coef'}), x='index', y=0, title='Model Coefficients')
                    st.plotly_chart(fig_coef, use_container_width=True)
                except Exception:
                    st.write("Could not show coefficients.")

            # Predictions scatter
            st.subheader("Actual vs Predicted (test set)")
            fig_sc = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, title='Actual vs Predicted', template='plotly_dark')
            st.plotly_chart(fig_sc, use_container_width=True)

            # Permutation importance (expensive)
            if model_type != "RandomForest" and st.checkbox("Compute permutation importance (slower)"):
                try:
                    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
                    perm = pd.DataFrame({'feature': features_to_use, 'importance': r.importances_mean}).sort_values('importance', ascending=False)
                    st.bar_chart(perm.set_index('feature'))
                except Exception as e:
                    st.info("Permutation importance failed: " + str(e))

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("<div class='footer'>Professional TCS Dashboard • Built by <strong>Sabavath Siddhu</strong> 💻 | Streamlit + Plotly</div>", unsafe_allow_html=True)
