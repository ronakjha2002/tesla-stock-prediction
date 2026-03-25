import matplotlib
matplotlib.use('Agg')
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #f0f2f8;
}

/* Main background */
.stApp {
    background: linear-gradient(160deg, #0d1117 0%, #0a0f1e 50%, #060c18 100%);
}

/* Sidebar - dark navy with clear text */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1520 100%) !important;
    border-right: 2px solid #1e3a5f !important;
}

section[data-testid="stSidebar"] * {
    color: #d0dff0 !important;
}

section[data-testid="stSidebar"] label {
    color: #90afd0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: #4fc3f7 !important;
    font-weight: 600 !important;
}

/* Sidebar file uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #0d2137 !important;
    border: 1.5px dashed #1e5080 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #7ab0d0 !important;
}

/* Sidebar selectbox and radio */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0d2137 !important;
    border: 1px solid #1e5080 !important;
    color: #d0dff0 !important;
}

section[data-testid="stSidebar"] .stRadio div {
    color: #d0dff0 !important;
}

section[data-testid="stSidebar"] .stSlider {
    color: #d0dff0 !important;
}

/* Sidebar divider */
section[data-testid="stSidebar"] hr {
    border-color: #1e3a5f !important;
}

/* Section labels in sidebar */
section[data-testid="stSidebar"] b {
    color: #4fc3f7 !important;
    font-size: 14px !important;
}

/* Main headings */
h1 {
    font-family: 'Space Mono', monospace !important;
    color: #ffffff !important;
    font-size: 2.4rem !important;
    letter-spacing: -1px !important;
    margin-bottom: 0 !important;
}

h2 {
    font-family: 'Space Mono', monospace !important;
    color: #e0eaf8 !important;
    font-size: 1.3rem !important;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
}

h3 {
    color: #b0c8e8 !important;
    font-size: 1.1rem !important;
}

/* General text */
p, li, span {
    color: #c8daf0 !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #0f2035 100%);
    border: 1px solid #1e3d6a;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    margin: 5px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #4a7a9a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #4fc3f7;
    line-height: 1.2;
}

.metric-unit {
    font-size: 11px;
    color: #4a7a9a;
    margin-top: 4px;
}

/* Model badge */
.model-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    margin: 4px;
}

.badge-rnn  { background: #1a1015; border: 1.5px solid #ff6b6b; color: #ff6b6b; }
.badge-lstm { background: #0a1525; border: 1.5px solid #4fc3f7; color: #4fc3f7; }

/* Info box */
.info-box {
    background: #0d1b2a;
    border-left: 3px solid #4fc3f7;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 14px;
    color: #a0c4e0 !important;
    line-height: 1.7;
}

.info-box b { color: #e0f0ff !important; }

/* Warning box */
.warn-box {
    background: #1a1508;
    border-left: 3px solid #ffd54f;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 13px;
    color: #c8b870 !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e3a5f;
    margin: 20px 0;
}

/* Run button */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    box-shadow: 0 6px 20px rgba(21, 101, 192, 0.6) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100

def direction_accuracy(actual, pred):
    actual = np.array(actual).flatten()
    pred   = np.array(pred).flatten()
    return (np.sign(np.diff(actual)) == np.sign(np.diff(pred))).mean() * 100

def create_sequences(dataset, window_size, forecast_horizon=1):
    X, y = [], []
    for i in range(len(dataset) - window_size - forecast_horizon + 1):
        X.append(dataset[i : i + window_size, 0])
        y.append(dataset[i + window_size + forecast_horizon - 1, 0])
    return np.array(X), np.array(y)

def set_dark_plot_style():
    plt.rcParams.update({
        'figure.facecolor':  '#0d1117',
        'axes.facecolor':    '#0d1b2a',
        'axes.edgecolor':    '#1e3a5f',
        'axes.labelcolor':   '#90afd0',
        'xtick.color':       '#607a90',
        'ytick.color':       '#607a90',
        'text.color':        '#d0dff0',
        'grid.color':        '#1e3a5f',
        'grid.linewidth':    0.6,
        'legend.facecolor':  '#0d1b2a',
        'legend.edgecolor':  '#1e3a5f',
        'legend.labelcolor': '#d0dff0',
        'font.family':       'monospace',
        'font.size':         10,
    })


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 4px 0;'>
    <span style='font-family: Space Mono, monospace; font-size: 11px;
                 color: #4fc3f7; letter-spacing: 4px; text-transform: uppercase;'>
        Deep Learning · Financial Forecasting
    </span>
</div>
""", unsafe_allow_html=True)

st.title("Tesla Stock Price Predictor")

st.markdown("""
<div class='info-box'>
    Upload your <b>TSLA dataset</b> (CSV or Excel) and saved <b>.keras</b> model files + <b>scaler.pkl</b>
    using the sidebar. Select a forecast horizon and model, then click <b>Run Prediction</b>.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**📂 Dataset**")
    csv_file = st.file_uploader("Upload TSLA file", type=["csv", "xlsx", "xls"], key="csv")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**🧠 Model Files (.keras)**")

    rnn_1_file   = st.file_uploader("RNN  — 1 day",  type=["keras"], key="rnn1")
    lstm_1_file  = st.file_uploader("LSTM — 1 day",  type=["keras"], key="lstm1")
    rnn_5_file   = st.file_uploader("RNN  — 5 day",  type=["keras"], key="rnn5")
    lstm_5_file  = st.file_uploader("LSTM — 5 day",  type=["keras"], key="lstm5")
    rnn_10_file  = st.file_uploader("RNN  — 10 day", type=["keras"], key="rnn10")
    lstm_10_file = st.file_uploader("LSTM — 10 day", type=["keras"], key="lstm10")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**⚖ Scaler**")
    scaler_file = st.file_uploader("Upload scaler.pkl", type=["pkl"], key="scaler")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**🎯 Prediction Settings**")
    horizon      = st.selectbox("Forecast Horizon", ["1 day", "5 days", "10 days"])
    model_choice = st.radio("Model", ["SimpleRNN", "LSTM", "Both (compare)"])
    window_size  = st.slider("Lookback Window (days)", 30, 90, 60, step=10)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Prediction")


# ── Main Logic ────────────────────────────────────────────────────────────────
if run_btn:
    missing = []
    if not csv_file:    missing.append("TSLA dataset")
    if not scaler_file: missing.append("scaler.pkl")

    horizon_int = int(horizon.split()[0])
    model_files = {
        (1,  "SimpleRNN"): rnn_1_file,
        (1,  "LSTM"):      lstm_1_file,
        (5,  "SimpleRNN"): rnn_5_file,
        (5,  "LSTM"):      lstm_5_file,
        (10, "SimpleRNN"): rnn_10_file,
        (10, "LSTM"):      lstm_10_file,
    }

    needed_models = []
    if model_choice in ["SimpleRNN", "Both (compare)"]: needed_models.append("SimpleRNN")
    if model_choice in ["LSTM",      "Both (compare)"]: needed_models.append("LSTM")

    for m in needed_models:
        if not model_files[(horizon_int, m)]:
            missing.append(f"{m} {horizon_int}-day model")

    if missing:
        st.error(f"⚠ Please upload: {', '.join(missing)}")
        st.stop()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading dataset..."):
        if csv_file.name.endswith('.csv'):
            df = pd.read_csv(csv_file)
        else:
            df = pd.read_excel(csv_file)

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        data = df[['Close']].copy()

        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp.write(scaler_file.read())
            tmp_scaler_path = tmp.name
        scaler = joblib.load(tmp_scaler_path)
        os.unlink(tmp_scaler_path)

        scaled_data = scaler.transform(data)
        train_size  = int(len(scaled_data) * 0.80)
        train_data  = scaled_data[:train_size]
        test_data   = scaled_data[train_size:]
        full_data   = np.vstack([train_data, test_data])

        X_tr, y_tr     = create_sequences(train_data, window_size, forecast_horizon=horizon_int)
        X_full, y_full = create_sequences(full_data,  window_size, forecast_horizon=horizon_int)
        X_te = X_full[len(X_tr):]
        y_te = y_full[len(y_tr):]
        X_te_r = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)

    # ── Load models & predict ─────────────────────────────────────────────────
    import tensorflow as tf

    predictions = {}
    colors = {'SimpleRNN': '#ff6b6b', 'LSTM': '#4fc3f7'}

    for model_name in needed_models:
        mf = model_files[(horizon_int, model_name)]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
            tmp.write(mf.read())
            tmp_model_path = tmp.name
        with st.spinner(f"Running {model_name} prediction..."):
            model = tf.keras.models.load_model(tmp_model_path)
            pred_scaled = model.predict(X_te_r, verbose=0)
            pred   = scaler.inverse_transform(pred_scaled)
            actual = scaler.inverse_transform(y_te.reshape(-1, 1))
            predictions[model_name] = pred
        os.unlink(tmp_model_path)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("## 📊 Model Performance")

    cols = st.columns(len(needed_models))
    for col, model_name in zip(cols, needed_models):
        pred  = predictions[model_name]
        rmse  = math.sqrt(mean_squared_error(actual, pred))
        mae   = mean_absolute_error(actual, pred)
        r2    = r2_score(actual, pred)
        mape_ = mape(actual, pred)
        dir_  = direction_accuracy(actual, pred)

        badge_class = "badge-rnn" if model_name == "SimpleRNN" else "badge-lstm"
        with col:
            st.markdown(f"<div style='text-align:center; margin-bottom:14px;'>"
                        f"<span class='model-badge {badge_class}'>{model_name} · {horizon_int}-day</span>"
                        f"</div>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>RMSE</div>
                    <div class='metric-value'>${rmse:.1f}</div>
                    <div class='metric-unit'>USD error</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>R² Score</div>
                    <div class='metric-value'>{r2:.3f}</div>
                    <div class='metric-unit'>variance explained</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>MAE</div>
                    <div class='metric-value'>${mae:.1f}</div>
                    <div class='metric-unit'>avg error</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>MAPE</div>
                    <div class='metric-value'>{mape_:.1f}%</div>
                    <div class='metric-unit'>pct error</div>
                </div>""", unsafe_allow_html=True)

            dir_color = "#4caf50" if dir_ > 52 else "#ef5350"
            st.markdown(f"""<div class='metric-card' style='margin-top:6px;'>
                <div class='metric-label'>Direction Accuracy</div>
                <div class='metric-value' style='color:{dir_color};'>{dir_:.1f}%</div>
                <div class='metric-unit'>correct up/down calls</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Prediction Plot ───────────────────────────────────────────────────────
    st.markdown("## 📈 Predicted vs Actual")
    set_dark_plot_style()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(actual, label='Actual', color='#ffffff', linewidth=2, zorder=3)
    for model_name, pred in predictions.items():
        ax.plot(pred, label=model_name, color=colors[model_name], alpha=0.85, linewidth=1.5)
    ax.set_title(f'Tesla Close Price — {horizon_int}-Day Forecast', fontsize=13, pad=14, color='#e0f0ff')
    ax.set_xlabel('Test Set Time Steps', color='#90afd0')
    ax.set_ylabel('Close Price (USD)', color='#90afd0')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Residuals ─────────────────────────────────────────────────────────────
    st.markdown("## 🔍 Prediction Error Over Time")

    fig2, ax2 = plt.subplots(figsize=(13, 3.5))
    for model_name, pred in predictions.items():
        residuals = actual.flatten() - pred.flatten()
        ax2.plot(residuals, label=f'{model_name} Error',
                 color=colors[model_name], alpha=0.8, linewidth=1.2)
    ax2.axhline(0, color='#ffffff', linewidth=1, linestyle='--', alpha=0.4)
    ax2.set_title('Residuals (Actual − Predicted)', fontsize=12, color='#e0f0ff')
    ax2.set_xlabel('Time Steps', color='#90afd0')
    ax2.set_ylabel('Error (USD)', color='#90afd0')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Full Historical ───────────────────────────────────────────────────────
    st.markdown("## 🗓 Full Historical Context")

    close_prices   = data['Close'].values
    test_start_idx = len(close_prices) - len(actual)

    fig3, ax3 = plt.subplots(figsize=(13, 5))
    ax3.plot(range(len(close_prices)), close_prices,
             color='#2a4a6a', linewidth=1, label='Full History', alpha=0.7)
    ax3.plot(range(test_start_idx, test_start_idx + len(actual)),
             actual, color='#ffffff', linewidth=1.8, label='Actual (Test)')
    for model_name, pred in predictions.items():
        ax3.plot(range(test_start_idx, test_start_idx + len(pred)),
                 pred, color=colors[model_name], linewidth=1.5,
                 alpha=0.85, label=f'{model_name} Prediction')
    ax3.axvline(test_start_idx, color='#ffd54f', linewidth=1.2,
                linestyle='--', alpha=0.7, label='Train/Test Split')
    ax3.set_title('Tesla Stock — Full History with Test Predictions', fontsize=12, color='#e0f0ff')
    ax3.set_xlabel('Trading Days Since IPO (June 2010)', color='#90afd0')
    ax3.set_ylabel('Close Price (USD)', color='#90afd0')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("## 💡 Interpretation")

    best_model = min(predictions.keys(),
                     key=lambda m: math.sqrt(mean_squared_error(actual, predictions[m])))
    best_rmse  = math.sqrt(mean_squared_error(actual, predictions[best_model]))
    best_r2    = r2_score(actual, predictions[best_model])
    avg_price  = data['Close'].tail(len(actual)).mean()

    st.markdown(f"""
    <div class='info-box'>
        <b>Best model for {horizon_int}-day prediction:</b>
        <span class='model-badge {"badge-rnn" if best_model == "SimpleRNN" else "badge-lstm"}'>{best_model}</span><br><br>
        → Average prediction error: <b>${best_rmse:.1f} RMSE</b><br>
        → Model explains <b>{best_r2*100:.1f}%</b> of price variance (R²)<br>
        → Error as % of average price: <b>{(best_rmse/avg_price*100):.1f}%</b>
    </div>
    <div class='warn-box'>
        ⚠ <b>Disclaimer:</b> This tool is for educational purposes only.
        Do not use predictions for actual trading decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 50px 20px;'>
            <div style='font-size: 72px; margin-bottom: 24px;'>📈</div>
            <div style='font-family: Space Mono, monospace; font-size: 14px;
                        color: #2a6aaa; letter-spacing: 3px; margin-bottom: 24px;'>
                UPLOAD → CONFIGURE → PREDICT
            </div>
            <div style='color: #2a5a8a; font-size: 13px; line-height: 2.2;
                        font-family: Space Mono, monospace;'>
                TSLA.csv / TSLA.xlsx<br>
                scaler.pkl<br>
                rnn_1day.keras · lstm_1day.keras<br>
                rnn_5day.keras · lstm_5day.keras<br>
                rnn_10day.keras · lstm_10day.keras
            </div>
        </div>
        """, unsafe_allow_html=True)
