import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1424;
    border-right: 1px solid #1e2640;
}

/* Title */
h1 {
    font-family: 'Space Mono', monospace !important;
    color: #e8eaf0 !important;
    letter-spacing: -1px;
}

h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #c8cfe0 !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin: 6px 0;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #5a6a8a;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #4fc3f7;
}

.metric-unit {
    font-size: 13px;
    color: #5a6a8a;
    margin-top: 2px;
}

/* Model badge */
.model-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    margin: 4px;
}

.badge-rnn  { background: #1a2535; border: 1px solid #e57373; color: #e57373; }
.badge-lstm { background: #1a2535; border: 1px solid #4fc3f7; color: #4fc3f7; }

/* Info box */
.info-box {
    background: #111827;
    border-left: 3px solid #4fc3f7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 14px;
    color: #9ba8c0;
    line-height: 1.7;
}

/* Warning box */
.warn-box {
    background: #1a1508;
    border-left: 3px solid #ffd54f;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 14px;
    color: #9ba060;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2640;
    margin: 20px 0;
}

/* Upload area */
.upload-hint {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #3a4a6a;
    text-align: center;
    padding: 8px;
}

/* Streamlit widget labels */
label {
    color: #8a9ab8 !important;
    font-size: 13px !important;
}

/* Select box */
.stSelectbox > div > div {
    background-color: #111827 !important;
    border-color: #1e2d4a !important;
    color: #e8eaf0 !important;
}

/* Radio */
.stRadio label {
    color: #c8cfe0 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    padding: 10px 24px;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    transform: translateY(-1px);
}

/* Matplotlib figure */
.stPlotlyChart, .stImage { border-radius: 12px; }
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
        'figure.facecolor':  '#0a0e1a',
        'axes.facecolor':    '#111827',
        'axes.edgecolor':    '#1e2640',
        'axes.labelcolor':   '#8a9ab8',
        'xtick.color':       '#5a6a8a',
        'ytick.color':       '#5a6a8a',
        'text.color':        '#c8cfe0',
        'grid.color':        '#1e2640',
        'grid.linewidth':    0.5,
        'legend.facecolor':  '#111827',
        'legend.edgecolor':  '#1e2640',
        'font.family':       'monospace',
    })


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 4px 0;'>
    <span style='font-family: Space Mono, monospace; font-size: 11px;
                 color: #4fc3f7; letter-spacing: 3px; text-transform: uppercase;'>
        Deep Learning · Financial Forecasting
    </span>
</div>
""", unsafe_allow_html=True)

st.title("Tesla Stock Price Predictor")

st.markdown("""
<div class='info-box'>
    Upload your <b>TSLA.csv</b> dataset and saved <b>.keras</b> model files + <b>scaler.pkl</b>
    using the sidebar. Then select a forecast horizon and model to visualize predictions.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 13px;
                color: #4fc3f7; letter-spacing: 2px; margin-bottom: 16px;'>
        ⚙ CONFIGURATION
    </div>
    """, unsafe_allow_html=True)

    # Dataset upload
    st.markdown("**📂 Dataset**")
    csv_file = st.file_uploader("Upload TSLA.csv", type=["csv","xlsx"], key="csv")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Model uploads
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

    # Settings
    st.markdown("**🎯 Prediction Settings**")
    horizon = st.selectbox("Forecast Horizon", ["1 day", "5 days", "10 days"])
    model_choice = st.radio("Model", ["SimpleRNN", "LSTM", "Both (compare)"])
    window_size = st.slider("Lookback Window (days)", 30, 90, 60, step=10)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Prediction")


# ── Main Logic ────────────────────────────────────────────────────────────────
if run_btn:
    # Validate uploads
    missing = []
    if not csv_file:    missing.append("TSLA.csv")
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
    if model_choice in ["SimpleRNN", "Both (compare)"]:
        needed_models.append("SimpleRNN")
    if model_choice in ["LSTM", "Both (compare)"]:
        needed_models.append("LSTM")

    for m in needed_models:
        if not model_files[(horizon_int, m)]:
            missing.append(f"{m} {horizon_int}-day model")

    if missing:
        st.error(f"Please upload: {', '.join(missing)}")
        st.stop()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
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

        # Load scaler
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            tmp.write(scaler_file.read())
            tmp_scaler_path = tmp.name
        scaler = joblib.load(tmp_scaler_path)
        os.unlink(tmp_scaler_path)

        scaled_data = scaler.transform(data)

        # Train/test split
        train_size  = int(len(scaled_data) * 0.80)
        train_data  = scaled_data[:train_size]
        test_data   = scaled_data[train_size:]
        full_data   = np.vstack([train_data, test_data])

        X_tr, y_tr = create_sequences(train_data, window_size, forecast_horizon=horizon_int)
        X_full, y_full = create_sequences(full_data, window_size, forecast_horizon=horizon_int)
        X_te = X_full[len(X_tr):]
        y_te = y_full[len(y_tr):]

        X_te_r = X_te.reshape(X_te.shape[0], X_te.shape[1], 1)

    # ── Load models & predict ─────────────────────────────────────────────────
    import tensorflow as tf

    predictions = {}
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
        pred = predictions[model_name]
        rmse  = math.sqrt(mean_squared_error(actual, pred))
        mae   = mean_absolute_error(actual, pred)
        r2    = r2_score(actual, pred)
        mape_ = mape(actual, pred)
        dir_  = direction_accuracy(actual, pred)

        badge_class = "badge-rnn" if model_name == "SimpleRNN" else "badge-lstm"
        with col:
            st.markdown(f"<div style='text-align:center; margin-bottom:12px;'>"
                        f"<span class='model-badge {badge_class}'>{model_name} · {horizon_int}-day</span>"
                        f"</div>", unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>RMSE</div>
                    <div class='metric-value'>${rmse:.1f}</div>
                    <div class='metric-unit'>USD error</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>R² Score</div>
                    <div class='metric-value'>{r2:.3f}</div>
                    <div class='metric-unit'>variance explained</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>MAE</div>
                    <div class='metric-value'>${mae:.1f}</div>
                    <div class='metric-unit'>avg error</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>MAPE</div>
                    <div class='metric-value'>{mape_:.1f}%</div>
                    <div class='metric-unit'>pct error</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='metric-card' style='margin-top:6px;'>
                <div class='metric-label'>Direction Accuracy</div>
                <div class='metric-value' style='color: {"#81c784" if dir_ > 52 else "#e57373"}'>{dir_:.1f}%</div>
                <div class='metric-unit'>correct up/down calls</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Prediction Plot ───────────────────────────────────────────────────────
    st.markdown("## 📈 Predicted vs Actual")

    set_dark_plot_style()
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(actual, label='Actual', color='#e8eaf0', linewidth=1.8, zorder=3)

    colors = {'SimpleRNN': '#e57373', 'LSTM': '#4fc3f7'}
    for model_name, pred in predictions.items():
        ax.plot(pred, label=model_name, color=colors[model_name],
                alpha=0.85, linewidth=1.4, zorder=2)

    ax.set_title(f'Tesla Close Price — {horizon_int}-Day Forecast', fontsize=13, pad=14)
    ax.set_xlabel('Test Set Time Steps')
    ax.set_ylabel('Close Price (USD)')
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Residuals Plot ────────────────────────────────────────────────────────
    st.markdown("## 🔍 Residuals (Prediction Error Over Time)")

    fig2, ax2 = plt.subplots(figsize=(13, 3.5))
    for model_name, pred in predictions.items():
        residuals = actual.flatten() - pred.flatten()
        ax2.plot(residuals, label=f'{model_name} Residuals',
                 color=colors[model_name], alpha=0.75, linewidth=1)
    ax2.axhline(0, color='#e8eaf0', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_title('Residuals Over Test Set', fontsize=12)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Error (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Full Historical + Test Prediction ────────────────────────────────────
    st.markdown("## 🗓 Full Historical Context")

    close_prices = data['Close'].values
    test_start_idx = len(close_prices) - len(actual)

    fig3, ax3 = plt.subplots(figsize=(13, 5))
    ax3.plot(range(len(close_prices)), close_prices,
             color='#3a4a6a', linewidth=1, label='Full History', alpha=0.6)
    ax3.plot(range(test_start_idx, test_start_idx + len(actual)),
             actual, color='#e8eaf0', linewidth=1.6, label='Actual (Test)')

    for model_name, pred in predictions.items():
        ax3.plot(range(test_start_idx, test_start_idx + len(pred)),
                 pred, color=colors[model_name],
                 linewidth=1.4, alpha=0.85, label=f'{model_name} Prediction')

    ax3.axvline(test_start_idx, color='#ffd54f', linewidth=1,
                linestyle='--', alpha=0.6, label='Train/Test Split')
    ax3.set_title('Tesla Stock — Full Historical Price with Test Predictions', fontsize=12)
    ax3.set_xlabel('Trading Days Since IPO (June 2010)')
    ax3.set_ylabel('Close Price (USD)')
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

    st.markdown(f"""
    <div class='info-box'>
        <b>Best model for {horizon_int}-day prediction:</b>
        <span class='model-badge {"badge-rnn" if best_model == "SimpleRNN" else "badge-lstm"}'>{best_model}</span><br><br>
        → Predictions are off by <b>${best_rmse:.1f}</b> on average (RMSE)<br>
        → Model explains <b>{best_r2*100:.1f}%</b> of price variance (R²)<br>
        → On a stock trading at ~$200–$780, a {best_rmse:.1f} USD RMSE represents
        approximately <b>{(best_rmse/data["Close"].tail(len(actual)).mean()*100):.1f}%</b> of average price.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='warn-box'>
        ⚠ <b>Disclaimer:</b> This tool is for educational purposes only.
        Stock price predictions should not be used for actual trading decisions.
        Past performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px;'>
        <div style='font-size: 64px; margin-bottom: 20px;'>📈</div>
        <div style='font-family: Space Mono, monospace; font-size: 16px;
                    color: #3a4a6a; letter-spacing: 2px;'>
            UPLOAD FILES → SELECT SETTINGS → RUN PREDICTION
        </div>
        <div style='margin-top: 24px; color: #2a3a5a; font-size: 13px; line-height: 2;'>
            Required files:<br>
            TSLA.csv &nbsp;·&nbsp; scaler.pkl<br>
            rnn_1day.keras &nbsp;·&nbsp; lstm_1day.keras<br>
            rnn_5day.keras &nbsp;·&nbsp; lstm_5day.keras<br>
            rnn_10day.keras &nbsp;·&nbsp; lstm_10day.keras
        </div>
    </div>
    """, unsafe_allow_html=True)
