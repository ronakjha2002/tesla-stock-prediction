# 📈 Tesla Stock Price Prediction
### Deep Learning with SimpleRNN & LSTM | Multi-Horizon Forecasting

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://tesla-stock-prediction-wesjc6bpmy8day8ubsg79e.streamlit.app/)
[![Domain](https://img.shields.io/badge/Domain-Financial%20Services-green)]()

---

## 🔍 Project Overview

This project builds and compares **SimpleRNN** and **LSTM** deep learning models to predict Tesla's (TSLA) stock closing price at three forecast horizons — **1 day**, **5 days**, and **10 days** ahead.

The dataset covers Tesla's historical stock prices from **June 2010 to February 2020** (2,416 trading days), sourced from Yahoo Finance.

A live interactive **Streamlit app** allows users to upload models and visualize predictions vs actual prices in real time.

---

## 🌐 Live Demo

👉 **[Open Streamlit App](https://tesla-stock-prediction-wesjc6bpmy8day8ubsg79e.streamlit.app/)**

Upload your model files and explore predictions interactively.

---

## 📊 Results Summary

| Model | Horizon | RMSE ($) | MAE ($) | R² | MAPE (%) |
|---|---|---|---|---|---|
| **SimpleRNN** | 1-day | **13.87** | 8.65 | **0.9627** | 2.74 |
| LSTM | 1-day | 22.23 | 14.72 | 0.9042 | 4.61 |
| **SimpleRNN** | 5-day | **28.56** | 20.16 | **0.8419** | 6.38 |
| LSTM | 5-day | 37.23 | 26.70 | 0.7727 | 8.45 |
| SimpleRNN | 10-day | 58.59 | 40.95 | 0.3345 | 13.35 |
| **LSTM** | **10-day** | **44.39** | 30.74 | **0.6181** | 9.58 |

> **Key finding:** SimpleRNN outperforms LSTM at short horizons (1-day, 5-day), while LSTM clearly wins at the 10-day horizon — confirming LSTM's advantage for long-range temporal dependencies.

---

## 🗂️ Project Structure

```
tesla-stock-prediction/
│
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
├── TSLA.xlsx                     # Tesla stock dataset (2010–2020)
├── Tesla_Stock_Prediction.ipynb  # Main Jupyter notebook
│
├── rnn_1day.keras                # Trained SimpleRNN model (1-day)
├── lstm_1day.keras               # Trained LSTM model (1-day)
├── rnn_5day.keras                # Trained SimpleRNN model (5-day)
├── lstm_5day.keras               # Trained LSTM model (5-day)
├── rnn_10day.keras               # Trained SimpleRNN model (10-day)
├── lstm_10day.keras              # Trained LSTM model (10-day)
└── scaler.pkl                    # Saved MinMaxScaler
```

---

## 🧠 Model Architecture

Both models use a **2-layer stacked architecture** with Dropout regularization:

```
Input (60 days × 1 feature)
        ↓
RNN/LSTM Layer (128 units, return_sequences=True)
        ↓
Dropout (0.2)
        ↓
RNN/LSTM Layer (64 units)
        ↓
Dropout (0.2)
        ↓
Dense (1) → Predicted Price
```

**Best hyperparameters** (found via Grid Search):
- Units: `128`
- Dropout: `0.2`
- Learning rate: `0.001`
- Optimizer: `Adam`
- Loss: `MSE`

---

## ⚙️ Pipeline

```
1. Data Loading       → TSLA.xlsx (2,416 rows, 7 columns)
2. EDA                → Price trends, volume, rolling stats, correlation
3. Feature Eng.       → Moving averages (MA7, MA21), Bollinger Bands
4. Preprocessing      → MinMaxScaler, 80/20 train-test split (no shuffle)
5. Sequence Creation  → 60-day lookback window, 3 forecast horizons
6. Hyperparameter     → Grid Search on LSTM (12 combinations)
   Tuning
7. Model Training     → 6 models: 2 architectures × 3 horizons
8. Evaluation         → RMSE, MAE, R², MAPE, Direction Accuracy
9. Deployment         → Streamlit Cloud
```

---

## 📦 Installation & Local Run

```bash
# 1. Clone the repository
git clone https://github.com/ronakjha2002/tesla-stock-prediction.git
cd tesla-stock-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## 📋 Requirements

```
streamlit
tensorflow-cpu
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
keras
```

---

## 📁 Dataset

| Column | Description |
|---|---|
| Date | Trading date |
| Open | Opening price |
| High | Daily high |
| Low | Daily low |
| Close | Closing price (target variable) |
| Adj Close | Adjusted closing price |
| Volume | Number of shares traded |

- **Source:** Yahoo Finance
- **Period:** June 29, 2010 — February 3, 2020
- **Records:** 2,416 trading days
- **Missing values:** None

---

## 💡 Key Insights

1. **LSTM wins at 10-day horizon** (R²=0.618 vs RNN's 0.334) — gating mechanism handles long-range dependencies better
2. **SimpleRNN surprisingly wins at 1-day** (R²=0.963) — simpler architecture avoids overfitting for short-term patterns
3. **Direction accuracy ~50%** across all models — predicting up/down is genuinely hard, consistent with financial literature
4. **Error compounds with horizon** — MAPE grows from 2.7% (1-day) to 13.4% (10-day) for SimpleRNN

---

## 🚀 Suggested Improvements

- Add **news sentiment analysis** using NLP on Tesla-related news
- Compare with **GRU** and **Transformer** architectures
- Include **technical indicators** (RSI, MACD) as multivariate features
- Use **Walk-Forward validation** for more realistic evaluation
- Incorporate **macroeconomic indicators** (interest rates, EV market trends)

---

## 👤 Author

**Ronak Jha**
M.Sc. Bioinformatics | ML Intern @ ACTREC – Tata Memorial Centre

[![GitHub](https://img.shields.io/badge/GitHub-ronakjha2002-black?logo=github)](https://github.com/ronakjha2002)

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock price predictions should not be used for actual trading or investment decisions. Past performance does not guarantee future results.
