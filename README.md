# 📈 Tesla Stock Price Prediction
### Deep Learning with SimpleRNN & LSTM | Multi-Horizon Forecasting

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://tesla-stock-prediction-wesjc6bpmy8day8ubsg79e.streamlit.app/)
[![Domain](https://img.shields.io/badge/Domain-Financial%20Services-green)]()
[![Open in nbviewer](https://img.shields.io/badge/View%20Notebook-nbviewer-orange)](https://nbviewer.org/github/ronakjha2002/tesla-stock-prediction/blob/main/Tesla_Stock_Prediction.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/ronakjha2002/tesla-stock-prediction/blob/main/Tesla_Stock_Prediction.ipynb)

---

## 🔍 Project Overview

This project builds and compares **SimpleRNN** and **LSTM** deep learning models to predict Tesla's (TSLA) stock closing price at three forecast horizons — **1 day**, **5 days**, and **10 days** ahead.

The dataset covers Tesla's historical stock prices from **June 2010 to February 2020** (2,416 trading days), sourced from Yahoo Finance.

A live interactive **Streamlit app** allows users to upload models and visualize predictions vs actual prices in real time.

---

## 🌐 Live Demo

👉 **[Open Streamlit App](https://tesla-stock-prediction-rj.streamlit.app/)**

Upload your model files and explore predictions interactively.

---

## 📊 Results Summary

| Model | Horizon | RMSE ($) | MAE ($) | R² | MAPE (%) | Dir Acc (%) |
|---|---|---|---|---|---|---|
| **SimpleRNN** | 1-day | **16.39** | 10.47 | **0.9479** | 3.32 | 50.1 |
| LSTM | 1-day | 24.08 | 16.17 | 0.8876 | 5.05 | 51.3 |
| **SimpleRNN** | 5-day | **28.70** | 20.36 | **0.8403** | 6.47 | 48.2 |
| LSTM | 5-day | 34.66 | 24.06 | 0.7672 | 7.52 | 52.4 |
| **SimpleRNN** | 10-day | **41.43** | 29.98 | **0.6673** | 9.47 | 52.4 |
| LSTM | 10-day | 44.49 | 30.58 | 0.6164 | 9.52 | 50.9 |

> **Key finding:** SimpleRNN outperforms LSTM across **all three horizons** in this run. Best models per horizon: **1-day** SimpleRNN (RMSE=$16.39, R²=0.9479), **5-day** SimpleRNN (RMSE=$28.70, R²=0.8403), **10-day** SimpleRNN (RMSE=$41.43, R²=0.6673). Direction accuracy ~49–52% across all models is consistent with financial literature.

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

**Best hyperparameters** (found via Grid Search over 12 combinations):
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
- **Missing values:** None (forward-fill applied as precaution)

---

## 💡 Key Insights

1. **SimpleRNN wins across all horizons** in this run — outperforms LSTM at 1-day (R²=0.948), 5-day (R²=0.840), and 10-day (R²=0.667)
2. **LSTM underperforms here** — likely due to overfitting on this relatively small dataset; simpler architecture generalizes better
3. **Both models achieve strong short-term accuracy** — 1-day MAPE of 3.32% (RNN) is practically useful
4. **Direction accuracy ~49–52%** across all models — essentially a coin flip, consistent with financial literature
5. **Error compounds with horizon** — MAPE grows from 3.3% (1-day) → 6.5% (5-day) → 9.5% (10-day) for SimpleRNN

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
