# 📈 Stock Volatility Predictor
### Implementing Gradient Descent & Ridge Regression from Scratch, and Checking My Work Against Sklearn

## 🚀 Overview

This project started as an attempt to take the linear regression and gradient descent math I'd learned in theory and actually implement it myself, rather than just calling `sklearn.linear_model.LinearRegression()`. The goal wasn't to build the most accurate volatility model possible — it was to make sure I genuinely understood what's happening under the hood of a basic ML training loop: the gradients, the update rule, and what regularization actually does to the loss surface.

Along the way, a few things didn't work the way I expected on the first try, and figuring out *why* taught me more than if everything had just matched immediately. I've tried to document those moments honestly below, rather than only showing the final clean result.

## 🎯 Problem Statement

Predict next-day realized volatility of the S&P 500 using its own recent volatility history. This relies on a known property of markets called volatility clustering — calm and turbulent periods tend to persist for a while — so recent volatility is at least somewhat informative about tomorrow's.

- **Inputs (X):** the previous 5 days of rolling 21-day realized volatility
- **Target (y):** next day's realized volatility
- **Data:** S&P 500 (`^GSPC`) daily prices via `yfinance`

## 🧠 Concepts I Was Practicing

**Linear Regression**
```
y = Xw + b
```

**Batch Gradient Descent**
```
w := w - α · ∇J(w)
b := b - α · ∂J/∂b
```
Written by hand in NumPy. `scikit-learn` is only used afterward, to check my implementation against a known-correct reference — never during training.

**Ridge Regression (L2 Regularization)**
```
J(w) = MSE(y, ŷ) + λ‖w‖²
```
Added after noticing my weights weren't matching sklearn's (see below) — this wasn't planned from the start, it came out of debugging.

**Bias-Variance Tradeoff** — I wanted to actually see this rather than just recite the definition, so I swept several λ values and measured test error at each one (see Results).

## 🏗️ Project Structure

```
stock-volatility-predictor/
│
├── data/
│   └── raw/                  # Downloaded market data (gitignored)
│
├── src/
│   ├── data_loader.py        # Fetch, clean, cache S&P 500 data
│   ├── features.py           # Log returns, rolling volatility, lag features
│   ├── model.py               # Gradient descent + Ridge, from scratch
│   ├── evaluate.py           # Metrics + sklearn comparison + plots
│   └── predict.py            # Load a trained model, forecast the next trading day
│
├── main.py                   # End-to-end pipeline: data → train → tune → evaluate → save
├── outputs/
│   └── trained_model.npz     # Saved weights/bias/normalization stats
├── requirements.txt
└── README.md
```

## ⚙️ Pipeline

```
Raw Prices → Log Returns → Rolling Volatility → Lag Features
    → Chronological Train/Test Split → Feature Standardization (train stats only)
    → Gradient Descent (λ sweep) → Evaluation vs Sklearn → Save Model → Predict
```

---

## 🔬 A Few Things I Got Wrong First, and What They Taught Me

I'm including this section because I think the mistakes were more instructive than the final working version.

### 1. My weights didn't match sklearn's, and I had to figure out why

My first working gradient descent run produced predictions close to sklearn's `LinearRegression`, but the actual weight values were noticeably different:

| | vol_lag_1 | vol_lag_2 | vol_lag_3 | vol_lag_4 | vol_lag_5 |
|---|---|---|---|---|---|
| Custom GD | 0.119 | 0.037 | 0.001 | -0.015 | -0.029 |
| Sklearn OLS | 0.165 | -0.013 | -0.018 | -0.001 | -0.019 |

My first instinct was that I'd made a bug somewhere. After reading more into it, I learned this is a known effect of multicollinearity: my 5 lagged volatility features are highly correlated with each other, so the loss surface isn't a clean bowl with one minimum — it's a flat, elongated valley where many different weight combinations give almost the same loss. Both models had found valid, low-loss solutions; they just landed in different spots along that valley. This is the actual textbook reason people reach for regularization, which is what pushed me to implement Ridge next.

### 2. My "same lambda" comparison against sklearn's Ridge was actually invalid

Once I added Ridge, I compared my custom weights to `sklearn.linear_model.Ridge` at what I thought was the same regularization strength. Sklearn's weights barely moved from its unregularized version, while mine shrank a lot — a sign that "same λ" didn't mean the same thing to both implementations.

Working through both loss functions explicitly showed why: my loss averages squared error over `n` samples, while sklearn's Ridge sums it. With around 1,600 training rows, my λ was effectively ~1,600x stronger, relatively, than the same-numbered α in sklearn. Once I scaled my λ by `n_train` to compare fairly, my weights and sklearn's Ridge matched to five decimal places. This was a useful reminder that a hyperparameter with the same name isn't automatically doing the same thing across two implementations — I had to actually derive both formulas to check.

### 3. The model lags at sudden volatility spikes

Looking at predicted-vs-actual plots, the model tracks overall volatility levels reasonably well, but tends to lag by roughly a day whenever there's a sharp jump. This makes sense once I thought about it: the model only ever sees past volatility, so it can extrapolate recent momentum but has no way to anticipate a shock before its inputs reflect it. I don't think this is fixable within this model's design — it would need forward-looking information (options-implied volatility, news) or a different model class (e.g. GARCH) to do better here.

---

## 📊 Results

### Ridge Lambda Sweep (5,000 epochs, chronological 80/20 split)

| λ (Lambda) | Test MSE |
|---|---|
| 0.0 (no regularization) | 0.000268 |
| 0.001 | 0.000262 |
| **0.01** | **0.000257** ✅ |
| 0.1 | 0.000321 |
| 1.0 | 0.000649 |

Seeing this U-shape appear from my own numbers, rather than just reading about it, made the bias-variance tradeoff click in a way it hadn't from theory alone.

### Final Model Comparison

| Model | Train MSE | Test MSE |
|---|---|---|
| Custom Gradient Descent (λ=0.01) | 0.000359 | **0.000257** |
| Sklearn LinearRegression (no reg.) | 0.000315 | 0.000268 |
| Sklearn Ridge (α equivalent to λ=0.01) | — | 0.000257 |

### Weight Convergence — Custom GD vs Sklearn Ridge (after fixing the scaling comparison)

| | vol_lag_1 | vol_lag_2 | vol_lag_3 | vol_lag_4 | vol_lag_5 |
|---|---|---|---|---|---|
| Custom Ridge GD | 0.10416383 | 0.03458131 | 0.00473984 | -0.00875307 | -0.02173006 |
| Sklearn Ridge | 0.10416385 | 0.03458128 | 0.00473984 | -0.00875306 | -0.02173007 |

Matching to five decimal places gave me real confidence that the gradient and update rule I wrote by hand are actually correct, not just producing plausible-looking output.

## 🔮 Live Prediction

Once trained, the model can forecast the next trading day's volatility from the most recent available data:

```bash
python -m main            # trains, tunes lambda, saves outputs/trained_model.npz
python -m src.predict     # loads the saved model and forecasts the next trading day
```

Example output:
```
Most recent trading data as of: 2026-07-02
Predicted next-day (annualized) volatility: 0.17405
```

Worth noting: this is only ever a next-*trading*-day forecast, since the underlying data skips weekends/holidays — "tomorrow" on a Friday means the following Monday.

## 🛠️ Tech Stack

Python 3.9 · NumPy · Pandas · Matplotlib · yfinance · scikit-learn (used only for validation, never for training)

## ▶️ How to Run

```bash
git clone https://github.com/uctiot007/stock-volatility-predictor.git
cd stock-volatility-predictor

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

python -m main
python -m src.predict
```

## 🔮 Things I'd Like to Try Next

- **Lasso (L1) regression** — compare against Ridge's shrinkage behavior
- **Volume as a feature** — currently only past volatility is used, volume is fetched but unused
- **Walk-forward cross-validation** — a single chronological split is a reasonable start, but rolling-window validation would be more rigorous
- **A GARCH baseline** — the standard econometric approach to volatility forecasting, worth comparing against directly
- **Non-linear models** — to check whether volatility clustering has structure a linear model can't capture

## 👤 Author

Akshat — learning quantitative finance and ML by building things and checking my understanding against known-correct implementations.

If you spot something I got wrong or could do better, I'd genuinely like to know — feel free to open an issue.
If you like this project, give it a ⭐ on GitHub — it helps visibility!
