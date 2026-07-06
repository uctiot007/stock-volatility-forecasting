# 📈 Stock Volatility Predictor
### Implementing Gradient Descent, Ridge, and Lasso from Scratch — and Learning When a Model Doesn't Beat the Obvious Baseline

## 🚀 Overview

This project started as an attempt to take the linear regression and gradient descent math I'd learned in theory and actually implement it myself, rather than just calling `sklearn.linear_model.LinearRegression()`. The goal was never to build the most accurate volatility model possible — it was to make sure I genuinely understood what's happening under the hood: the gradients, the update rule, what regularization actually does to the loss surface, and — it turned out — how easy it is to fool yourself with an evaluation that isn't rigorous enough.

The most useful part of this project ended up being a finding I didn't expect going in: once evaluated properly, my model does not reliably beat the simplest possible baseline. I think that result, and the process of uncovering it, is more interesting than a clean accuracy number would have been, so this README leads with it rather than hiding it.

## 🎯 Problem Statement

Predict next-day realized volatility of the S&P 500 using its own recent volatility history (and, as an experiment, trading volume). This relies on a known property of markets called volatility clustering — calm and turbulent periods tend to persist for a while — so recent volatility is at least somewhat informative about tomorrow's.

- **Inputs (X):** the previous 5 days of rolling 21-day realized volatility, plus (in one experiment) lagged volume change and a rolling volume average
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

**Ridge (L2) and Lasso (L1) Regularization**
```
Ridge:  J(w) = MSE + λΣw²        →  dw += 2λw
Lasso:  J(w) = MSE + λΣ|w|       →  dw += λ·sign(w)
```
Both implemented from scratch, with the bias term deliberately excluded from the penalty in each case.

**Bias-Variance Tradeoff** and **walk-forward validation** — I wanted to actually see these rather than recite the definitions, so I swept regularization strength and used rolling-origin train/test splits instead of trusting a single split (see below for why that mattered).

## 🏗️ Project Structure

```
stock-volatility-predictor/
│
├── data/
│   └── raw/                  # Downloaded market data (gitignored)
│
├── src/
│   ├── data_loader.py        # Fetch, clean, cache S&P 500 data
│   ├── features.py           # Log returns, rolling volatility, lag + volume features
│   ├── model.py               # Gradient descent, Ridge (L2), Lasso (L1) — from scratch
│   ├── evaluate.py           # Metrics, naive baseline, sklearn comparison, plots
│   ├── walk_forward.py       # Rolling-origin walk-forward validation
│   └── predict.py            # Load a trained model, forecast the next trading day
│
├── main.py                   # End-to-end pipeline: data → train → tune → validate → save
├── outputs/
│   └── trained_model.npz     # Saved weights/bias/normalization stats
├── requirements.txt
└── README.md
```

## ⚙️ Pipeline

```
Raw Prices → Log Returns → Rolling Volatility (+ Volume Features)
    → Chronological Train/Test Split → Feature Standardization (train stats only)
    → Ridge & Lasso λ Sweeps → Naive Baseline → Walk-Forward Validation
    → Sklearn Comparison → Save Model → Predict
```

---

## 🔬 What I Actually Found (Not Just the Clean Version)

### 1. My weights didn't match sklearn's at first, and the reason was multicollinearity

My first gradient descent run produced predictions close to sklearn's `LinearRegression`, but the learned weights differed meaningfully — expected once I understood why: the 5 lagged volatility features are highly correlated with each other, so the loss surface isn't a clean bowl with one minimum, it's a flat, elongated valley where many different weight combinations give almost identical loss. This is the textbook reason to reach for regularization, which is what led me to implement Ridge next.

### 2. My "same lambda" comparison against sklearn's Ridge was initially invalid

My loss averages squared error over `n` samples; sklearn's Ridge sums it. With ~1,600 training rows, the same nominal λ was roughly 1,600x stronger in my version. Scaling my λ by `n_train` before comparing fixed this — afterward, my weights matched sklearn's Ridge to five decimal places. Useful reminder: a same-named hyperparameter across two implementations doesn't guarantee the same behavior unless you actually check the math.

### 3. A single train/test split was hiding a much less flattering picture

Once I added a **naive baseline** (simply predicting "tomorrow's volatility = today's volatility," no model at all) and **walk-forward validation** (5 rolling train/test folds, never training on future data), the picture changed:

| Method | Test MSE |
|---|---|
| Naive baseline (today = tomorrow) | 0.000248 |
| Custom Ridge GD (single 80/20 split) | 0.000255–0.000257 |
| Custom Ridge GD (walk-forward average, 5 folds) | 0.000385–0.000398 |

On the single split, my model looked roughly competitive with the naive baseline. Under walk-forward validation — a more honest test, since it evaluates the model across several different historical periods rather than one lucky/unlucky slice — the model's average error was clearly *worse* than just guessing "no change." I hadn't computed the naive baseline until fairly late in the project, and once I did, it was obvious my earlier single-split result had been somewhat flattering.

### 4. Adding volume didn't help, and Lasso independently confirmed why

My first instinct was that the model was missing information, so I added lagged trading volume change and a rolling volume average as extra features. This barely changed anything (single-split MSE moved from 0.000257 to 0.000255; walk-forward average got slightly *worse*, 0.000385 → 0.000398).

To check this more rigorously, I ran Lasso, which — unlike Ridge — can drive individual weights all the way to exactly zero when a feature isn't pulling its weight. Lasso's result was clean and unambiguous:

```
Lasso weights: [0.1338, 0.0001, -0.0001, -0.0001, -0.0202, 0.0002, 0.0001]
                vol_lag_1  vol_lag_2  vol_lag_3  vol_lag_4  vol_lag_5  volume_change  volume_ma
```

Lasso effectively zeroed out `vol_lag_2` through `vol_lag_4` and both volume features, keeping real weight on only `vol_lag_1` and `vol_lag_5`. Its test MSE landed at **0.000248 — identical to the naive baseline to six decimal places.** Given the freedom to pick whichever features actually mattered, an independent model landed on almost exactly "just use yesterday's volatility," which is what the naive baseline already does. That's a strong, independently-derived confirmation, not just a coincidence from one evaluation method.

---

## 📊 Full Results

### Ridge Lambda Sweep (5,000 epochs, single 80/20 split)

| λ | Test MSE |
|---|---|
| 0.0 | 0.000266 |
| 0.001 | 0.000260 |
| **0.01** | **0.000255** ✅ |
| 0.1 | 0.000317 |
| 1.0 | 0.000599 |

### Lasso Lambda Sweep

| λ | Test MSE |
|---|---|
| 0.0 | 0.000266 |
| **0.001** | **0.000248** ✅ (matches naive baseline) |
| 0.01 | 0.000249 |
| 0.1 | 0.002009 |
| 1.0 | 0.260348 |

Note how sharply Lasso degrades past λ=0.01 compared to Ridge — L1's harsher, non-smooth penalty pushes weights to zero much more aggressively, so it's far less forgiving of an overly large λ.

### Sklearn Validation (Ridge, λ=0.01)

| | vol_lag_1 | vol_lag_2 | vol_lag_3 | vol_lag_4 | vol_lag_5 | volume_change | volume_ma |
|---|---|---|---|---|---|---|---|
| Custom Ridge GD | 0.10315 | 0.03505 | 0.00460 | -0.00905 | -0.02241 | 0.00113 | 0.00215 |
| Sklearn Ridge (matched α) | 0.10315 | 0.03505 | 0.00460 | -0.00905 | -0.02241 | 0.00113 | 0.00215 |

Matching to five decimal places confirms the gradient and update rule are implemented correctly.

### The Honest Final Comparison

| Method | Test MSE |
|---|---|
| Naive baseline | 0.000248 |
| Custom Ridge (single split) | 0.000255 |
| Custom Lasso (single split) | 0.000248 |
| Custom Ridge (walk-forward avg) | 0.000398 |

**Does the model beat the naive baseline? No — not reliably.** Under the more honest walk-forward evaluation, it's clearly worse; even in the best case (Lasso), it only ties the baseline, not beats it.

---

## 🤔 What I Take Away From This

A 5–7 feature linear model on lagged daily volatility does not add meaningful predictive value over simply assuming tomorrow looks like today, at least on this dataset and feature set. This isn't actually a surprising result in finance — it's well known that beating a naive/persistence-style forecast is genuinely hard for volatility, and it's part of why more structured models (GARCH-family models, which explicitly model volatility as mean-reverting with shocks, rather than a plain linear autoregression) are the standard tool for this problem. That's the natural next thing I want to test — comparing this linear approach against an actual GARCH baseline, to see whether the ceiling is "linear models in general" or something specific to how I've set this one up.

## 🔮 Live Prediction

```bash
python -m main            # trains, sweeps Ridge/Lasso, validates, saves outputs/trained_model.npz
python -m src.predict     # loads the saved model and forecasts the next trading day
```

Example output:
```
Most recent trading data as of: 2026-07-02
Predicted next-day (annualized) volatility: 0.17405
```

Worth noting: this is only ever a next-*trading*-day forecast, since the data skips weekends/holidays — "tomorrow" on a Friday means the following Monday. Given the findings above, this forecast should be read with real skepticism rather than confidence.

## 🛠️ Tech Stack

Python 3.9 · NumPy · Pandas · Matplotlib · yfinance · scikit-learn (validation only, never used for training)

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

- **A GARCH baseline** — the standard econometric approach to volatility forecasting; the natural next test given the linear model's ceiling
- **Longer/shorter lag windows** — check whether 5 days is the right amount of history, or whether that choice itself is limiting
- **Non-linear models** — check whether volatility clustering has structure a linear model fundamentally can't capture
- **A proper walk-forward hyperparameter selection** — right now λ is chosen on the single split, then evaluated separately via walk-forward; ideally λ itself would be selected using walk-forward validation too

## 👤 Author

Akshat — learning quantitative finance and ML by building things, checking my understanding against known-correct implementations, and trying not to let a flattering number stop me from evaluating it properly.

If you spot something I got wrong or could do better, I'd genuinely like to know — feel free to open an issue.
If this project is useful or interesting, a ⭐ on GitHub is appreciated.
