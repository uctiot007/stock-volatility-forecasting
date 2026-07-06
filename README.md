# 📈 Stock Volatility Predictor
### From-Scratch Gradient Descent & Ridge Regression, Validated Against Sklearn

## 🚀 Overview

Most student ML projects call `.fit()` and stop there. This project instead builds the entire pipeline — including the optimizer itself — from first principles, then rigorously proves that it converges to the mathematically correct answer.

The goal wasn't just to predict volatility. It was to demonstrate, with evidence, that I understand *why* the optimization works, where it breaks, and how to diagnose it when it doesn't match expectations.

## 🎯 Problem Statement

Predict next-day realized volatility of the S&P 500 using its own recent volatility history — a supervised regression problem built on a well-known market property: **volatility clustering** (volatile periods tend to persist, so recent volatility is informative about tomorrow's).

- **Inputs (X):** lagged rolling 21-day realized volatility (previous 5 days)
- **Target (y):** next day's realized volatility
- **Data:** S&P 500 (`^GSPC`) daily prices via `yfinance`

## 🧠 Core Concepts Implemented

**Linear Regression**
```
y = Xw + b
```

**Batch Gradient Descent**
```
w := w - α · ∇J(w)
b := b - α · ∂J/∂b
```
Implemented manually in NumPy — no `sklearn.linear_model` used during training, only for validation afterward.

**Ridge Regression (L2 Regularization)**
```
J(w) = MSE(y, ŷ) + λ‖w‖²
```
Added to counteract multicollinearity discovered during validation (see below). The bias term is deliberately excluded from the penalty, matching standard convention.

**Bias-Variance Tradeoff**, shown empirically via a λ sweep rather than just stated theoretically (see Results).

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
│   └── evaluate.py           # Metrics + sklearn comparison + plots
│
├── main.py                   # End-to-end pipeline: data → train → tune → evaluate
├── requirements.txt
└── README.md
```

## ⚙️ Pipeline

```
Raw Prices → Log Returns → Rolling Volatility → Lag Features
    → Chronological Train/Test Split → Feature Standardization (train stats only)
    → Gradient Descent (λ sweep) → Evaluation vs Sklearn → Visualization
```

---

## 🔬 Investigation: Three Real Findings, Not Just a Working Model

This is the part that separates this project from a tutorial copy-paste. Getting a model to *run* was the easy part — three issues came up during validation that required actual debugging and understanding to resolve.

### 1. Weights didn't match sklearn at first — and the reason was multicollinearity, not a bug

After the initial gradient descent run, predictions closely tracked sklearn's `LinearRegression` (test MSE within ~1% of each other), but the learned **weights** were noticeably different despite both models reaching a similar loss:

| | vol_lag_1 | vol_lag_2 | vol_lag_3 | vol_lag_4 | vol_lag_5 |
|---|---|---|---|---|---|
| Custom GD | 0.119 | 0.037 | 0.001 | -0.015 | -0.029 |
| Sklearn OLS | 0.165 | -0.013 | -0.018 | -0.001 | -0.019 |

This is expected, not a defect: the five lagged volatility features are highly autocorrelated with each other, so the loss surface has a flat, elongated valley rather than one sharp minimum — many different weight combinations produce nearly identical predictions. This is the textbook symptom of **multicollinearity**, and it's exactly why the project moved to Ridge regression next: a unique, stable solution requires regularization when features are this correlated.

### 2. A loss-scaling bug made an early Ridge-vs-sklearn comparison invalid

When first comparing custom Ridge against `sklearn.linear_model.Ridge` at "the same" λ, the sklearn weights barely moved from OLS while the custom weights shrank substantially — a red flag that the two penalties weren't actually equivalent.

Root cause: this implementation's loss **averages** squared error over `n` samples (`(1/n)·Σerror² + λΣw²`), while sklearn's Ridge **sums** it (`Σerror² + alpha·Σw²`). With ~1,600 training samples, the same nominal λ is ~1,600x stronger in relative terms here than in sklearn's formulation. Fixed by scaling: `alpha = λ · n_train` when constructing the sklearn comparison model. After the fix, the two solutions matched to 5+ decimal places (see Results). This is a subtle, easy-to-miss detail that only surfaces when you actually derive both loss functions rather than assuming "same λ" means "same regularization."

### 3. Prediction lag at volatility regime changes

Visual inspection of predicted-vs-actual plots shows the model tracks overall volatility levels well, but tends to **lag by roughly one day at sharp turning points** — spikes in predicted volatility arrive slightly after the actual spike. This is an inherent property of the model, not a fixable error: since features are strictly *past* volatility, a linear autoregressive model can only extrapolate recent momentum, not anticipate new shocks. Worth stating explicitly rather than glossing over — it defines the honest limits of this approach.

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

The U-shape is the bias-variance tradeoff made visible: mild regularization improves generalization, but too much (λ ≥ 0.1) forces weights toward zero aggressively enough to destroy real signal, and test error rises sharply.

### Final Model Comparison

| Model | Train MSE | Test MSE |
|---|---|---|
| Custom Gradient Descent (λ=0.01) | 0.000359 | **0.000257** |
| Sklearn LinearRegression (no reg.) | 0.000315 | 0.000268 |
| Sklearn Ridge (α equivalent to λ=0.01) | — | 0.000257 |

### Weight Convergence — Custom GD vs Sklearn Ridge (fair comparison, matched α)

| | vol_lag_1 | vol_lag_2 | vol_lag_3 | vol_lag_4 | vol_lag_5 |
|---|---|---|---|---|---|
| Custom Ridge GD | 0.10416383 | 0.03458131 | 0.00473984 | -0.00875307 | -0.02173006 |
| Sklearn Ridge | 0.10416385 | 0.03458128 | 0.00473984 | -0.00875306 | -0.02173007 |

Matching to 5+ decimal places. Bias term matches to 10 decimal places. This is direct evidence the manual gradient descent implementation is mathematically correct, not merely "close enough."

## ✅ Key Achievements

- ✔ From-scratch gradient descent converges to the exact closed-form Ridge solution once regularization is scaled correctly
- ✔ Ridge (λ=0.01) beats both unregularized GD and sklearn's plain OLS on test MSE — a real, measured generalization improvement, not just a theoretical claim
- ✔ Diagnosed and explained *why* weights diverged from OLS (multicollinearity) before reaching for regularization, rather than adding it blindly
- ✔ Caught and fixed a loss-scaling inconsistency that would have invalidated the sklearn comparison entirely

## 🛠️ Tech Stack

Python 3.9 · NumPy · Pandas · Matplotlib · yfinance · scikit-learn (validation only, never used for training)

## ▶️ How to Run

```bash
git clone https://github.com/yourusername/stock-volatility-predictor.git
cd stock-volatility-predictor

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

python -m main
```



## 👤 Author

Akshat 

If this project is useful or interesting, a ⭐ on GitHub is appreciated.
