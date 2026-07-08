# 📈 Stock Volatility Predictor
### Testing Five Model Families From Scratch — and Finding That None of Them Beat Guessing "Tomorrow Looks Like Today"

## 🚀 Overview

This project started as an attempt to take the linear regression and gradient descent math I'd learned in theory and actually implement it myself, rather than just calling `sklearn.linear_model.LinearRegression()`. The goal was never to build the most accurate volatility model possible — it was to make sure I genuinely understood what's happening under the hood: the gradients, the update rule, what regularization actually does to the loss surface, and — it turned out — how easy it is to fool yourself with an evaluation that isn't rigorous enough.

It grew from there. Once I had a working linear model, I kept asking "but is this actually good?" — which led to adding a naive baseline, then walk-forward validation, then testing whether a more standard time-series tool (ARIMA) or a model purpose-built for volatility (GARCH, and later GJR-GARCH) could do better. The most useful part of this project ended up being a finding I didn't expect going in: across every model I tried, simple to sophisticated, none of them reliably beat the simplest possible baseline — and the more sophisticated the model got, the worse it tended to do. I think that result, and the process of uncovering it properly, is more interesting than a clean accuracy number would have been, so this README leads with it rather than hiding it.

## 🎯 Problem Statement

Predict next-day realized volatility of the S&P 500 using its own recent history. This relies on a known property of markets called volatility clustering — calm and turbulent periods tend to persist for a while — so recent volatility is at least somewhat informative about tomorrow's.

- **Inputs:** the previous 5 days of rolling 21-day realized volatility, plus (in one experiment) lagged volume change and a rolling volume average, plus (for the time-series models) the raw daily return series itself
- **Target:** next day's realized volatility
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

**Bias-Variance Tradeoff** and **walk-forward validation** — I wanted to actually see these rather than recite the definitions, so I swept regularization strength and used rolling-origin train/test splits instead of trusting a single split.

**ARIMA and GARCH-family models** — used via `statsmodels` and `arch` respectively, since implementing their maximum-likelihood estimation from scratch is a much larger undertaking than the linear optimizer above and wasn't really the point of this stage. These were brought in specifically to test whether standard, purpose-built time-series tools could do what my from-scratch linear model couldn't.

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
│   ├── walk_forward.py       # Rolling-origin walk-forward validation for the linear model
│   ├── arima_baseline.py     # ARIMA, multi-step and rolling one-step-ahead
│   ├── garch_baseline.py     # GARCH / GJR-GARCH, rolling one-step-ahead
│   └── predict.py            # Load a trained model, forecast the next trading day
│
├── main.py                   # End-to-end pipeline: data → train → tune → validate every model → save
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
    → ARIMA (multi-step, then rolling one-step) → GARCH → GJR-GARCH
    → Sklearn Comparison → Save Model → Predict
```

---

## 🔬 What I Actually Found (Not Just the Clean Version)

### 1. My weights didn't match sklearn's at first, and the reason was multicollinearity

My first gradient descent run produced predictions close to sklearn's `LinearRegression`, but the learned weights differed meaningfully — expected once I understood why: the 5 lagged volatility features are highly correlated with each other, so the loss surface isn't a clean bowl with one minimum, it's a flat, elongated valley where many different weight combinations give almost identical loss. This is the textbook reason to reach for regularization, which is what led me to implement Ridge next.

### 2. My "same lambda" comparison against sklearn's Ridge was initially invalid

My loss averages squared error over `n` samples; sklearn's Ridge sums it. With ~1,600 training rows, the same nominal λ was roughly 1,600x stronger in my version. Scaling my λ by `n_train` before comparing fixed this — afterward, my weights matched sklearn's Ridge to five decimal places. Useful reminder: a same-named hyperparameter across two implementations doesn't guarantee the same behavior unless you actually check the math.

### 3. A single train/test split was hiding a much less flattering picture

Once I added a **naive baseline** (simply predicting "tomorrow's volatility = today's volatility," no model at all) and **walk-forward validation** (5 rolling train/test folds, never training on future data), the picture changed. On a single 80/20 split, my model looked roughly competitive with the naive baseline. Under walk-forward validation — a more honest test, since it evaluates the model across several different historical periods rather than one lucky/unlucky slice — the model's average error was clearly worse than just guessing "no change." I hadn't computed the naive baseline until fairly late in the project, and once I did, it was obvious my earlier single-split result had been somewhat flattering.

### 4. Adding volume didn't help, and Lasso independently confirmed why

My first instinct was that the model was missing information, so I added lagged trading volume change and a rolling volume average as extra features. This barely changed anything. To check more rigorously, I ran Lasso, which — unlike Ridge — can drive individual weights all the way to exactly zero when a feature isn't pulling its weight:

```
Lasso weights: [0.1338, 0.0001, -0.0001, -0.0001, -0.0202, 0.0002, 0.0001]
                vol_lag_1  vol_lag_2  vol_lag_3  vol_lag_4  vol_lag_5  volume_change  volume_ma
```

Lasso effectively zeroed out `vol_lag_2` through `vol_lag_4` and both volume features, keeping real weight on only `vol_lag_1` and `vol_lag_5`. Its test MSE landed at 0.000248 — identical to the naive baseline to six decimal places. Given the freedom to pick whichever features actually mattered, an independent model landed on almost exactly "just use yesterday's volatility," which is what the naive baseline already does.

### 5. A single multi-step ARIMA forecast was a badly unfair comparison to my daily-updating linear model

My first ARIMA attempt fit once per fold and forecast the entire ~330-day test window in one shot. ARIMA's forecasts converge toward the series' unconditional mean as the horizon grows, since the autoregressive term's influence decays with distance — so by 20-30 days into a 330-day forecast, it was essentially predicting a flat line regardless of what volatility actually did. That produced a badly inflated error (MSE 0.0159) that wasn't really testing ARIMA's forecasting ability, just its ability to guess a long-run average. Switching to rolling one-step-ahead forecasting (refitting periodically and always forecasting only the next day) brought this down to 0.000776 — still worse than naive, but a fair comparison instead of a broken one. This was a good reminder that the evaluation protocol matters as much as the model itself.

### 6. More model sophistication made things worse, not better

Once I had ARIMA evaluated fairly, I tested GARCH(1,1) — the model actually designed for volatility, unlike ARIMA which is a general time-series tool applied somewhat awkwardly here. I expected GARCH to finally beat naive. It didn't. I then tried refitting daily instead of weekly (worse), switching to Student's t errors for fatter tails (roughly the same), and GJR-GARCH to capture the well-documented asymmetric response of volatility to negative vs. positive returns (also roughly the same, still worse than naive). Every variant I tried was worse than the simple linear model, which was itself worse than naive.

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

Lasso degrades much more sharply than Ridge past λ=0.01 — L1's harsher, non-smooth penalty pushes weights to zero much more aggressively, so it's far less forgiving of an overly large λ.

### Sklearn Validation (Ridge, λ=0.01)

| | vol_lag_1 | vol_lag_2 | vol_lag_3 | vol_lag_4 | vol_lag_5 | volume_change | volume_ma |
|---|---|---|---|---|---|---|---|
| Custom Ridge GD | 0.10315 | 0.03505 | 0.00460 | -0.00905 | -0.02241 | 0.00113 | 0.00215 |
| Sklearn Ridge (matched α) | 0.10315 | 0.03505 | 0.00460 | -0.00905 | -0.02241 | 0.00113 | 0.00215 |

Matching to five decimal places confirms the gradient and update rule are implemented correctly.

### The Complete, Honest Final Comparison

| Rank | Method | Test MSE |
|---|---|---|
| 1 | Naive baseline (today = tomorrow) | 0.000248 |
| 1 (tie) | Custom Lasso (single split) | 0.000248 |
| 3 | Custom Ridge (single split) | 0.000255 |
| 4 | Custom Ridge (walk-forward avg, 5 folds) | 0.000398 |
| 5 | ARIMA (rolling one-step, refit weekly) | 0.000776 |
| 6 | GARCH(1,1), normal errors, refit weekly | 0.003166 |
| 7 | GJR-GARCH(1,1,1), Student's t, refit weekly | 0.005714 |
| 8 | GARCH(1,1), normal errors, refit daily | 0.005895 |
| 9 | ARIMA (single multi-step forecast per fold — unfair comparison, included for completeness) | 0.015889 |

**Does any model beat the naive baseline? No.** And the ranking isn't noise — it's monotonic with model complexity. Every step from naive toward a more sophisticated model made the walk-forward error worse, not better.

---

## 🤔 What I Take Away From This

Beating a naive persistence forecast is a genuinely known-hard problem in volatility forecasting, and I ran into that directly rather than just reading about it. Across five model families — a hand-implemented linear model with L2/L1 regularization, ARIMA, GARCH, and GJR-GARCH with fat-tailed errors — none reliably beat "assume tomorrow looks like today" under walk-forward validation on this dataset. If anything, additional model complexity made results worse: more parameters to estimate on a relatively short, noisy daily series seems to have added estimation error faster than it captured real signal.

I don't take this as "the models are wrong" so much as "daily volatility, using only its own past values, may simply not carry enough exploitable signal to beat a trivial baseline over this period" — which is itself a real, if less flattering, finding. A few things I'd want to check before drawing a stronger conclusion: whether a longer/shorter lag window changes anything, whether the naive baseline holds up as well on other tickers or asset classes, and whether adding genuinely external information (implied volatility, macro data, news) — rather than more transformations of the same past-price data — is what's actually needed to move past this ceiling.

## 🔮 Live Prediction

```bash
python -m main            # trains every model, validates, saves outputs/trained_model.npz
python -m src.predict     # loads the saved linear model and forecasts the next trading day
```

Example output:
```
Most recent trading data as of: 2026-07-02
Predicted next-day (annualized) volatility: 0.17405
```

Worth noting: this is only ever a next-*trading*-day forecast, since the data skips weekends/holidays. Given the findings above, this forecast should be read with real skepticism rather than confidence — the underlying model has not been shown to beat simply using the most recent volatility reading directly.

## 🛠️ Tech Stack

Python 3.9 · NumPy · Pandas · Matplotlib · yfinance · scikit-learn (validation only) · statsmodels (ARIMA) · arch (GARCH / GJR-GARCH)

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

Note: the full `main.py` run trains and validates every model above, including several ARIMA and GARCH refits — expect it to take a few minutes, not seconds.

## 🔮 Things I'd Like to Try Next

- **Longer/shorter lag windows** — check whether 5 days is the right amount of history, or whether that choice itself is limiting every model tested
- **A genuinely external feature** — implied volatility (e.g. VIX), macro indicators, or news-based signals, rather than more transformations of the same past price/volume data
- **Testing on other tickers or asset classes** — to see whether "nothing beats naive" is specific to this period/index or a broader pattern
- **Non-linear models** — check whether volatility clustering has structure a linear model fundamentally can't capture, even if the linear and GARCH-family results so far are discouraging
- **A proper walk-forward hyperparameter selection** — right now λ is chosen on the single split, then evaluated separately via walk-forward; ideally λ itself would be selected using walk-forward validation too

## 👤 Author

Akshat — learning quantitative finance and ML by building things, checking my understanding against known-correct implementations, and trying not to let a flattering number stop me from evaluating it properly.

If you spot something I got wrong or could do better, I'd genuinely like to know — feel free to open an issue.