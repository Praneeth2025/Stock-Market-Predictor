<img width="1213" height="437" alt="image" src="https://github.com/user-attachments/assets/c0f095fb-84cd-4f8a-86f7-d0fdd2107ee5" />


# Stock Market Predictor (Buy / Sell / Hold)

Predict short‑term stock movement as **Buy / Sell / Hold** by combining **technical analysis (Supertrend, RSI, ATR)** with **machine learning** and engineered features.

> **Impact**: The ML system **significantly outperforms a traditional trading strategy baseline (30.6%)**, with balanced performance across all three classes. Evaluation emphasizes **macro F1**, **precision/recall**, and **confusion matrices** rather than a single accuracy number.

---

## 🔎 Project Overview

* **Goal**: Classify each time step into **Buy / Sell / Hold** to support rule‑based or discretionary trading.
* **Data**: Time‑series OHLCV with derived indicators (Supertrend, RSI, ATR, moving averages, returns, volatility measures), plus selected engineered features.
* **Learning Setup**: Supervised 3‑class classification with careful **time‑aware splits** (to prevent leakage), cross‑validation, and ablation of features/indicators.
* **Why this matters**: Demonstrates a practical workflow for **feature engineering on market data** + **robust model evaluation** that is more realistic than naïve accuracy on shuffled splits.

---

## 🗂️ Repository Structure

> The repo centers around the `project_root/` workspace and notebooks. If you add or rename files, update the table below.

```
project_root/
├─ data/
│  ├─ raw/                # Unmodified input data (CSV/Parquet). Do not commit private market data.
│  ├─ interim/            # Intermediate artifacts created during cleaning/feature generation.
│  └─ processed/          # Final training tables with labels & features.
│
├─ notebooks/
│  ├─ 01_data_ingestion.ipynb          # Load OHLCV, align time zones, handle missing values/outliers
│  ├─ 02_feature_engineering.ipynb     # Supertrend/RSI/ATR, rolling stats, returns, volatility, etc.
│  ├─ 03_labeling_and_splits.ipynb     # Create Buy/Sell/Hold labels; time‑series CV; leak checks
│  ├─ 04_modeling_and_tuning.ipynb     # Train/tune baselines + ensembles; class weights
│  └─ 05_evaluation_backtest.ipynb     # Confusion matrix, F1/precision/recall, simple backtests
│
├─ src/
│  ├─ features/
│  │  ├─ indicators.py     # Supertrend/RSI/ATR helpers; moving averages; returns; volatility
│  │  └─ engineer.py       # Feature pipelines, scaling, target lags/leads
│  ├─ models/
│  │  ├─ train.py          # Fit/serialize models; Optuna/Bayes tuning hooks
│  │  └─ evaluate.py       # Metrics, confusion matrices, class‑wise reports
│  ├─ utils/
│  │  ├─ io.py             # I/O helpers; reproducible paths under project_root
│  │  └─ time_splits.py    # Time‑series splitters, purge/embargo for leakage control
│  └─ __init__.py
│
├─ models/                 # Saved model artifacts (pickle/joblib)
├─ reports/
│  └─ figures/             # Plots: feature importance, confusion matrix, CV diagnostics
├─ configs/
│  └─ config.yaml          # Symbols, horizons, thresholds, features to enable/disable
├─ requirements.txt        # Python deps (or use the list below)
└─ create_structure.py     # Helper to scaffold the folder tree
```

> **Note**: The repo also includes a top‑level `README.md` (this file) and the helper script `create_structure.py` that scaffolds the above directories.

---

## 🧰 Features & Methods

* **Technical indicators**: Supertrend, RSI, ATR (+ MAs/EMAs, rolling volatility, returns).
* **Leakage control**: Time‑aware train/val/test splits; optional purge/embargo around split boundaries.
* **Modeling**: Strong tabular baselines (Logistic/RF/XGBoost/LightGBM) and optional neural baselines.
* **Imbalance handling**: Class weights / focal‑style loss alternatives / thresholding on decision scores.
* **Evaluation**: Macro F1, per‑class precision/recall, confusion matrices, ROC‑AUC (macro/OVR), PR‑AUC.
* **Interpretability**: Permutation importance / SHAP (optional) to understand drivers of Buy vs Sell vs Hold.

---

## ⚙️ Setup

### 1) Python

* Use Python **3.9+**.

### 2) Install dependencies

If `requirements.txt` is missing, the following minimal set works for notebooks:

```bash
pip install -U \
  pandas numpy scikit-learn xgboost lightgbm \
  pandas-ta yfinance ta \
  matplotlib seaborn ipykernel joblib pyyaml
```

> `pandas‑ta` or `ta` can compute RSI/ATR; `pandas‑ta` includes **Supertrend**. If unavailable, implement Supertrend in `src/features/indicators.py`.

### 3) Project root

Run from the repository root to ensure relative paths resolve under `project_root/`.

---

## 🚀 How to Run

### Notebook workflow (recommended)

1. **01\_data\_ingestion.ipynb** – Place input CSVs under `project_root/data/raw/` and load OHLCV for desired symbols.
2. **02\_feature\_engineering.ipynb** – Build indicators (Supertrend/RSI/ATR) + engineered features (returns, volatility, rolling stats).
3. **03\_labeling\_and\_splits.ipynb** – Define the Buy/Sell/Hold target (e.g., future return window and thresholds), create time‑series CV folds.
4. **04\_modeling\_and\_tuning.ipynb** – Train/tune models; log metrics; save best model to `models/`.
5. **05\_evaluation\_backtest.ipynb** – Inspect confusion matrix & per‑class metrics; run a simple, constraints‑aware backtest to contextualize predictions.

### CLI (optional, if `src/` scripts are used)

```bash
python -m src.models.train --config configs/config.yaml
python -m src.models.evaluate --config configs/config.yaml --model models/best_model.joblib
```

---

## 🧪 Labels & Baseline

* **Target**: 3 classes — **Buy / Sell / Hold** — defined by forward‑looking return over a fixed horizon with symmetric thresholds (configurable in `configs/config.yaml`).
* **Baselines**:

  * **Traditional strategy** (reference): 30.6% baseline effectiveness.
  * **ML baselines**: majority‑class, logistic regression, and tuned tree ensembles.
* **What we report**: Macro F1, class‑wise precision/recall, and qualitative confusion‑matrix analysis (not just accuracy).

---

## 📈 Evaluation & Diagnostics

* **Cross‑validation**: Rolling or expanding‑window CV; report mean±std for each fold.
* **Confusion matrix**: Ensure no single class (Buy/Sell/Hold) is neglected; quantify trade‑off between precision vs recall for Buy/Sell.
* **Calibration**: Optionally apply probability calibration (Platt/Isotonic) and threshold optimization per class.
* **Error analysis**: Slice by volatility regime, session (open/close), and symbol to detect brittleness.

---

## 🔬 Reproducibility

* **Determinism**: Set global seeds; log library versions.
* **Data versioning**: Keep raw data immutable; regenerate processed data from notebooks/scripts.
* **Configs**: Centralize hyperparameters and feature toggles in `configs/config.yaml`.

---

## 🛑 Limitations & Notes

* **Past performance ≠ future results**; this repo is **educational** and **not financial advice**.
* **Transaction costs / slippage** are not included unless explicitly modeled in the backtest.
* **Data quality** (splits, survivorship bias, look‑ahead leaks) critically affects results; see notebooks for checks.

---

## 🗺️ Roadmap

* [ ] Add symbol universe & horizon presets in `configs/`
* [ ] Add LightGBM + CatBoost baselines
* [ ] Export shapley plots and feature importance to `reports/figures/`
* [ ] Add simple live inference script for daily signals
* [ ] Expand backtesting to include transaction costs & position sizing

---

## 📦 Example Artifacts

* `models/best_model.joblib` – Serialized classifier
* `reports/figures/` – Feature importance, ROC, PR curves, confusion matrices
* `project_root/data/processed/train.parquet` – Final training table with features + label (schema documented in notebooks)

---

## 📝 How to Cite / Describe on a Resume

> *“Designed and evaluated a 3‑class stock movement classifier (Buy/Sell/Hold) on 11k+ records. Combined Supertrend/RSI/ATR with engineered features and time‑series CV. The ML system **significantly outperformed a 30.6% traditional strategy baseline**, with balanced macro‑F1 across classes and rigorous leakage controls.”*

---

## 🔐 License & Disclaimer

* Add a `LICENSE` file (e.g., MIT) if you want others to reuse the code.
* This repository is for **research/education only**, **not** investment advice.

---

## 🙏 Acknowledgements

* Technical indicators via `pandas‑ta` / `ta` (or custom implementations).
* Community discussions and public educational resources on time‑series ML and leakage‑aware validation.
