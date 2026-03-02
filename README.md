# house-price-prediction


# Ames Housing Price Prediction

A machine learning pipeline for predicting house sale prices using the Ames Housing dataset. The project implements a full ML workflow, from EDA and feature engineering through hyperparameter tuning and stacking ensemble, using proper software engineering practices with a modular `src/` layout.

---

## Results

| Model | CV RMSE (log-scale) |
|---|---|
| Ridge (tuned) | 0.11385660769451969 |
| LightGBM (tuned) | 0.11697679343758974 |
| XGBoost (tuned) | 0.11285154116922876— |
| Random Forest (tuned) | 0.12980564062409744 |
| **Stacking Ensemble** | 0.11167192843807346 |

> RMSE is on log scale since `SalePrice` is log transformed.
> Stacking Ensemble Holdout CV RMSE (on log-scale): 0.1156916071118393

---

## Project Structure

```
├── data/
│   ├── raw/                  # Original CSV from Kaggle
│   ├── processed/            # Encoded parquet files (train/holdout)
│   └── external/             # EDA HTML report
│
├── models/                   # Saved models and Optuna studies
│   ├── encoding_pipe.pkl
│   ├── study_lgb.pkl
│   ├── study_xgb.pkl
│   ├── study_rf.pkl
│   ├── study_ridge.pkl
│   ├── lgb_model.pkl
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   ├── ridge_model.pkl
│   ├── ensemble_model.pkl
│   └── stack_model.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       ├── evaluate.py
│       ├── predict.py
│       ├── stacking.py
│       └── train.py
│
├── main.py
└── requirements.txt
```
---

## Pipeline Overview

```
load_raw_data()
      │
   split()                         ← holdout 15% held out before any transformation
      │
changes_df() / changes_df_holdout()   ← imputation (leakage-safe: neighborhood medians
      │                                   derived from train only)
engineer_features()
      │
pre_encoding()                     ← ordinal fillna
      │
encoding()                         ← fit OrdinalEncoder + OneHotEncoder on train only
      │
train_all()                        ← LGB, XGB, RF, Ridge (with StandardScaler)
      │
train_stack()                      ← StackingRegressor with Ridge meta-learner
      │
score_all()                        ← CV RMSE + holdout RMSE for all models
```

---

## Setup

```bash
git clone <repo>
cd ames-housing
pip install -r requirements.txt
```

Place the raw Kaggle CSV at `data/raw/train_raw.csv`.

---

## Usage

**Step 1 — Run Optuna hyperparameter tuning** (one-time, saves studies to `models/`):
```bash
python -m src.models.tune
```

**Step 2 — Run the full pipeline:**
```bash
python main.py
```

This will:
- Preprocess and engineer features
- Fit the encoding pipeline
- Train and save all four base models
- Train and save the stacking ensemble
- Print CV and holdout RMSE for all models

---

## Methodology

### Data Cleaning
- Rows 523, 1298 removed as multi-feature outliers detected via IQR flagging
- Row 1379 dropped due to missing `Electrical` value
- NaN values in amenity columns (`PoolQC`, `GarageType`, `BsmtQual`, etc.) imputed as `"NA"` — representing genuine absence of the amenity, not missing data
- `LotFrontage` imputed using per-neighborhood training medians to avoid leakage

### Feature Engineering
18 features engineered from raw columns, including:
- `HouseAge`, `HouseRemodelAge`, `IsRemodeled`
- `TotalLivingArea` (above ground + basement)
- `TotalBaths`, `TotalPorchSF`
- `HasBasement`, `HasGarage`, `HasFireplace`, `HasPorch`, `HasSecondFloor`
- `FinishedBasementRatioPrimary`, `FinishedBasementRatioTotal`
- `MoSold_sin`, `MoSold_cos` (cyclical month encoding)
- `GarageAge`, `PremiumRoof`, `GasHeating`

### Encoding
- 19 ordinal columns encoded with explicit category orderings via `OrdinalEncoder`
- 18 nominal columns one-hot encoded via `OneHotEncoder`
- Encoding pipeline fit only on training data and saved for inference

### Modeling
Baseline comparison across 8 models; top 4 selected for tuning:

| Model | Baseline CV RMSE |
|---|---|
| Ridge | 0.1222 |
| GBR | 0.1245 |
| LightGBM | 0.1314 |
| RandomForest | 0.1373 |
| XGBoost | 0.1375 |

Each of the top 4 tuned with **Optuna (TPE sampler, 100 trials)** using 5-fold CV. Final stacking ensemble combines all four tuned models with a Ridge meta-learner.

---

## Key Design Decisions

- **Holdout split before all transformations** — prevents any leakage from holdout into preprocessing statistics
- **Neighborhood median imputation for LotFrontage** — more accurate than global median; medians saved for inference
- **Ridge always wrapped in StandardScaler** — ordinal/numerical features are on different scales; critical for Ridge regularization to work correctly
- **Optuna studies saved separately** — allows reloading best params without re-running 400 CV fits

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
lightgbm
optuna
joblib
scipy
ydata-profiling
matplotlib
seaborn
pyarrow
```
