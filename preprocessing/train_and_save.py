"""
train_and_save.py

This script does NOT introduce any new preprocessing or feature-engineering
logic. Every step below is copied 1:1 from the original notebook
(final_day_project.ipynb):

  - Cell 5   -> missing value imputation (group median by machine_type)
  - Cell 11  -> outlier removal (IQR, per machine_type, non-failure rows only)
  - Cell 15  -> OrdinalEncoder on operating_mode + one-hot on machine_type
  - Cell 17  -> StandardScaler on the 8 numeric feature columns
  - Cell 19  -> train/test split (random_state=42, stratified on failure_within_24h)
  - Cell 21/25 -> classifier comparison for failure_within_24h / failure_type
                  (XGBoost was the best performer for both in the notebook's
                  own benchmark, so it is the model persisted for deployment)
  - Cell 29  -> XGBRegressor for estimated_repair_cost (exact hyperparameters
                  copied verbatim from the notebook)

The ONLY thing this script adds that the notebook did not do is: it SAVES
the fitted scaler / encoder / models to disk (joblib), because the notebook
never persisted anything. No RUL model is trained (per explicit instruction
to skip RUL).
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              mean_absolute_error, mean_squared_error, r2_score)
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "predictive_maintenance_v3.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load  (notebook cell 1)
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(DATA_PATH)
df = df_raw.copy()

machine_types = df['machine_type'].unique()
numeric_cols = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                 'pressure_level', 'rpm', 'hours_since_maintenance',
                 'ambient_temp', 'rul_hours']

# ---------------------------------------------------------------------------
# 2. Fill missing values (notebook cell 5) - group median by machine_type
# ---------------------------------------------------------------------------
missing_cols = ['vibration_rms', 'temperature_motor',
                 'current_phase_avg', 'pressure_level', 'rpm']

for col in missing_cols:
    df[col] = df.groupby('machine_type')[col].transform(
        lambda x: x.fillna(x.median())
    )

# Also persist the per-machine-type medians so the app can impute a single
# missing field the same way at inference time if ever needed.
impute_medians = {
    m: df_raw[df_raw['machine_type'] == m][missing_cols].median().to_dict()
    for m in machine_types
}

# ---------------------------------------------------------------------------
# 3. Drop outlier rows (notebook cell 11) - IQR per machine_type,
#    only rows that are non-failure (failure_within_24h == 0 & failure_type == 'none')
# ---------------------------------------------------------------------------
rows_to_drop = pd.Series(False, index=df.index)

for m in machine_types:
    mask_type = df['machine_type'] == m
    sub_df = df[mask_type]

    for col in numeric_cols:
        Q1 = sub_df[col].quantile(0.25)
        Q3 = sub_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        is_outlier = (df[col] < lower) | (df[col] > upper)
        is_no_failure = (df['failure_within_24h'] == 0) & (df['failure_type'] == 'none')

        rows_to_drop |= (mask_type & is_outlier & is_no_failure)

df = df[~rows_to_drop].reset_index(drop=True)
print(f"Rows dropped as outliers: {rows_to_drop.sum()} | remaining: {df.shape[0]}")

# ---------------------------------------------------------------------------
# 4. Encoding (notebook cell 15)
# ---------------------------------------------------------------------------
operating_mode_order = ['idle', 'normal', 'peak']
ordinal_encoder = OrdinalEncoder(categories=[operating_mode_order])
df['operating_mode'] = ordinal_encoder.fit_transform(df[['operating_mode']])

df = pd.get_dummies(df, columns=['machine_type'], prefix='machine_type', dtype=int)
machine_type_cols = sorted([c for c in df.columns if c.startswith('machine_type_')])

# ---------------------------------------------------------------------------
# 5. Scaling (notebook cell 17)
# ---------------------------------------------------------------------------
feature_cols = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                 'pressure_level', 'rpm', 'operating_mode',
                 'hours_since_maintenance', 'ambient_temp']

scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

all_feature_cols = feature_cols + machine_type_cols

# ---------------------------------------------------------------------------
# 6. Train / test split (notebook cell 19)
# ---------------------------------------------------------------------------
X = df[all_feature_cols]
y = df[['failure_within_24h', 'failure_type', 'estimated_repair_cost']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df['failure_within_24h']
)

metrics = {}

# ---------------------------------------------------------------------------
# 7. failure_within_24h classifier (notebook cell 21) - XGBoost was the
#    best performer in the notebook's own comparison (Acc 0.979)
# ---------------------------------------------------------------------------
clf_fail24h = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss')
clf_fail24h.fit(X_train, y_train['failure_within_24h'])
pred = clf_fail24h.predict(X_test)
metrics['failure_within_24h'] = {
    'Accuracy': accuracy_score(y_test['failure_within_24h'], pred),
    'Precision': precision_score(y_test['failure_within_24h'], pred),
    'Recall': recall_score(y_test['failure_within_24h'], pred),
    'F1': f1_score(y_test['failure_within_24h'], pred),
}

# ---------------------------------------------------------------------------
# 8. failure_type classifier (notebook cells 24-25) - XGBoost + LabelEncoder
# ---------------------------------------------------------------------------
le_failure_type = LabelEncoder()
y_train_ft_encoded = le_failure_type.fit_transform(y_train['failure_type'])
y_test_ft_encoded = le_failure_type.transform(y_test['failure_type'])

clf_failtype = XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
clf_failtype.fit(X_train, y_train_ft_encoded)
pred_ft = clf_failtype.predict(X_test)
metrics['failure_type'] = {
    'Accuracy': accuracy_score(y_test_ft_encoded, pred_ft),
    'Precision': precision_score(y_test_ft_encoded, pred_ft, average='macro', zero_division=0),
    'Recall': recall_score(y_test_ft_encoded, pred_ft, average='macro', zero_division=0),
    'F1': f1_score(y_test_ft_encoded, pred_ft, average='macro', zero_division=0),
}

# ---------------------------------------------------------------------------
# 9. estimated_repair_cost regressor (notebook cell 29) - exact hyperparams
# ---------------------------------------------------------------------------
reg_cost = XGBRegressor(
    n_estimators=2500,
    learning_rate=0.046,
    max_depth=7,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0,
    reg_alpha=0.05,
    reg_lambda=1,
    random_state=87,
    n_jobs=-1
)
reg_cost.fit(X_train, y_train['estimated_repair_cost'])
pred_cost = reg_cost.predict(X_test)
metrics['estimated_repair_cost'] = {
    'MAE': mean_absolute_error(y_test['estimated_repair_cost'], pred_cost),
    'RMSE': mean_squared_error(y_test['estimated_repair_cost'], pred_cost) ** 0.5,
    'R2': r2_score(y_test['estimated_repair_cost'], pred_cost) * 100,
}

print(json.dumps(metrics, indent=2))

# ---------------------------------------------------------------------------
# 10. Persist everything the app needs
# ---------------------------------------------------------------------------
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
joblib.dump(ordinal_encoder, MODELS_DIR / "operating_mode_encoder.joblib")
joblib.dump(le_failure_type, MODELS_DIR / "failure_type_label_encoder.joblib")
joblib.dump(clf_fail24h, MODELS_DIR / "model_failure_within_24h.joblib")
joblib.dump(clf_failtype, MODELS_DIR / "model_failure_type.joblib")
joblib.dump(reg_cost, MODELS_DIR / "model_estimated_repair_cost.joblib")

config = {
    "feature_cols_to_scale": feature_cols,
    "machine_type_cols": machine_type_cols,
    "all_feature_cols": all_feature_cols,
    "machine_types": sorted(list(machine_types)),
    "operating_mode_order": operating_mode_order,
    "failure_type_classes": le_failure_type.classes_.tolist(),
    "impute_medians": impute_medians,
    "raw_input_ranges": {
        col: {"min": float(df_raw[col].min()), "max": float(df_raw[col].max()),
              "median": float(df_raw[col].median())}
        for col in ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                    'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp']
    },
    "metrics": metrics,
}
with open(MODELS_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nSaved artifacts to:", MODELS_DIR)
for p in sorted(MODELS_DIR.iterdir()):
    print(" -", p.name)
