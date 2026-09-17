"""
validate.py

Proves the persisted pipeline (utils/predict.py) produces IDENTICAL
predictions to running the notebook's own preprocessing + models in one
continuous in-memory session, for the same raw inputs.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.predict import predict, get_config  # noqa: E402

# --- Re-run the notebook's exact in-memory pipeline (no persistence) to get a "ground truth" ---
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

df_raw = pd.read_csv(BASE_DIR / "data" / "predictive_maintenance_v3.csv")
df = df_raw.copy()
machine_types = df['machine_type'].unique()
numeric_cols = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                 'pressure_level', 'rpm', 'hours_since_maintenance',
                 'ambient_temp', 'rul_hours']
missing_cols = ['vibration_rms', 'temperature_motor', 'current_phase_avg', 'pressure_level', 'rpm']
for col in missing_cols:
    df[col] = df.groupby('machine_type')[col].transform(lambda x: x.fillna(x.median()))

rows_to_drop = pd.Series(False, index=df.index)
for m in machine_types:
    mask_type = df['machine_type'] == m
    sub_df = df[mask_type]
    for col in numeric_cols:
        Q1, Q3 = sub_df[col].quantile(0.25), sub_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        is_outlier = (df[col] < lower) | (df[col] > upper)
        is_no_failure = (df['failure_within_24h'] == 0) & (df['failure_type'] == 'none')
        rows_to_drop |= (mask_type & is_outlier & is_no_failure)
df = df[~rows_to_drop].reset_index(drop=True)

operating_mode_order = ['idle', 'normal', 'peak']
ordinal_encoder = OrdinalEncoder(categories=[operating_mode_order])
df['operating_mode'] = ordinal_encoder.fit_transform(df[['operating_mode']])
df = pd.get_dummies(df, columns=['machine_type'], prefix='machine_type', dtype=int)
machine_type_cols = sorted([c for c in df.columns if c.startswith('machine_type_')])

feature_cols = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                 'pressure_level', 'rpm', 'operating_mode',
                 'hours_since_maintenance', 'ambient_temp']
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
all_feature_cols = feature_cols + machine_type_cols

X = df[all_feature_cols]
y = df[['failure_within_24h', 'failure_type', 'estimated_repair_cost']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['failure_within_24h']
)

clf1 = XGBClassifier(random_state=42, eval_metric='logloss')
clf1.fit(X_train, y_train['failure_within_24h'])

le = LabelEncoder()
y_train_ft = le.fit_transform(y_train['failure_type'])
clf2 = XGBClassifier(random_state=42, eval_metric='mlogloss')
clf2.fit(X_train, y_train_ft)

reg = XGBRegressor(n_estimators=2500, learning_rate=0.046, max_depth=7, min_child_weight=2,
                    subsample=0.9, colsample_bytree=0.9, gamma=0, reg_alpha=0.05, reg_lambda=1,
                    random_state=87, n_jobs=-1)
reg.fit(X_train, y_train['estimated_repair_cost'])

# --- pick 5 raw test rows from the ORIGINAL (unscaled, unencoded) CSV to run through both paths ---
rng = np.random.default_rng(7)
raw_test_indices = sorted(rng.choice(len(df_raw), size=200, replace=False).tolist())
mismatches = 0

checked = 0
for idx in raw_test_indices:
    raw_row = df_raw.iloc[idx]

    raw_input = {
        "machine_type": raw_row["machine_type"],
        "vibration_rms": raw_row["vibration_rms"],
        "temperature_motor": raw_row["temperature_motor"],
        "current_phase_avg": raw_row["current_phase_avg"],
        "pressure_level": raw_row["pressure_level"],
        "rpm": raw_row["rpm"],
        "operating_mode": raw_row["operating_mode"],
        "hours_since_maintenance": raw_row["hours_since_maintenance"],
        "ambient_temp": raw_row["ambient_temp"],
    }

    # skip rows with NaN raw values (imputation is a training-time-only step)
    if pd.isna(pd.Series(raw_input)).any():
        continue

    # notebook-style in-memory pipeline for this single raw row
    nb_row = pd.DataFrame([{
        'vibration_rms': raw_input['vibration_rms'],
        'temperature_motor': raw_input['temperature_motor'],
        'current_phase_avg': raw_input['current_phase_avg'],
        'pressure_level': raw_input['pressure_level'],
        'rpm': raw_input['rpm'],
        'hours_since_maintenance': raw_input['hours_since_maintenance'],
        'ambient_temp': raw_input['ambient_temp'],
    }])
    nb_row['operating_mode'] = ordinal_encoder.transform([[raw_input['operating_mode']]])[0][0]
    for c in machine_type_cols:
        nb_row[c] = 0
    nb_row[f"machine_type_{raw_input['machine_type']}"] = 1
    nb_row[feature_cols] = scaler.transform(nb_row[feature_cols])
    nb_row = nb_row[all_feature_cols]

    nb_fail24h = int(clf1.predict(nb_row)[0])
    nb_failtype = le.inverse_transform([clf2.predict(nb_row)[0]])[0]
    nb_cost = float(reg.predict(nb_row)[0])

    # persisted GUI pipeline
    gui_result = predict(raw_input)

    match_fail24h = nb_fail24h == gui_result['failure_within_24h']
    match_failtype = nb_failtype == gui_result['failure_type']
    match_cost = abs(nb_cost - gui_result['estimated_repair_cost_raw']) < 1e-4

    checked += 1
    if not (match_fail24h and match_failtype and match_cost):
        mismatches += 1
        print(f"MISMATCH at idx {idx}: nb=({nb_fail24h},{nb_failtype},{nb_cost:.4f}) "
              f"gui=({gui_result['failure_within_24h']},{gui_result['failure_type']},{gui_result['estimated_repair_cost_raw']:.4f})")

print(f"Checked {checked} raw rows.")
if mismatches == 0:
    print("VALIDATION PASSED: persisted pipeline matches notebook-style pipeline exactly.")
else:
    print(f"VALIDATION FAILED: {mismatches} mismatches found.")
    sys.exit(1)
