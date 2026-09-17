"""
predict.py

Single shared prediction function used by the GUI (and by the validation
script). It reproduces, for one new input row, exactly the same
preprocessing steps applied at training time:

  1. no imputation needed (GUI enforces all fields are filled)
  2. OrdinalEncoder.transform() on operating_mode (fitted encoder)
  3. one-hot expansion of machine_type, aligned to the training column set
  4. StandardScaler.transform() on the 8 numeric feature columns (fitted scaler)
  5. reindex to the exact training feature column order
  6. run the three persisted models and inverse-transform failure_type
"""
import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_scaler = None
_operating_mode_encoder = None
_failure_type_le = None
_model_fail24h = None
_model_failtype = None
_model_cost = None
_config = None


def _load_artifacts():
    """Lazy-load all artifacts once per process."""
    global _scaler, _operating_mode_encoder, _failure_type_le
    global _model_fail24h, _model_failtype, _model_cost, _config

    if _scaler is not None:
        return

    _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    _operating_mode_encoder = joblib.load(MODELS_DIR / "operating_mode_encoder.joblib")
    _failure_type_le = joblib.load(MODELS_DIR / "failure_type_label_encoder.joblib")
    _model_fail24h = joblib.load(MODELS_DIR / "model_failure_within_24h.joblib")
    _model_failtype = joblib.load(MODELS_DIR / "model_failure_type.joblib")
    _model_cost = joblib.load(MODELS_DIR / "model_estimated_repair_cost.joblib")
    with open(MODELS_DIR / "config.json") as f:
        _config = json.load(f)


def get_config():
    _load_artifacts()
    return _config


def build_feature_row(raw_input: dict) -> pd.DataFrame:
    """
    raw_input keys expected (raw, un-encoded, un-scaled values):
      machine_type            (str, one of config['machine_types'])
      vibration_rms            (float)
      temperature_motor        (float)
      current_phase_avg        (float)
      pressure_level           (float)
      rpm                      (float)
      operating_mode           (str, one of ['idle','normal','peak'])
      hours_since_maintenance  (float)
      ambient_temp             (float)

    Returns a single-row DataFrame with columns in the exact order the
    models were trained on.
    """
    _load_artifacts()
    cfg = _config

    row = {
        'vibration_rms': raw_input['vibration_rms'],
        'temperature_motor': raw_input['temperature_motor'],
        'current_phase_avg': raw_input['current_phase_avg'],
        'pressure_level': raw_input['pressure_level'],
        'rpm': raw_input['rpm'],
        'hours_since_maintenance': raw_input['hours_since_maintenance'],
        'ambient_temp': raw_input['ambient_temp'],
    }
    df = pd.DataFrame([row])

    # operating_mode -> ordinal encode (same fitted encoder as training)
    df['operating_mode'] = _operating_mode_encoder.transform(
        [[raw_input['operating_mode']]]
    )[0][0]

    # machine_type -> one-hot, aligned to training columns
    for col in cfg['machine_type_cols']:
        df[col] = 0
    mt_col = f"machine_type_{raw_input['machine_type']}"
    if mt_col in cfg['machine_type_cols']:
        df[mt_col] = 1
    else:
        raise ValueError(
            f"Unknown machine_type '{raw_input['machine_type']}'. "
            f"Expected one of {cfg['machine_types']}."
        )

    # scale the numeric feature columns with the fitted scaler
    df[cfg['feature_cols_to_scale']] = _scaler.transform(df[cfg['feature_cols_to_scale']])

    # final column order must match training exactly
    df = df[cfg['all_feature_cols']]
    return df


def predict(raw_input: dict) -> dict:
    """Run all three models on one raw input and return human-readable results."""
    _load_artifacts()
    X = build_feature_row(raw_input)

    fail24h_pred = int(_model_fail24h.predict(X)[0])
    fail24h_proba = float(_model_fail24h.predict_proba(X)[0][1])

    failtype_encoded = _model_failtype.predict(X)[0]
    failtype_label = _failure_type_le.inverse_transform([failtype_encoded])[0]
    failtype_proba_arr = _model_failtype.predict_proba(X)[0]
    failtype_confidence = float(failtype_proba_arr.max())

    cost_pred_raw = float(_model_cost.predict(X)[0])
    # Display-only clipping: the model can output small negative values for
    # very low-risk inputs (raw XGBoost regression, not from any notebook
    # logic). We floor the *displayed* figure at 0 since a negative repair
    # cost isn't meaningful to a user; the raw model output is kept
    # available (estimated_repair_cost_raw) for validation purposes.
    cost_pred_display = max(0.0, cost_pred_raw)

    return {
        "failure_within_24h": fail24h_pred,
        "failure_within_24h_probability": fail24h_proba,
        "failure_type": str(failtype_label),
        "failure_type_confidence": failtype_confidence,
        "estimated_repair_cost": cost_pred_display,
        "estimated_repair_cost_raw": cost_pred_raw,
    }
