"""
Predictive Maintenance AI — Streamlit GUI

This app does NOT train or modify any model. It loads the artifacts
persisted by preprocessing/train_and_save.py (scaler, encoders, and the
three XGBoost models) and runs them through utils/predict.py, which
reproduces the exact preprocessing pipeline from the original notebook.

Run with:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from utils.predict import predict, get_config  # noqa: E402

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem;
    }
    .stButton>button {
        font-weight: 600;
        height: 3em;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    try:
        return get_config(), None
    except Exception as e:
        return None, str(e)


cfg, load_err = load_config()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🛠️ Predictive Maintenance AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enter live sensor readings for a machine to get an AI-driven risk '
    'assessment: 24-hour failure risk, most likely failure type, and estimated repair cost — '
    'powered by XGBoost models trained on historical sensor and maintenance data.</div>',
    unsafe_allow_html=True,
)

if load_err:
    st.error(
        "⚠️ Could not load the trained models. Make sure you've run "
        "`python preprocessing/train_and_save.py` first so the `models/` folder is populated.\n\n"
        f"Details: {load_err}"
    )
    st.stop()

if "result" not in st.session_state:
    st.session_state.result = None
if "form_key" not in st.session_state:
    st.session_state.form_key = 0


def reset_form():
    st.session_state.result = None
    st.session_state.form_key += 1


# ---------------------------------------------------------------------------
# Sidebar - about / info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About this system")
    st.write(
        "This tool uses three XGBoost models trained on historical machine "
        "sensor data to predict:"
    )
    st.markdown(
        "- **Failure risk** within the next 24 hours\n"
        "- **Most likely failure type**, if any\n"
        "- **Estimated repair cost**"
    )
    st.divider()
    st.subheader("Model performance (test set)")
    m = cfg["metrics"]
    st.caption("Failure within 24h (classifier)")
    st.write(f"Accuracy: **{m['failure_within_24h']['Accuracy']*100:.1f}%**  |  F1: **{m['failure_within_24h']['F1']:.3f}**")
    st.caption("Failure type (classifier)")
    st.write(f"Accuracy: **{m['failure_type']['Accuracy']*100:.1f}%**  |  F1 (macro): **{m['failure_type']['F1']:.3f}**")
    st.caption("Estimated repair cost (regressor)")
    st.write(f"MAE: **${m['estimated_repair_cost']['MAE']:.0f}**  |  R²: **{m['estimated_repair_cost']['R2']:.1f}%**")
    st.divider()
    st.caption(
        "Note: this deployment covers failure risk, failure type, and repair "
        "cost only. Remaining Useful Life (RUL) prediction was intentionally "
        "excluded — no RUL model exists in the source project."
    )

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("📥 Machine & Sensor Inputs")

ranges = cfg["raw_input_ranges"]

with st.form(key=f"predict_form_{st.session_state.form_key}"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Machine identity**")
        machine_type = st.selectbox(
            "Machine type",
            options=cfg["machine_types"],
            help="Type of equipment being monitored.",
        )
        operating_mode = st.selectbox(
            "Operating mode",
            options=cfg["operating_mode_order"],
            index=1,
            help="Current operating regime of the machine.",
        )
        hours_since_maintenance = st.number_input(
            "Hours since last maintenance",
            min_value=0.0,
            max_value=float(ranges["hours_since_maintenance"]["max"]) * 1.5,
            value=float(round(ranges["hours_since_maintenance"]["median"], 1)),
            step=1.0,
            help=f"Typical range in training data: "
                 f"{ranges['hours_since_maintenance']['min']:.0f} – {ranges['hours_since_maintenance']['max']:.0f} hrs",
        )
        ambient_temp = st.number_input(
            "Ambient temperature (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=float(round(ranges["ambient_temp"]["median"], 1)),
            step=0.1,
            help=f"Typical range: {ranges['ambient_temp']['min']:.1f} – {ranges['ambient_temp']['max']:.1f} °C",
        )

    with col2:
        st.markdown("**Sensor readings**")
        vibration_rms = st.number_input(
            "Vibration RMS",
            min_value=0.0,
            value=float(round(ranges["vibration_rms"]["median"], 2)),
            step=0.01,
            help=f"Typical range: {ranges['vibration_rms']['min']:.2f} – {ranges['vibration_rms']['max']:.2f}",
        )
        temperature_motor = st.number_input(
            "Motor temperature (°C)",
            value=float(round(ranges["temperature_motor"]["median"], 1)),
            step=0.1,
            help=f"Typical range: {ranges['temperature_motor']['min']:.1f} – {ranges['temperature_motor']['max']:.1f} °C",
        )
        current_phase_avg = st.number_input(
            "Current phase average (A)",
            min_value=0.0,
            value=float(round(ranges["current_phase_avg"]["median"], 2)),
            step=0.01,
            help=f"Typical range: {ranges['current_phase_avg']['min']:.2f} – {ranges['current_phase_avg']['max']:.2f} A",
        )
        pressure_level = st.number_input(
            "Pressure level",
            min_value=0.0,
            value=float(round(ranges["pressure_level"]["median"], 1)),
            step=0.1,
            help=f"Typical range: {ranges['pressure_level']['min']:.1f} – {ranges['pressure_level']['max']:.1f}",
        )
        rpm = st.number_input(
            "RPM",
            min_value=0.0,
            value=float(round(ranges["rpm"]["median"], 1)),
            step=1.0,
            help=f"Typical range: {ranges['rpm']['min']:.1f} – {ranges['rpm']['max']:.1f}",
        )

    col_predict, col_reset = st.columns([3, 1])
    with col_predict:
        submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")
    with col_reset:
        cleared = st.form_submit_button("↺ Reset", use_container_width=True)

if cleared:
    reset_form()
    st.rerun()

if submitted:
    raw_input = {
        "machine_type": machine_type,
        "vibration_rms": vibration_rms,
        "temperature_motor": temperature_motor,
        "current_phase_avg": current_phase_avg,
        "pressure_level": pressure_level,
        "rpm": rpm,
        "operating_mode": operating_mode,
        "hours_since_maintenance": hours_since_maintenance,
        "ambient_temp": ambient_temp,
    }

    # Basic validation beyond Streamlit's built-in numeric constraints
    errors = []
    for k, v in raw_input.items():
        if k in ("machine_type", "operating_mode"):
            continue
        if v is None:
            errors.append(f"'{k}' is required.")
        elif isinstance(v, (int, float)) and v != v:  # NaN check
            errors.append(f"'{k}' must be a valid number.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Running prediction pipeline..."):
            try:
                result = predict(raw_input)
                st.session_state.result = result
            except Exception as e:
                st.session_state.result = None
                st.error(
                    "Something went wrong while generating the prediction. "
                    f"Details: {e}"
                )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.result:
    r = st.session_state.result
    st.divider()
    st.subheader("📊 Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        risk_label = "⚠️ HIGH RISK" if r["failure_within_24h"] == 1 else "✅ LOW RISK"
        st.metric(
            label="Failure risk (next 24h)",
            value=risk_label,
            delta=f"{r['failure_within_24h_probability']*100:.1f}% probability",
            delta_color="inverse",
        )

    with c2:
        st.metric(
            label="Most likely failure type",
            value=r["failure_type"].replace("_", " ").title(),
            delta=f"{r['failure_type_confidence']*100:.1f}% confidence",
        )

    with c3:
        st.metric(
            label="Estimated repair cost",
            value=f"${r['estimated_repair_cost']:,.2f}",
        )

    if r["failure_within_24h"] == 1:
        st.warning(
            f"This machine shows a **high risk of failure within 24 hours** "
            f"(predicted type: **{r['failure_type'].replace('_', ' ')}**). "
            "Consider scheduling maintenance soon."
        )
    else:
        st.success("This machine currently shows a low risk of failure within the next 24 hours.")

    with st.expander("See raw prediction details"):
        st.json(r)
else:
    st.info("Fill in the machine and sensor values above, then click **Predict** to see results.")
