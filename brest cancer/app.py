import joblib
import numpy as np
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")

MODEL_PATH = "breast_cancer_ann.keras"
SCALER_PATH = "scaler.pkl"

BASE = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
# Same column order as the CSV / notebook
FEATURES = (
    [f"mean_{b}" for b in BASE]
    + [f"{b}_error" for b in BASE]
    + [f"worst_{b}" for b in BASE]
)

# (values, actual diagnosis) taken from the dataset
SAMPLES = {
    "Sample 1": ([15.7, 20.31, 101.2, 766.6, 0.09597, 0.08799, 0.06593, 0.05189, 0.1618, 0.05549, 0.3699, 1.15, 2.406, 40.98, 0.004626, 0.02263, 0.01954, 0.009767, 0.01547, 0.00243, 20.11, 32.82, 129.3, 1269.0, 0.1414, 0.3547, 0.2902, 0.1541, 0.3437, 0.08631], "Malignant"),
    "Sample 2": ([13.46, 18.75, 87.44, 551.1, 0.1075, 0.1138, 0.04201, 0.03152, 0.1723, 0.06317, 0.1998, 0.6068, 1.443, 16.07, 0.004413, 0.01443, 0.01509, 0.007369, 0.01354, 0.001787, 15.35, 25.16, 101.9, 719.8, 0.1624, 0.3124, 0.2654, 0.1427, 0.3518, 0.08665], "Benign"),
    "Sample 3": ([11.06, 17.12, 71.25, 366.5, 0.1194, 0.1071, 0.04063, 0.04268, 0.1954, 0.07976, 0.1779, 1.03, 1.318, 12.3, 0.01262, 0.02348, 0.018, 0.01285, 0.0222, 0.008313, 11.69, 20.74, 76.08, 411.1, 0.1662, 0.2031, 0.1256, 0.09514, 0.278, 0.1168], "Benign"),
    "Sample 4": ([20.26, 23.03, 132.4, 1264.0, 0.09078, 0.1313, 0.1465, 0.08683, 0.2095, 0.05649, 0.7576, 1.509, 4.554, 87.87, 0.006016, 0.03482, 0.04232, 0.01269, 0.02657, 0.004411, 24.22, 31.59, 156.1, 1750.0, 0.119, 0.3539, 0.4098, 0.1573, 0.3689, 0.08368], "Malignant"),
    "Sample 5": ([18.45, 21.91, 120.2, 1075.0, 0.0943, 0.09709, 0.1153, 0.06847, 0.1692, 0.05727, 0.5959, 1.202, 3.766, 68.35, 0.006001, 0.01422, 0.02855, 0.009148, 0.01492, 0.002205, 22.52, 31.39, 145.6, 1590.0, 0.1465, 0.2275, 0.3965, 0.1379, 0.3109, 0.0761], "Malignant"),
    "Sample 6": ([10.16, 19.59, 64.73, 311.7, 0.1003, 0.07504, 0.005025, 0.01116, 0.1791, 0.06331, 0.2441, 2.09, 1.648, 16.8, 0.01291, 0.02222, 0.004174, 0.007082, 0.02572, 0.002278, 10.65, 22.88, 67.88, 347.3, 0.1265, 0.12, 0.01005, 0.02232, 0.2262, 0.06742], "Benign"),
}
PLACEHOLDER = "— choose a sample —"


@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def set_values(values):
    for name, v in zip(FEATURES, values):
        st.session_state[name] = float(v)


def on_sample_change():
    choice = st.session_state.sample_choice
    if choice in SAMPLES:
        set_values(SAMPLES[choice][0])


def on_paste_apply():
    raw = st.session_state.pasted.replace("[", "").replace("]", "")
    try:
        values = [float(x) for x in raw.replace("\n", ",").split(",") if x.strip()]
    except ValueError:
        st.session_state.paste_error = "Could not read the numbers. Use comma-separated values."
        return
    if len(values) != len(FEATURES):
        st.session_state.paste_error = f"Expected {len(FEATURES)} values, got {len(values)}."
        return
    st.session_state.paste_error = None
    set_values(values)


# Initialise inputs once
if "initialised" not in st.session_state:
    set_values(SAMPLES["Sample 2"][0])
    st.session_state.initialised = True
    st.session_state.paste_error = None

st.title("🩺 Breast Cancer Prediction")
st.caption(
    "An artificial neural network trained on the Wisconsin Breast Cancer dataset. "
    "For learning purposes only, not a medical diagnosis."
)

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(
        f"Could not load `{MODEL_PATH}` / `{SCALER_PATH}`. Put both files in the same "
        f"folder as app.py.\n\nDetails: {e}"
    )
    st.stop()

# ---- Quick-fill options ----
left, right = st.columns(2)
with left:
    st.selectbox(
        "Load a sample from the dataset",
        [PLACEHOLDER] + list(SAMPLES),
        key="sample_choice",
        on_change=on_sample_change,
    )
    if st.session_state.sample_choice in SAMPLES:
        st.caption(f"Actual diagnosis: **{SAMPLES[st.session_state.sample_choice][1]}**")
with right:
    st.text_area(
        "…or paste 30 comma-separated values",
        key="pasted",
        height=100,
        placeholder="[15.7, 20.31, 101.2, ...]",
    )
    st.button("Apply pasted values", on_click=on_paste_apply)
    if st.session_state.paste_error:
        st.error(st.session_state.paste_error)

st.divider()

# ---- Feature inputs ----
tabs = st.tabs(["Mean values", "Standard error", "Worst values"])
for t, start in zip(tabs, (0, 10, 20)):
    with t:
        cols = st.columns(3)
        for i, name in enumerate(FEATURES[start:start + 10]):
            with cols[i % 3]:
                st.number_input(
                    name.replace("_", " ").title(),
                    key=name,
                    min_value=0.0,
                    format="%.6f",
                )

# ---- Prediction ----
if st.button("Predict", type="primary", use_container_width=True):
    x = np.array([[st.session_state[n] for n in FEATURES]])
    prob = float(model.predict(scaler.transform(x), verbose=0)[0][0])

    st.subheader("Result")
    if prob > 0.5:
        st.error("Prediction: **Malignant** (positive)")
        confidence = prob
    else:
        st.success("Prediction: **Benign** (negative)")
        confidence = 1 - prob

    c1, c2 = st.columns(2)
    c1.metric("Malignancy probability", f"{prob:.1%}")
    c2.metric("Model confidence", f"{confidence:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
