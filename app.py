import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Mental Health Treatment Predictor",
                   page_icon="🧠", layout="centered")

MODEL_PATH = Path("trained_models/best_model.joblib")
DROP_COLS = ["Timestamp", "comments", "state"]


# ----------------------------------------------------------------------
# Shared cleaning — MUST match the notebook so the app's dropdowns show
# exactly the categories the model was trained on.
# ----------------------------------------------------------------------
def normalize_gender(value):
    s = str(value).strip().lower()
    male = {"m", "male", "cis male", "man", "maile", "mal", "msle", "mail",
            "make", "malr", "cis man", "male (cis)", "guy (-ish) ^_^", "male-ish"}
    female = {"f", "female", "cis female", "woman", "femake", "femail",
              "cis-female/femme", "female (cis)", "female (trans)", "trans woman"}
    if s in male:
        return "Male"
    if s in female:
        return "Female"
    return "Other"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    """Prefer the cleaned dataset the notebook saves; otherwise clean the raw file
    on the fly so the dropdowns are always tidy and consistent with training."""
    cleaned = Path("survey_cleaned.csv")
    if cleaned.exists():
        data = pd.read_csv(cleaned)
    else:
        data = pd.read_csv("survey.csv")
        if "treatment" in data.columns:
            data["treatment"] = data["treatment"].map({"Yes": 1, "No": 0})
        data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")
        data = data[(data["Age"] >= 18) & (data["Age"] <= 80)].copy()
        if "Gender" in data.columns:
            data["Gender"] = data["Gender"].apply(normalize_gender)
    if "treatment" in data.columns:
        data = data.drop(columns=["treatment"])
    return data


def get_expected_columns(loaded_model, reference_df):
    """Infer the training feature list from the fitted pipeline/voting model."""
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)
    if hasattr(loaded_model, "estimators_") and loaded_model.estimators_:
        first = loaded_model.estimators_[0]
        if hasattr(first, "feature_names_in_"):
            return list(first.feature_names_in_)
        if hasattr(first, "named_steps") and "preprocessing" in first.named_steps:
            prep = first.named_steps["preprocessing"]
            if hasattr(prep, "feature_names_in_"):
                return list(prep.feature_names_in_)
    return list(reference_df.columns)


def pretty_label(name):
    return name.replace("_", " ").strip().title()


def build_options(expected_cols, reference_df):
    """Sorted, de-duplicated category lists for each non-Age feature."""
    opts = {}
    for col in expected_cols:
        if col == "Age" or col not in reference_df.columns:
            opts[col] = []
            continue
        vals = reference_df[col].dropna().astype(str).str.strip()
        opts[col] = sorted(v for v in set(vals) if v and v.lower() != "nan")
    return opts


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
st.title("🧠 Mental Health Treatment Predictor")
st.caption("Predicts whether someone is likely to have sought mental health "
           "treatment, from workplace and personal factors (OSMI Tech Survey).")

model = load_model()
if model is None:
    st.error("Model not found at `trained_models/best_model.joblib`. "
             "Run the notebook end-to-end first to generate it.")
    st.stop()

reference_df = load_reference_data()
EXPECTED_COLUMNS = get_expected_columns(model, reference_df)
OPTIONS = build_options(EXPECTED_COLUMNS, reference_df)

# sensible default selections for a few well-known fields
PREFERRED_DEFAULTS = {"no_employees": "26-100"}

with st.form("predict_form"):
    st.write("Fill in the details below, then click **Predict**.")
    user_input = {}
    left, right = st.columns(2)

    for i, col in enumerate(EXPECTED_COLUMNS):
        target = left if i % 2 == 0 else right
        with target:
            if col == "Age":
                user_input[col] = st.number_input("Age", min_value=18, max_value=80,
                                                  value=30, step=1)
            else:
                choices = OPTIONS.get(col, [])
                if choices:
                    default_idx = 0
                    pref = PREFERRED_DEFAULTS.get(col)
                    if pref and pref in choices:
                        default_idx = choices.index(pref)
                    elif "No" in choices:
                        default_idx = choices.index("No")
                    elif "Don't know" in choices:
                        default_idx = choices.index("Don't know")
                    user_input[col] = st.selectbox(pretty_label(col), choices, index=default_idx)
                else:
                    user_input[col] = st.text_input(pretty_label(col), value="")

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([user_input], columns=EXPECTED_COLUMNS)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### Result")
    if prediction == 1:
        st.warning("**Likely to have sought mental health treatment.**")
    else:
        st.success("**Not likely to have sought mental health treatment.**")

    st.metric("Probability of seeking treatment", f"{probability:.1%}")
    st.progress(float(probability))
    st.caption("This is a statistical estimate from survey data, not a clinical "
               "or diagnostic judgement.")

st.divider()
st.caption("Model: soft-voting ensemble (LogReg · Decision Tree · KNN · SVM). "
           "Dropdown options are derived from the cleaned training data, so they "
           "match the categories the model actually learned.")