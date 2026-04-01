import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model (prefer final notebook artifact)
model_path = Path("best_model.joblib") if Path("best_model.joblib").exists() else Path("model.pkl")
model = joblib.load(model_path)


def get_expected_columns(loaded_model):
    """Infer the training feature list from the fitted pipeline/voter model."""
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)

    if hasattr(loaded_model, "estimators_") and loaded_model.estimators_:
        first_est = loaded_model.estimators_[0]
        if hasattr(first_est, "feature_names_in_"):
            return list(first_est.feature_names_in_)
        if hasattr(first_est, "named_steps") and "preprocessing" in first_est.named_steps:
            prep = first_est.named_steps["preprocessing"]
            if hasattr(prep, "feature_names_in_"):
                return list(prep.feature_names_in_)

    return ["Age", "Gender", "family_history", "work_interfere", "benefits"]


EXPECTED_COLUMNS = get_expected_columns(model)


def load_reference_data():
    """Load training-like data to derive robust UI choices for each feature."""
    data = pd.read_csv("survey.csv")
    if "treatment" in data.columns:
        data["treatment"] = data["treatment"].map({"Yes": 1, "No": 0})

    drop_cols = ["Timestamp", "comments", "state"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")

    if "treatment" in data.columns:
        data = data.drop(columns=["treatment"])

    return data


REFERENCE_DF = load_reference_data()


def normalize_categorical_value(column_name, value):
    val = str(value).strip()
    if column_name == "Gender":
        lower = val.lower()
        if lower in {"m", "male"}:
            return "Male"
        if lower in {"f", "female"}:
            return "Female"
    return val


def build_categorical_options():
    options_map = {}
    for col_name in EXPECTED_COLUMNS:
        if col_name == "Age":
            continue

        if col_name not in REFERENCE_DF.columns:
            options_map[col_name] = []
            continue

        values = (
            REFERENCE_DF[col_name]
            .dropna()
            .astype(str)
            .str.strip()
        )
        values = [normalize_categorical_value(col_name, v) for v in values if v]
        unique_vals = sorted(set(values))
        options_map[col_name] = unique_vals

    return options_map


CATEGORICAL_OPTIONS = build_categorical_options()


def pretty_label(column_name):
    return column_name.replace("_", " ").strip().title()

st.title("Mental Health Treatment Predictor")
st.caption(f"Using model: {model_path.name}")
st.write("Fill the inputs below. Providing more fields improves prediction quality.")

# Build form dynamically for all expected model features.
user_input = {}
left_col, right_col = st.columns(2)

for index, col_name in enumerate(EXPECTED_COLUMNS):
    target_col = left_col if index % 2 == 0 else right_col

    with target_col:
        if col_name == "Age":
            user_input[col_name] = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25,
                step=1,
            )
        else:
            options = CATEGORICAL_OPTIONS.get(col_name, [])

            if options:
                default_index = 0
                if col_name == "no_employees" and "26-100" in options:
                    default_index = options.index("26-100")
                elif "Don't know" in options:
                    default_index = options.index("Don't know")
                elif "No" in options:
                    default_index = options.index("No")

                user_input[col_name] = st.selectbox(
                    pretty_label(col_name),
                    options,
                    index=default_index,
                )
            else:
                user_input[col_name] = st.text_input(pretty_label(col_name), value="")

if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=EXPECTED_COLUMNS)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("This person is likely to seek mental health treatment.")
    else:
        st.success("This person is not likely to seek mental health treatment.")

    st.info(f"Predicted probability of seeking treatment: {probability:.2%}")