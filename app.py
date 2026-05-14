import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
try:
    model = joblib.load("logistic_regression_dropout_model.joblib")
except FileNotFoundError:
    st.error(
        "Model file 'logistic_regression_dropout_model.joblib' not found. "
        "Make sure it is inside your project folder."
    )
    st.stop()


# ----------------------------
# Mappings
# ----------------------------
education_mapping = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4
}

age_mapping = {
    "0-35": 0,
    "35-55": 1,
    "55<=": 2
}

imd_mapping = {
    "0-10%": 0,
    "10-20": 1,
    "20-30%": 2,
    "30-40%": 3,
    "40-50%": 4,
    "50-60%": 5,
    "60-70%": 6,
    "70-80%": 7,
    "80-90%": 8,
    "90-100%": 9,
    "Unknown": -1
}


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Student Dropout Prediction", layout="centered")
st.title("Student Dropout Prediction")
st.markdown("Enter student details to predict their dropout risk.")

st.markdown("---")
st.subheader("📋 Input Field Guide")
st.markdown(
    """
    Use the sidebar on the left to fill in the student's details. Here's what each field means:

    | Field | Description |
    |---|---|
    | **Number of Previous Attempts** | How many times the student has previously attempted assignments or tests/quizzes in the course. A higher number may indicate repeated struggles with assessments. |
    | **Studied Credits** | The total number of course credits the student is currently enrolled in. Reflects the student's academic workload. |
    | **Total Clicks (Engagement)** | The total number of times the student has clicked on anything within the Learning Management System (LMS) or portal — including links, buttons, resources, videos, and course materials. This is a measure of overall platform engagement. |
    | **Median Assessment Score** | The student's median score (0–100) across all assessments taken so far. Provides a snapshot of academic performance. |
    | **Disability Status** | Whether the student has a declared disability (*Yes* or *No*). |
    | **Highest Education** | The highest level of education the student had completed before enrolling in this course. |
    | **Age Band** | The student's age group: *0–35*, *35–55*, or *55 and above*. |
    | **IMD Band (Deprivation)** | The Index of Multiple Deprivation (IMD) band for the student's home area. Lower percentages (e.g. *0–10%*) indicate higher deprivation; higher percentages indicate more affluent areas. *Unknown* means no data is available. |
    | **Gender** | The student's gender (*Female* or *Male*). |
    | **Region** | The UK region where the student is based. |
    """
)
st.markdown("---")

# ----------------------------
# Sidebar Inputs
# ----------------------------
with st.sidebar:
    st.header("Student Features")

    num_of_prev_attempts = st.number_input(
        "Number of Previous Attempts",
        min_value=0,
        max_value=10,
        value=0
    )

    studied_credits = st.number_input(
        "Studied Credits",
        min_value=30,
        max_value=655,
        value=60
    )

    sum_click = st.number_input(
        "Total Clicks (Engagement)",
        min_value=0,
        value=500
    )

    score = st.number_input(
        "Median Assessment Score",
        min_value=0,
        max_value=100,
        value=70
    )

    disability = st.selectbox(
        "Disability Status",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    selected_education = st.selectbox(
        "Highest Education",
        options=list(education_mapping.keys())
    )
    highest_education = education_mapping[selected_education]

    selected_age_band = st.selectbox(
        "Age Band",
        options=list(age_mapping.keys())
    )
    age_band = age_mapping[selected_age_band]

    selected_imd_band = st.selectbox(
        "IMD Band",
        options=list(imd_mapping.keys())
    )
    imd_band = imd_mapping[selected_imd_band]

    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"]
    )

    selected_region = st.selectbox(
        "Region",
        options=[
            "East Anglian Region",
            "East Midlands Region",
            "London Region",
            "North Region",
            "North Western Region",
            "Scotland",
            "South East Region",
            "South Region",
            "South West Region",
            "Wales",
            "West Midlands Region",
            "Yorkshire Region"
        ]
    )


# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict Dropout Risk"):

    try:
        # Pull exact training columns from model
        X_cols = model.feature_names_in_.tolist()
    except AttributeError:
        st.error("Model does not contain feature names.")
        st.stop()

    # Create blank dataframe with correct columns
    input_data = pd.DataFrame(
        np.zeros((1, len(X_cols))),
        columns=X_cols
    )

    # Numerical / encoded features
    base_features = {
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "sum_click": sum_click,
        "score": score,
        "disability": disability,
        "highest_education": highest_education,
        "age_band": age_band,
        "imd_band": imd_band
    }

    for feature, value in base_features.items():
        if feature in input_data.columns:
            input_data[feature] = value

    # One-hot gender
    gender_col = f"gender_{gender}"
    if gender_col in input_data.columns:
        input_data[gender_col] = 1

    # One-hot region
    region_col = f"region_{selected_region}"
    if region_col in input_data.columns:
        input_data[region_col] = 1

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ Student is likely to DROP OUT\n\nRisk Score: {probability:.2%}"
        )
    else:
        st.success(
            f"✅ Student is likely to COMPLETE\n\nSuccess Score: {(1 - probability):.2%}"
        )
