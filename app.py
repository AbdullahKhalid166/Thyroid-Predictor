import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ Load models
rf_model = joblib.load("models/thyroid_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

st.title("Thyroid Disease Predictor")

# ======================
# INPUTS
# ======================

age = st.number_input("Age", 0, 100)

sex = st.selectbox("Sex", ["M", "F", "Unknown"])

on_thyroxine = st.selectbox("On Thyroxine", [0, 1])
query_on_thyroxine = st.selectbox("Query Thyroxine", [0, 1])
on_antithyroid_meds = st.selectbox("Antithyroid Meds", [0, 1])
sick = st.selectbox("Sick", [0, 1])
pregnant = st.selectbox("Pregnant", [0, 1])
thyroid_surgery = st.selectbox("Thyroid Surgery", [0, 1])
I131_treatment = st.selectbox("I131 Treatment", [0, 1])
query_hypothyroid = st.selectbox("Query Hypothyroid", [0, 1])
query_hyperthyroid = st.selectbox("Query Hyperthyroid", [0, 1])
lithium = st.selectbox("Lithium", [0, 1])
goitre = st.selectbox("Goitre", [0, 1])
tumor = st.selectbox("Tumor", [0, 1])
hypopituitary = st.selectbox("Hypopituitary", [0, 1])
psych = st.selectbox("Psych", [0, 1])

TSH = st.number_input("TSH", value=0.0)
T3 = st.number_input("T3", value=0.0)
TT4 = st.number_input("TT4", value=0.0)
T4U = st.number_input("T4U", value=0.0)
FTI = st.number_input("FTI", value=0.0)

referral_source = st.text_input("Referral Source", "SVI")

# ======================
# PREDICTION BUTTON
# ======================

if st.button("Predict"):

    try:
        # Basic validation
        if age <= 0 or age > 120:
            st.error("Please fill input correctly. Age must be between 1 and 120.")
            st.stop()

        if referral_source.strip() == "":
            st.error("Please fill input correctly. Referral source cannot be empty.")
            st.stop()

        new_patient = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'on_thyroxine': on_thyroxine,
            'query_on_thyroxine': query_on_thyroxine,
            'on_antithyroid_meds': on_antithyroid_meds,
            'sick': sick,
            'pregnant': pregnant,
            'thyroid_surgery': thyroid_surgery,
            'I131_treatment': I131_treatment,
            'query_hypothyroid': query_hypothyroid,
            'query_hyperthyroid': query_hyperthyroid,
            'lithium': lithium,
            'goitre': goitre,
            'tumor': tumor,
            'hypopituitary': hypopituitary,
            'psych': psych,
            'TSH_measured': 1,
            'TSH': TSH,
            'T3_measured': 1,
            'T3': T3,
            'TT4_measured': 1,
            'TT4': TT4,
            'T4U_measured': 1,
            'T4U': T4U,
            'FTI_measured': 1,
            'FTI': FTI,
            'TBG_measured': 0,
            'TBG': np.nan,
            'referral_source': referral_source.strip().upper()
        }])

        rf_pred = rf_model.predict(new_patient)[0]
        rf_prob = rf_model.predict_proba(new_patient)[0][1]

        xgb_pred = xgb_model.predict(new_patient)[0]
        xgb_prob = xgb_model.predict_proba(new_patient)[0][1]

        st.subheader("Results")

        st.write("### Random Forest")
        st.write("Prediction:", "Disease" if rf_pred == 1 else "Normal")
        st.write(f"Probability: {rf_prob * 100:.2f}%")

        st.write("### XGBoost")
        st.write("Prediction:", "Disease" if xgb_pred == 1 else "Normal")
        st.write(f"Probability: {xgb_prob * 100:.2f}%")

    except Exception:
        st.error("Please fill input correctly.")