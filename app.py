# app.py (Streamlit version)

import streamlit as st
import pandas as pd
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("rich_symptom_disease_data.csv")
symptom_list = df.columns.drop("disease").tolist()

st.set_page_config(page_title="Medical Symptom Checker", layout="centered")

st.title(" Medical Symptom Checker")
st.markdown("Select your symptoms from the list below:")


selected_symptoms = st.multiselect("Symptoms", options=symptom_list)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Encode symptoms
        input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        prediction = model.predict([input_data])[0]
        st.success(f" Predicted Disease: **{prediction}**")
