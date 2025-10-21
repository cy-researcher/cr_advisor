import streamlit as st
import pandas as pd
import joblib
import os

# If model doesn’t exist, train it
if not os.path.exists("career_model.pkl"):
    os.system("python train_model.py")

# Load model and encoders
model = joblib.load("career_model.pkl")
mlb_skills = joblib.load("mlb_skills.pkl")
mlb_interests = joblib.load("mlb_interests.pkl")
edu_encoder = joblib.load("edu_encoder.pkl")

st.set_page_config(page_title="AI Career Advisor", page_icon="🎓")

st.title("🎓 AI Career Advisor")
st.markdown("Find your ideal **AI/Tech career path** based on your skills, interests, and education!")

# --- Input Section ---
skills_input = st.multiselect(
    "Select your technical skills:",
    mlb_skills.classes_
)

interests_input = st.multiselect(
    "Select your personal interests:",
    mlb_interests.classes_
)

education_input = st.selectbox(
    "Select your education level:",
    edu_encoder.classes_
)

# --- Predict Button ---
if st.button("🔍 Suggest Career"):
    if not skills_input or not interests_input:
        st.warning("Please select at least one skill and one interest.")
    else:
        # Prepare input features
        skills_vec = mlb_skills.transform([skills_input])
        interests_vec = mlb_interests.transform([interests_input])
        edu_vec = edu_encoder.transform([education_input])

        import numpy as np
        X_input = pd.concat([
            pd.DataFrame(skills_vec, columns=mlb_skills.classes_),
            pd.DataFrame(interests_vec, columns=mlb_interests.classes_),
            pd.DataFrame(edu_vec, columns=["education_level"])
        ], axis=1)

        prediction = model.predict(X_input)[0]
        st.success(f"💡 Recommended Career: **{prediction}**")

        st.markdown("""
        ---
        ✅ *Tip:* Add more skills or interests for better personalization.
        """)

st.markdown("---")
st.caption("Powered by AI + Human Career Intelligence 🚀")
