import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/salary_model.pkl")

# Page config
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0f172a;
        color: #ffffff;
    }

    .main > div {
        padding: 2rem;
        background: linear-gradient(to right, #1e293b, #0f172a);
        border-radius: 16px;
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.3);
    }

    h1, h4 {
        text-align: center;
    }

    .stButton > button {
        background-color: #10b981;
        color: white;
        border: none;
        padding: 0.6em 1.4em;
        font-size: 16px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #059669;
        transform: scale(1.05);
    }

    .salary-box {
        background-color: #dcfce7;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #22c55e;
        margin-top: 20px;
    }

    .salary-text {
        font-size: 32px;
        font-weight: bold;
        color: #14532d;
    }

    .form-title {
        color: #22c55e;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 16px;
    }

    label {
        font-weight: 500;
        color: #f3f4f6 !important;
    }

    input, div[data-baseweb="select"] {
        background-color: #1f2937 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Job titles
job_titles = sorted(list(set([
    "Software Engineer", "Data Analyst", "Senior Manager", "Sales Associate", "Director",
    "Marketing Analyst", "Product Manager", "Sales Manager", "Marketing Coordinator",
    "Senior Scientist", "Software Developer", "HR Manager", "Financial Analyst",
    "Project Manager", "Customer Service Rep", "Operations Manager", "Marketing Manager",
    "Senior Engineer", "Data Entry Clerk", "Sales Director", "Business Analyst",
    "VP of Operations", "IT Support", "Recruiter", "Financial Manager",
    "Social Media Specialist", "Software Manager", "Junior Developer", "Senior Consultant",
    "Product Designer", "CEO", "Accountant", "Data Scientist", "Marketing Specialist",
    "Technical Writer", "HR Generalist", "Project Engineer", "Customer Success Rep",
    "Sales Executive", "UX Designer", "Operations Director", "Network Engineer",
    "Administrative Assistant", "Strategy Consultant", "Copywriter", "Account Manager",
    "Director of Marketing", "Help Desk Analyst", "Customer Service Manager",
    "Business Intelligence Analyst", "Event Coordinator", "VP of Finance", "Graphic Designer",
    "UX Researcher", "Social Media Manager", "Senior Data Scientist", "Junior Accountant",
    "Digital Marketing Manager", "IT Manager", "Customer Service Representative",
    "Business Development Manager", "Senior Financial Analyst", "Web Developer",
    "Research Director", "Technical Support Specialist", "Creative Director",
    "Human Resources Director", "Content Marketing Manager", "Technical Recruiter",
    "Sales Representative", "Chief Technology Officer", "Junior Designer", "Financial Advisor",
    "Junior Account Manager", "Senior Project Manager", "Principal Scientist",
    "Supply Chain Manager", "Training Specialist", "Research Scientist",
    "Junior Software Developer", "Public Relations Manager", "Operations Analyst",
    "Product Marketing Manager", "Senior HR Manager", "Junior Web Developer",
    "Senior Project Coordinator", "Chief Data Officer", "Digital Content Producer",
    "IT Support Specialist", "Senior Marketing Analyst", "Customer Success Manager",
    "Senior Graphic Designer", "Software Project Manager", "Supply Chain Analyst",
    "Senior Business Analyst", "Junior Marketing Analyst", "Office Manager",
    "Principal Engineer", "Junior HR Generalist", "Senior Product Manager",
    "Junior Operations Analyst", "Junior Data Scientist", "Senior Operations Analyst",
    "Senior Human Resources Coordinator", "Senior UX Designer", "Junior Product Manager",
    "Senior IT Project Manager", "Senior Quality Assurance Analyst",
    "Director of Sales and Marketing", "Senior Account Executive", "Director of Business Development",
    "Junior Social Media Manager", "Senior Human Resources Specialist",
    "Senior Data Analyst", "Senior Software Developer", "Director of Human Capital",
    "Junior Advertising Coordinator", "Junior UX Designer", "Senior Marketing Director",
    "Senior HR Generalist", "Sales Operations Manager", "Junior Web Designer",
    "Senior Training Specialist", "Senior Research Scientist", "Junior Sales Representative",
    "Junior Marketing Manager", "Junior Data Analyst", "Senior Product Marketing Manager",
    "Junior Business Analyst", "Senior Sales Manager", "Junior Marketing Specialist",
    "Senior Accountant", "Director of Sales", "Junior Recruiter",
    "Senior Business Development Manager", "Junior Customer Support Specialist",
    "Senior IT Support Specialist", "Junior Financial Analyst", "Senior Operations Manager",
    "Director of Human Resources", "Junior Software Engineer", "Senior Sales Representative",
    "Director of Product Management", "Junior Copywriter", "Senior Human Resources Manager",
    "Junior Business Development Associate", "Senior Account Manager", "Senior Researcher",
    "Junior HR Coordinator", "Senior Financial Advisor", "Senior Project Coordinator",
    "Junior Business Operations Analyst", "Senior Financial Manager", "Senior Data Engineer",
    "Senior IT Consultant", "Senior Product Designer", "Junior Financial Advisor",
    "Director of Engineering", "Junior Operations Manager", "Senior Operations Coordinator",
    "Director of HR", "Junior Project Manager"
])))

education_levels = ["Bachelor's", "Master's", "PhD"]

# UI Title
st.markdown("<h1>💼 Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4>Get your salary estimation instantly</h4>", unsafe_allow_html=True)
st.markdown("---")

# Form
with st.form(key="salary_prediction_form"):
    st.markdown("<div class='form-title'>👤 Personal & Job Information</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age", min_value=18, max_value=70, value=18)
        gender = st.selectbox("🧑 Gender", ["Male", "Female", "Other"])
        education = st.selectbox("🎓 Education Level", education_levels)

    with col2:
        job_title = st.selectbox("💼 Job Title", job_titles)
        experience = st.number_input("📈 Years of Experience", min_value=0, max_value=50, value=0)

    submitted = st.form_submit_button("🔍 Predict Salary")

# Prediction Logic
if submitted:
    exp_per_age = experience / age if age > 0 else 0
    seniority = int(any(x in job_title for x in ["Senior", "Manager", "Director", "Lead", "Chief", "CEO"]))

    user_input = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience,
        "Experience_per_Age": exp_per_age,
        "Seniority": seniority
    }])

    try:
        pred_log = model.predict(user_input)
        predicted_salary = np.expm1(pred_log)[0]

        st.markdown(f"""
            <div class="salary-box">
                <div>💰 <span class="salary-text">Estimated Salary: ${predicted_salary:,.2f}</span></div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Prediction failed due to an internal error.")
        st.exception(e)

    with st.expander("📘 How this works"):
        st.markdown("""
        - Model: **XGBoost Regressor**
        - Features used:
            - Age, Gender, Education Level, Job Title
            - Experience-to-age ratio
            - Seniority based on title
        - Output is predicted using **logarithmic transformation**
        """)
