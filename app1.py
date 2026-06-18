import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

with open('attrition_model (4).pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="👔")
st.title("👔 Employee Attrition Prediction System")
st.write("Fill in employee details to predict if they will leave.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", ["Sales", "HR", "IT", "Finance", "Marketing"])
    job_title = st.selectbox("Job Title", ["Manager", "Engineer", "Analyst", "Executive", "Director"])
    years = st.number_input("Years at Company", min_value=0, max_value=40, value=5)

with col2:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    avg_hours = st.number_input("Average Monthly Hours", min_value=0, max_value=400, value=160)
    promotion = st.selectbox("Promotion in Last 5 Years", [0, 1])
    salary = st.selectbox("Salary", ["low", "medium", "high"])

if st.button("🔍 Predict Attrition"):
    # Encode exactly same as training
    gender_enc = 1 if gender == "Male" else 0
    
    dept_map = {"Sales": 0, "HR": 1, "IT": 2, "Finance": 3, "Marketing": 4}
    dept_enc = dept_map.get(department, 0)
    
    job_map = {"Manager": 0, "Engineer": 1, "Analyst": 2, "Executive": 3, "Director": 4}
    job_enc = job_map.get(job_title, 0)
    
    salary_map = {"low": 1, "medium": 2, "high": 0}
    salary_enc = salary_map.get(salary, 1)

    features = np.array([[age, gender_enc, dept_enc, job_enc,
                          years, satisfaction, avg_hours,
                          promotion, salary_enc]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ Employee is likely to LEAVE the organization!")
    else:
        st.success("✅ Employee is likely to STAY in the organization!")