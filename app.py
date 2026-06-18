import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('attrition_model_new.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
feature_names = model_package['feature_names']

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
    gender_enc = 1 if gender == "Male" else 0
    dept_map = {"Sales": 4, "HR": 1, "IT": 2, "Finance": 0, "Marketing": 3}
    job_map = {"Manager": 2, "Engineer": 1, "Analyst": 0, "Executive": 3, "Director": 4}
    salary_map = {"low": 1, "medium": 2, "high": 0}

    input_dict = {
        'Age': age,
        'Gender': gender_enc,
        'Department': dept_map.get(department, 0),
        'Job_Title': job_map.get(job_title, 0),
        'Years_at_Company': years,
        'Satisfaction_Level': satisfaction,
        'Average_Monthly_Hours': avg_hours,
        'Promotion_Last_5Years': promotion,
        'Salary': salary_map.get(salary, 1)
    }

    input_df = pd.DataFrame([input_dict])[feature_names]
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Employee is likely to LEAVE the organization!")
    else:
        st.success("✅ Employee is likely to STAY in the organization!")