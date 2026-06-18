import streamlit as st
import pickle
import numpy as np

with open('../MODEL/attrition_model (4).pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Employee Attrition Prediction", 
                   page_icon="👔")
st.title("👔 Employee Attrition Prediction System")
st.write("Fill in employee details to predict if they will leave.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
    job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
    years_at_company = st.number_input("Years at Company", min_value=0, value=5)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    distance = st.number_input("Distance From Home (km)", min_value=0, value=10)

with col2:
    environment = st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])
    work_life = st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4])
    job_level = st.selectbox("Job Level (1-5)", [1, 2, 3, 4, 5])
    num_companies = st.number_input("Number of Companies Worked", min_value=0, value=2)
    training = st.number_input("Training Times Last Year", min_value=0, value=2)
    performance = st.selectbox("Performance Rating (1-4)", [1, 2, 3, 4])

if st.button("🔍 Predict Attrition"):
    ot = 1 if overtime == "Yes" else 0

    features = np.array([[age, monthly_income, job_satisfaction,
                          years_at_company, ot, distance,
                          environment, work_life, job_level,
                          num_companies, training, performance]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error("⚠️ Employee is likely to LEAVE the organization!")
    else:
        st.success("✅ Employee is likely to STAY in the organization!")
