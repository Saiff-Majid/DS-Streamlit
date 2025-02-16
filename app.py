import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models
salary_model = joblib.load("salary_model.pkl")  
role_model = joblib.load("role_model.pkl")      

# Load preprocessed dataset
df = pd.read_csv("df_cleaned.csv")  

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", 
    ["Introduction", "Data Preprocessing & Cleaning", "Exploratory Data Analysis", "Salary Prediction", "Role Prediction", "Conclusion"]
)

# ----- INTRODUCTION -----
if page == "Introduction":
    st.image("https://datascientest.com/wp-content/uploads/2022/03/logo-2021.png")
    st.title("🔍 Data Science Job Market Analysis & Prediction")
    st.markdown("""
    ## **Project Overview**
    - Analyzes **Kaggle Machine Learning & Data Science Surveys (2020-2022)**.
    - Explores **salaries, job roles, skills, and tools** used in Data Science.
    - **Two ML models**:
      - 📈 **Salary Prediction** (Regression)
      - 🎯 **Role Prediction** (Classification)
    - **Goal:** Help professionals understand job market trends and expected salaries.
    """)



# ----- DATA PREPROCESSING -----
elif page == "Data Preprocessing & Cleaning":
    st.title("🛠️ Data Preprocessing & Cleaning")
    
    st.markdown("""
    ## **Why Clean Data?**
    - 🗑️ **Remove missing values**.
    - 🔄 **Standardize categories** (e.g., role names).
    - 🔍 **Feature engineering** to make data useful for ML models.

    ## **Steps in Preprocessing**
    1️⃣ **Merged Kaggle Survey Data (2020-2022)**.  
    2️⃣ **Identified & cleaned missing values**.  
    3️⃣ **Mapped job titles and education levels**.  
    4️⃣ **Extracted salary ranges and calculated averages**.  
    """)

    st.subheader("Sample Merged Data")
    st.dataframe(df.head())

    st.write("### Data Distribution by Year")
    year_counts = df["Year"].value_counts()
    fig_year = px.bar(year_counts, x=year_counts.index, y=year_counts.values, labels={"x": "Year", "y": "Count"}, title="Survey Responses per Year")
    st.plotly_chart(fig_year)

# ----- EXPLORATORY DATA ANALYSIS -----
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    st.markdown("## **Job Role Distribution**")
    role_counts = df["Role_Title"].value_counts()
    fig_role = px.bar(role_counts, x=role_counts.index, y=role_counts.values, title="Distribution of Roles")
    st.plotly_chart(fig_role)

    st.markdown("## **Salary Distribution**")
    fig_salary = px.histogram(df, x="Average_Salary", nbins=50, title="Distribution of Salaries")
    st.plotly_chart(fig_salary)

    st.markdown("## **Programming Experience vs Salary**")
    fig_exp = px.scatter(df, x="Programming_Experience_Midpoint", y="Average_Salary", color="Role_Title", title="Experience vs Salary")
    st.plotly_chart(fig_exp)

    st.markdown("## **Salary by Education Level**")
    avg_salary_by_education = df.groupby('Education')['Average_Salary'].mean().reset_index()
    fig_edu = px.bar(avg_salary_by_education, x="Education", y="Average_Salary", title="Average Salary by Education")
    st.plotly_chart(fig_edu)

# ----- SALARY PREDICTION -----
elif page == "Salary Prediction":
    st.title("💰 Salary Prediction")
    st.write("Enter your details to estimate your expected salary based on Kaggle Survey Data.")

    # User Inputs
    prog_exp = st.slider("Programming Experience (Years)", 0, 20, 3)
    ml_exp = st.slider("ML Experience (Years)", 0, 10, 2)
    age = st.slider("Age (Years)", 18, 70, 30)

    # Company Size Options (Mapped to Midpoints)
    company_size_dict = {
        "Small (0-49 employees)": 25,
        "Medium (50-249 employees)": 150,
        "Large (250-999 employees)": 625,
        "Enterprise (1000-9,999 employees)": 5500,
        "Mega Enterprise (10,000+ employees)": 10000
    }
    company_size = st.selectbox("Company Size", list(company_size_dict.keys()))

    # Education Level Options
    education_levels = [
        "High School", "Some College", "Bachelor", "Master", "PhD",
        "Professional", "Professional Doctorate", "Unknown"
    ]
    education = st.selectbox("Education Level", education_levels)

    # Gender Options
    gender_options = ["Male", "Female", "Other"]
    gender = st.selectbox("Gender", gender_options)

    # Country Selection
    countries = [
        "Argentina", "Australia", "Austria", "Bangladesh", "Belarus", "Belgium",
        "Brazil", "Cameroon", "Canada", "Chile", "China", "Colombia",
        "Czech Republic", "Denmark", "Ecuador", "Egypt", "Ethiopia", "France",
        "Germany", "Ghana", "Greece", "Hong Kong (S.A.R.)", "India", "Indonesia",
        "Iran, Islamic Republic of...", "Iraq", "Ireland", "Israel", "Italy",
        "Japan", "Kazakhstan", "Kenya", "Malaysia", "Mexico", "Morocco",
        "Nepal", "Netherlands", "Nigeria", "Norway", "Pakistan", "Peru",
        "Philippines", "Poland", "Portugal", "Republic of Korea", "Romania",
        "Russia", "Saudi Arabia", "Singapore", "South Africa", "South Korea",
        "Spain", "Sri Lanka", "Sweden", "Switzerland", "Taiwan", "Thailand",
        "Tunisia", "Turkey", "Uganda", "Ukraine", "United Arab Emirates",
        "United Kingdom of Great Britain and Northern Ireland",
        "United States of America", "Viet Nam", "Zimbabwe"
    ]
    country = st.selectbox("Country", countries)

    # Convert Inputs to DataFrame Matching Model Features
    user_input = pd.DataFrame(columns=salary_model.feature_names_in_)

    # Set all feature columns to 0 (default)
    user_input.loc[0] = np.zeros(len(salary_model.feature_names_in_))

    # Assign user input values to the correct feature columns
    user_input["Programming_Experience_Midpoint"] = prog_exp
    user_input["ML_Experience_Midpoint"] = ml_exp
    user_input["Age_Midpoint"] = age
    user_input["Company_Size"] = company_size_dict[company_size]

    # One-Hot Encoding for Categorical Features
    if f"Country_{country}" in user_input.columns:
        user_input[f"Country_{country}"] = 1

    if f"Education_{education}" in user_input.columns:
        user_input[f"Education_{education}"] = 1

    if f"Gender_{gender}" in user_input.columns:
        user_input[f"Gender_{gender}"] = 1

    # Make prediction
    if st.button("Predict Salary"):
        salary_prediction = salary_model.predict(user_input)
        st.subheader(f"💰 Estimated Salary: **${salary_prediction[0]:,.2f} USD**")



# ----- ROLE PREDICTION -----
elif page == "Role Prediction":
    st.title("🧑‍💻 Role Prediction")
    st.write("Find out which job best suits your skills!")

    age = st.slider("Age", 18, 60, 25)
    prog_exp = st.slider("Programming Experience (Years)", 0, 20, 3)
    ml_exp = st.slider("ML Experience (Years)", 0, 10, 2)
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    language = st.selectbox("Primary Programming Language", ["Python", "R", "SQL", "Java"])

    input_data_role = np.array([[age, prog_exp, ml_exp, company_size_map[company_size], language]])

    role_prediction = role_model.predict(input_data_role)
    st.write(f"### 🎯 Recommended Role: **{role_prediction[0]}**")

# ----- CONCLUSION -----
elif page == "Conclusion":
    st.title("📌 Conclusion & Future Work")
    st.markdown("""
    - **Key Findings**:
      - Salary is **highly influenced** by experience, company size, and country.
      - Role prediction is **more complex** due to overlapping skills.
    - **Future Improvements**:
      - Improve **role classification** by refining job definitions.
      - Expand dataset for better accuracy.
    """)
    st.success("Thank you for exploring our project!")
