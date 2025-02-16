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
# ----- SALARY PREDICTION -----
elif page == "Salary Prediction":
    st.title("💰 Salary Prediction")
    st.write("Enter your details to estimate your expected salary.")

    # Load model & feature list
    salary_model = joblib.load("salary_model.pkl")
    feature_list = [
        'Programming_Experience_Midpoint', 'ML_Experience_Midpoint', 'Company_Size',
        'Age_Midpoint', 'Gender', 'Education', 'Country'
    ]

    # Dropdown Selections
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education Level", ["High School", "Some College", "Bachelor", "Master", "PhD", "Professional", "Professional Doctorate"])
    country = st.selectbox("Country", df["Country"].unique())

    # Sliders for numerical inputs
    prog_exp = st.slider("Programming Experience (Years)", 0, 20, 3)
    ml_exp = st.slider("ML Experience (Years)", 0, 10, 2)
    age = st.slider("Age", 18, 70, 30)
    company_size = st.selectbox("Company Size", ["0-49 employees", "50-249 employees", "250-999 employees", "1000-9,999 employees", "10,000 or more employees"])

    # Checkboxes for Programming Languages
    st.subheader("Programming Languages Used:")
    lang_options = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB', 'C#', 'PHP']
    selected_langs = {f'Language - {lang}': st.checkbox(lang, value=False) for lang in lang_options}

    # Checkboxes for IDEs
    st.subheader("Preferred IDEs:")
    ide_options = ['Jupyter Notebook', 'RStudio', 'VSCode', 'PyCharm', 'Spyder', 'Notepad++', 'MATLAB']
    selected_ides = {f'IDE - {ide}': st.checkbox(ide, value=False) for ide in ide_options}

    # Checkboxes for ML Frameworks
    st.subheader("ML Frameworks Used:")
    framework_options = ['Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Xgboost', 'LightGBM', 'CatBoost']
    selected_frameworks = {f'Framework - {fw}': st.checkbox(fw, value=False) for fw in framework_options}

    # Checkboxes for ML Algorithms
    st.subheader("ML Algorithms Used:")
    algo_options = ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Networks', 'Transformers']
    selected_algos = {f'Algorithm - {algo}': st.checkbox(algo, value=False) for algo in algo_options}

    # Checkboxes for Learning Platforms
    st.subheader("Learning Platforms Used:")
    platform_options = ['Coursera', 'edX', 'Kaggle Learn', 'DataCamp', 'Udacity', 'Udemy', 'LinkedIn Learning']
    selected_platforms = {f'Learning - {platform}': st.checkbox(platform, value=False) for platform in platform_options}

    # Mapping categorical inputs
    company_size_map = {
        "0-49 employees": 25, "50-249 employees": 150, "250-999 employees": 625,
        "1000-9,999 employees": 5500, "10,000 or more employees": 10000
    }
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    education_map = {
        "High School": "High School", "Some College": "Some College", "Bachelor": "Bachelor",
        "Master": "Master", "PhD": "PhD", "Professional": "Professional", "Professional Doctorate": "Professional Doctorate"
    }

    # Create user input dataframe
    user_input = pd.DataFrame({
        "Programming_Experience_Midpoint": [prog_exp],
        "ML_Experience_Midpoint": [ml_exp],
        "Company_Size": [company_size_map[company_size]],
        "Age_Midpoint": [age],
        "Gender": [gender],
        "Education": [education],
        "Country": [country]
    })

    # Convert categorical features to one-hot encoding
    user_input = pd.get_dummies(user_input, columns=["Education", "Country", "Gender"])

    # Merge with checkboxes inputs
    for key, value in {**selected_langs, **selected_ides, **selected_frameworks, **selected_algos, **selected_platforms}.items():
        user_input[key] = int(value)  # Convert True/False to 1/0

    # Ensure user input matches the trained model's features
    for col in salary_model.feature_names_in_:
        if col not in user_input.columns:
            user_input[col] = 0  # Add missing columns with default 0

    # Reorder columns to match model input
    user_input = user_input[salary_model.feature_names_in_]

    # Predict salary
    salary_prediction = salary_model.predict(user_input)

    # Display Prediction
    st.write(f"### 📌 Predicted Salary: **${salary_prediction[0]:,.2f}**")




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
