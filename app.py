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
    # st.title("📊 Exploratory Data Analysis")

    st.title("📊 Exploratory Data Analysis")
    st.markdown("## **Distribution of Role by Gender**")
    # Role_Title and Gender
    df_filtered = df[df["Gender"].isin(["Male", "Female"])]

    # Create the stacked bar chart
    fig = px.bar(
        df_filtered, 
        x="Role_Title", 
        color="Gender",
        title="Distribution of Role by Gender",
        labels={"Role_Title": "Role", "count": "Number of Respondents"},
        barmode="stack"
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Role",
        yaxis_title="Number of Respondents",
        xaxis_tickangle=-45
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Explanation text
    st.write("""
    Respondents whose gender was not Male or Female, and those with a role classified as "Other" 
    were excluded from this visualization. Generally, there were more male respondents compared to female respondents.
    """)
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

    st.subheader("📊 Proportional Distribution of Education Levels within Roles")

    # Filter necessary columns
    df_filtered_edu = df[["Role_Title", "Education"]]

    # Normalize count to get proportions
    df_edu_counts = df_filtered_edu.groupby(["Role_Title", "Education"]).size().reset_index(name="Count")
    df_edu_counts["Proportion"] = df_edu_counts.groupby("Role_Title")["Count"].transform(lambda x: x / x.sum())

    # Create the stacked bar chart
    fig_edu = px.bar(
        df_edu_counts, 
        x="Role_Title", 
        y="Proportion", 
        color="Education",
        title="Proportional Distribution of Education Levels within Roles",
        labels={"Role_Title": "Role", "Proportion": "Proportion", "Education": "Education Level"},
        barmode="stack"
    )

    # Update layout
    fig_edu.update_layout(
        xaxis_title="Role",
        yaxis_title="Proportion",
        xaxis_tickangle=-45
    )

    # Display in Streamlit
    st.plotly_chart(fig_edu, use_container_width=True)

    # Explanation text
    st.write("""
    Distribution of educational levels within roles revealed that generally most respondents had 
    **Bachelors or Masters** as the highest education level. The highest proportion of respondents 
    working on **data-related professions** (Data Analyst, Data Engineer, Data Scientists, Machine Learning Engineer, 
    and Statisticians) had **Masters** as their highest education level.  
    Majority of those working as **Research Scientists and Professors** had **PhD** as their highest education level.
    """)

# ----- SALARY PREDICTION -----
# Load trained model
salary_model = joblib.load("salary_model.pkl")
feature_list = salary_model.feature_names_in_

# Initialize session state variables for interactive responses
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.gender = None
    st.session_state.education = None
    st.session_state.country = None
    st.session_state.prog_exp = None
    st.session_state.ml_exp = None
    st.session_state.age = None
    st.session_state.company_size = None
    st.session_state.selected_langs = []
    st.session_state.selected_ides = []
    st.session_state.selected_frameworks = []
    st.session_state.selected_algos = []
    st.session_state.selected_platforms = []

# Mapping categorical inputs
company_size_map = {
    "0-49 employees": 25, "50-249 employees": 150, "250-999 employees": 625,
    "1000-9,999 employees": 5500, "10,000 or more employees": 10000
}
gender_map = {"Male": 1, "Female": 0, "Other": 2}

# Define selectable options
education_levels = ["High School", "Some College", "Bachelor", "Master", "PhD", "Professional", "Professional Doctorate"]
company_sizes = list(company_size_map.keys())

# Step-by-step questioning
if st.session_state.step == 1:
    st.write("I am...")
    st.session_state.gender = st.radio("Select your gender:", ["Male", "Female", "Other"], index=0)
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 2:
    st.write(f"I am a {st.session_state.gender} and I have a...")
    st.session_state.education = st.radio("Select your education level:", education_levels, index=2)
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 3:
    st.write(f"I am a {st.session_state.gender} with a {st.session_state.education} degree. I live in...")
    st.session_state.country = st.selectbox("Select your country:", df["Country"].unique())
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 4:
    st.write(f"I am a {st.session_state.gender} with a {st.session_state.education} degree living in {st.session_state.country}.")
    st.session_state.prog_exp = st.slider("How many years of programming experience do you have?", 0, 20, 3)
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 5:
    st.write(f"I have {st.session_state.prog_exp} years of programming experience.")
    st.session_state.ml_exp = st.slider("How many years of ML experience do you have?", 0, 10, 2)
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 6:
    st.write(f"I have {st.session_state.prog_exp} years of programming and {st.session_state.ml_exp} years of ML experience.")
    st.session_state.age = st.slider("What is your age?", 18, 70, 30)
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 7:
    st.write(f"I am {st.session_state.age} years old, with {st.session_state.prog_exp} years of programming and {st.session_state.ml_exp} years of ML experience.")
    st.session_state.company_size = st.selectbox("What is your company size?", company_sizes)
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 8:
    st.write("📌 Select the programming languages you use:")
    lang_options = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB', 'C#', 'PHP']
    st.session_state.selected_langs = [lang for lang in lang_options if st.checkbox(lang, value=False, key=f'lang_{lang}')]
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 9:
    st.write("🖥️ Select your preferred IDEs:")
    ide_options = ['Jupyter Notebook', 'RStudio', 'VSCode', 'PyCharm', 'Spyder', 'Notepad++', 'MATLAB']
    st.session_state.selected_ides = [ide for ide in ide_options if st.checkbox(ide, value=False, key=f'ide_{ide}')]
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 10:
    st.write("🤖 Select the ML frameworks you use:")
    framework_options = ['Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Xgboost', 'LightGBM', 'CatBoost']
    st.session_state.selected_frameworks = [fw for fw in framework_options if st.checkbox(fw, value=False, key=f'fw_{fw}')]
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 11:
    st.write("📊 Select the ML algorithms you use:")
    algo_options = ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Networks', 'Transformers']
    st.session_state.selected_algos = [algo for algo in algo_options if st.checkbox(algo, value=False, key=f'algo_{algo}')]
    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 12:
    st.write("🎓 Select the learning platforms you use:")
    platform_options = ['Coursera', 'edX', 'Kaggle Learn', 'DataCamp', 'Udacity', 'Udemy', 'LinkedIn Learning']
    st.session_state.selected_platforms = [platform for platform in platform_options if st.checkbox(platform, value=False, key=f'platform_{platform}')]
    if st.button("Predict Salary"):
        st.session_state.step += 1
        st.experimental_rerun()

elif st.session_state.step == 13:
    # Create user input dataframe
    user_input = pd.DataFrame({
        "Programming_Experience_Midpoint": [st.session_state.prog_exp],
        "ML_Experience_Midpoint": [st.session_state.ml_exp],
        "Company_Size": [company_size_map[st.session_state.company_size]],
        "Age_Midpoint": [st.session_state.age],
        "Gender": [st.session_state.gender],
        "Education": [st.session_state.education],
        "Country": [st.session_state.country]
    })

    # Convert categorical features to one-hot encoding
    user_input = pd.get_dummies(user_input, columns=["Education", "Country", "Gender"])

    # Merge checkboxes
    for key in st.session_state.selected_langs + st.session_state.selected_ides + st.session_state.selected_frameworks + st.session_state.selected_algos + st.session_state.selected_platforms:
        user_input[key] = 1

    # Ensure user input matches model's features
    for col in feature_list:
        if col not in user_input.columns:
            user_input[col] = 0

    user_input = user_input[feature_list]

    # Predict salary
    salary_prediction = salary_model.predict(user_input)
    
    # Display final result
    st.markdown(f"## 📌 Predicted Salary: **${salary_prediction[0]:,.2f}** 💰")

# ----- ROLE PREDICTION -----
elif page == "Role Prediction":
    st.title("🧑‍💻 Role Prediction")
    st.write("Find out which job best suits your skills!")

    # Load trained model & label encoder
    role_model = joblib.load("role_model.pkl")
    label_encoder = joblib.load("role_label_encoder.pkl")  # Load the label encoder
    role_feature_list = role_model.feature_names_in_

    # User Input Fields
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox(
        "Education Level",
        ["High School", "Some College", "Bachelor", "Master", "PhD", "Professional", "Professional Doctorate"]
    )
    country = st.selectbox("Country", df["Country"].unique())

    prog_exp = st.slider("Programming Experience (Years)", 0, 20, 3)
    ml_exp = st.slider("ML Experience (Years)", 0, 10, 2)
    age = st.slider("Age", 18, 70, 30)

    # Checkboxes for Programming Languages
    st.subheader("📌 Programming Languages Used:")
    lang_options = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB', 'C#', 'PHP']
    selected_langs = {f'Language - {lang}': st.checkbox(lang, value=False, key=f'lang_{lang}') for lang in lang_options}

    # Checkboxes for IDEs
    st.subheader("🖥️ Preferred IDEs:")
    ide_options = ['Jupyter Notebook', 'RStudio', 'VSCode', 'PyCharm', 'Spyder', 'Notepad++', 'MATLAB']
    selected_ides = {f'IDE - {ide}': st.checkbox(ide, value=False, key=f'ide_{ide}') for ide in ide_options}

    # Checkboxes for ML Frameworks
    st.subheader("🤖 ML Frameworks Used:")
    framework_options = ['Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Xgboost', 'LightGBM', 'CatBoost']
    selected_frameworks = {f'Framework - {fw}': st.checkbox(fw, value=False, key=f'fw_{fw}') for fw in framework_options}

    # Checkboxes for ML Algorithms
    st.subheader("📊 ML Algorithms Used:")
    algo_options = ['Linear or Logistic Regression', 'Decision Trees or Random Forests', 'Gradient Boosting Machines', 
                    'Bayesian Approaches', 'Neural Networks', 'Transformers']
    selected_algos = {f'Algorithm - {algo}': st.checkbox(algo, value=False, key=f'algo_{algo}') for algo in algo_options}

    # Checkboxes for Learning Platforms
    st.subheader("🎓 Learning Platforms Used:")
    platform_options = ['Coursera', 'edX', 'Kaggle Learn', 'DataCamp', 'Udacity', 'Udemy', 'LinkedIn Learning']
    selected_platforms = {f'Learning - {platform}': st.checkbox(platform, value=False, key=f'platform_{platform}') for platform in platform_options}

    # Mapping categorical inputs
    gender_map = {"Male": 1, "Female": 0, "Other": 2}

    # Create user input dataframe
    user_input = pd.DataFrame({
        "Programming_Experience_Midpoint": [prog_exp],
        "ML_Experience_Midpoint": [ml_exp],
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
    for col in role_feature_list:
        if col not in user_input.columns:
            user_input[col] = 0  # Add missing columns with default 0

    # Reorder columns to match model input
    user_input = user_input[role_feature_list]

    # **Prediction Button**
    if st.button("Predict My Role"):
        # Predict role
        role_prediction_encoded = role_model.predict(user_input)[0]
        
        # Convert prediction back to actual job title
        predicted_role = label_encoder.inverse_transform([role_prediction_encoded])[0]

        # Display Prediction
        st.markdown(f"## 🎯 Recommended Role: **{predicted_role}**")

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