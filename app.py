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
    ["Introduction", "Data Preprocessing & Cleaning", "Exploratory Data Analysis", "Salary Prediction Analysis", "Salary Prediction", "Role Prediction Analysis", "Role Prediction", "Conclusion"]
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
    - 🔍 **Feature engineering** to make data useful for ML models.""")

    st.write("""The first step of the project was to figure out how to combine the three years of surveys. 
             Not all the questions were identical, so we had to find a solution in python to compile them.""")
    
    code_1="""def extract_base_question(question):
        return question.split('-')[0].strip()

    # Function to get unique questions in the same order as they appear in the dataset
    def get_unique_ordered_questions(columns):
        seen = set()
        unique_questions = []
        for col in columns:
            base_question = extract_base_question(col)
            if base_question not in seen:
                seen.add(base_question)
                unique_questions.append(base_question)
        return unique_questions


    # Extract unique base questions in the same order for each dataset
    unique_questions_2020 = get_unique_ordered_questions(df_2020.columns)
    unique_questions_2021 = get_unique_ordered_questions(df_2021.columns)
    unique_questions_2022 = get_unique_ordered_questions(df_2022.columns)

    # Print the number of unique base questions for each dataset
    print(f"Number of unique base questions in 2020 dataset: {len(unique_questions_2020)}")
    print(f"Number of unique base questions in 2021 dataset: {len(unique_questions_2021)}")
    print(f"Number of unique base questions in 2022 dataset: {len(unique_questions_2022)}")

    # Print the unique base questions with a count for each dataset
    def print_questions_with_count(questions, dataset_year):
        print(f"\nUnique base questions for {dataset_year} dataset:")
        for idx, question in enumerate(questions, start=1):
            print(f"Q{idx}: {question}")

    print_questions_with_count(unique_questions_2020, "2020")
    print_questions_with_count(unique_questions_2021, "2021")
    print_questions_with_count(unique_questions_2022, "2022")""" 

    st.code(code_1, language="python")
                
    st.write("""Our code to create a set of unique questions for each year.""")

    code_2= """#Convvert unique questions into sets
    unique_set_2020 = set(unique_questions_2020)
    unique_set_2021 = set(unique_questions_2021)
    unique_set_2022 = set(unique_questions_2022)

    # Find the intersection of the three sets to get common questions across all datasets
    common_questions_all = unique_set_2022.intersection(unique_set_2021, unique_set_2020)

    # Output the number of common questions and list them
    print(f"Number of questions common across all three datasets: {len(common_questions_all)}")"""

    st.code(code_2, language="python")

    st.write("Once combined we had a DataFrame like this.")
    
    image_1=Image.open(".DS-Streamlit/Screenshot.png") 
    st.image(image_1,caption="DF_combined.head()")
    


    st.markdown("""
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


# ----- SALARY PREDICTION ANALYSIS -----
elif page == "Salary Prediction Analysis":
    st.title("📊 Salary Prediction Analysis")
    st.markdown("""
    ## **Overview**
    - The goal was to **predict salaries** based on demographic and technical features.
    - Used **Machine Learning models** like **XGBoost, Random Forest, and LightGBM**.
    - The best model achieved **high accuracy (R² = 0.997)**.
    
    ## **Data Preparation & Preprocessing**
    - **Dropped unnecessary columns** like categorical groupings.
    - **Encoded categorical features** (One-hot encoding for Education, Country, Gender).
    - **Standardized numerical values** (Programming Experience, ML Experience, Company Size, Age).
    
    ## **Model Selection & Tuning**
    - **Tried multiple models**: Linear Regression, LightGBM, Random Forest, XGBoost.
    - **XGBoost performed best** after **hyperparameter tuning**.
    - Used **feature importance analysis** to identify key predictors:
      - **Country, Company Size, ML Frameworks (JAX), and Programming Experience** were the most significant.
    
    ## **Results**
    - The **final model** explained **99.7% of salary variance (R² = 0.997)**.
    - **Feature Importance:**
    """)
    
    # Feature Importance Visualization

    salary_feature_importance = pd.DataFrame({
        "Feature": salary_model.feature_names_in_,
        "Importance": salary_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_salary_importance = px.bar(salary_feature_importance, x="Feature", y="Importance", title="Feature Importance for Salary Prediction")
    st.plotly_chart(fig_salary_importance)
    
    st.markdown("""
    ## **Conclusion**
    - **Country & Company Size** play a huge role in salary.
    - **Programming & ML experience** significantly affect earnings.
    - **US salaries** are more stable in predictions because the dataset has more samples.
    - **Lack of enough data** from some countries makes the model generalize slightly, sometimes overestimating salaries.
    """)


# ----- SALARY PREDICTION -----

elif page == "Salary Prediction":
    st.title("💰 Salary Prediction")
    st.write("Enter your details to estimate your expected salary.")

    # Load model & feature list
    salary_model = joblib.load("salary_model.pkl")
    feature_list = salary_model.feature_names_in_

    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox(
            "Education Level",
            ["High School", "Some College", "Bachelor", "Master", "PhD", "Professional", "Professional Doctorate"]
        )
        country = st.selectbox("Country", df["Country"].unique())
        company_size = st.selectbox(
            "Company Size",
            ["0-49 employees", "50-249 employees", "250-999 employees", "1000-9,999 employees", "10,000 or more employees"]
        )
        prog_exp = st.slider("Programming Experience (Years)", 0, 20, 3)
        ml_exp = st.slider("ML Experience (Years)", 0, 10, 2)
        age = st.slider("Age", 18, 70, 30)

        st.subheader("🤖 ML Frameworks Used:")
        framework_options = ['Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Xgboost', 'LightGBM', 'CatBoost']
        selected_frameworks = {f'Framework - {fw}': st.checkbox(fw, key=f'fw_{fw}') for fw in framework_options}
        
        st.subheader("📊 ML Algorithms Used:")
        algo_options = ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Networks', 'Transformers']
        selected_algos = {f'Algorithm - {algo}': st.checkbox(algo, key=f'algo_{algo}') for algo in algo_options}
    
    with col2:
        st.subheader("📌 Programming Languages Used:")
        lang_options = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB', 'C#', 'PHP']
        selected_langs = {f'Language - {lang}': st.checkbox(lang, key=f'lang_{lang}') for lang in lang_options}
        
        st.subheader("🖥️ Preferred IDEs:")
        ide_options = ['Jupyter Notebook', 'RStudio', 'VSCode', 'PyCharm', 'Spyder', 'Notepad++', 'MATLAB']
        selected_ides = {f'IDE - {ide}': st.checkbox(ide, key=f'ide_{ide}') for ide in ide_options}
        
        st.subheader("🎓 Learning Platforms Used:")
        platform_options = ['Coursera', 'edX', 'Kaggle Learn', 'DataCamp', 'Udacity', 'Udemy', 'LinkedIn Learning']
        selected_platforms = {f'Learning - {platform}': st.checkbox(platform, key=f'platform_{platform}') for platform in platform_options}
    
    if st.button("Predict"):
        # Mapping categorical inputs
        company_size_map = {
            "0-49 employees": 25, "50-249 employees": 150, "250-999 employees": 625,
            "1000-9,999 employees": 5500, "10,000 or more employees": 10000
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
            user_input[key] = int(value)
        
        # Ensure user input matches the trained model's features
        for col in feature_list:
            if col not in user_input.columns:
                user_input[col] = 0  
        
        # Reorder columns to match model input
        user_input = user_input[feature_list]

        # Predict salary
        salary_prediction = float(salary_model.predict(user_input)[0])

        # Display Prediction
        st.markdown(f"## 📌 Predicted Salary: **${salary_prediction:,.2f}** 💰")
        
        
        # Compute comparison to average salary for similar age group in the same country
        age_category = "18-24" if age < 25 else "25-34" if age < 35 else "35-44" if age < 45 else "45-54" if age < 55 else "55+"
        
        df_filtered = df[(df["Country"] == country) & (df["Age_Midpoint"].between(age - 3, age + 3))]
        
        if not df_filtered.empty:
            avg_salary_same_group = df_filtered["Average_Salary"].mean()
            salary_diff_age_group = ((salary_prediction - avg_salary_same_group) / avg_salary_same_group) * 100

            
            salary_diff_age_group = float(salary_diff_age_group)

            salary_diff_text = "more" if salary_prediction > avg_salary_same_group else "less"
            salary_diff_statement = f"Your predicted salary is <b><u>{salary_diff_text}</u></b> by <b><u>{abs(salary_diff_age_group):.2f}%</u></b> compared to participants in your country within the same age category <u><b>({age_category})</u></b>."
        else:
            salary_diff_statement = "There is not enough data to compare your salary prediction with others in your country and age group."
            

        # Generate summary paragraph
        summary = f"<div style='font-size:22px'>You are a <b><u>{gender}</u></b> with a <b><u>{education}</u></b> degree from <b><u>{country}</u></b>. "
        summary += f"You have <b><u>{prog_exp}</u></b> years of programming experience and <b><u>{ml_exp}</u></b> years of ML experience. "
        summary += f"You work in a company with <b><u>{company_size}</u></b>. "
        summary += f"You are <b><u>{age}</u></b> years old. "
        
        for category, selections in zip(
            ["Languages", "IDEs", "Frameworks", "Algorithms", "Learning Platforms"],
            [selected_langs, selected_ides, selected_frameworks, selected_algos, selected_platforms]
        ):
            used_items = [key.split(' - ')[1] for key, val in selections.items() if val]
            if used_items:
                summary += f"You use {category.lower()} like: <b><u>{', '.join(used_items)}</u></b>. "
        
        summary += f"<br><br>Your predicted salary is <b><u>${salary_prediction:,.2f}</u></b>. {salary_diff_statement}"
        
        summary += "</div>"
        st.markdown(summary, unsafe_allow_html=True)

        # visualization for salary comparision

        if not df_filtered.empty:
            
            salary_comparison_df = pd.DataFrame({
                "Category": ["Average Salary (Same Age & Country)", "Predicted Salary"],
                "Salary (USD)": [avg_salary_same_group, salary_prediction]
            })

            
            fig_salary_comparison = px.bar(
                salary_comparison_df,
                x="Category",
                y="Salary (USD)",
                title=f"Salary Comparison: Predicted vs Average ({age_category}) in {country}",
                color="Category",
                text_auto=True  
            )

            
            fig_salary_comparison.update_layout(
                yaxis_title="Salary (USD)",
                xaxis_title="Category",
                bargap=0.4,
                template="plotly_white"
            )

            
            st.plotly_chart(fig_salary_comparison)

# ----- ROLE PREDICTION ANALYSIS -----
elif page == "Role Prediction Analysis":
    st.title("🧑‍💻 Role Prediction Analysis")
    st.markdown("""
    ## **Overview**
    - The goal was to predict **job roles** based on skills, experience, and demographics.
    - **More challenging than salary prediction** due to high overlap between job roles.
    
    ## **Data Preprocessing & Feature Engineering**
    - **One-hot encoded categorical variables** (Education, Country, Gender).
    - **Converted programming & ML experience** to numerical midpoints.
    - **Company size was mapped to numerical values**.
    
    ## **Model Selection & Performance**
    - Models used: **Logistic Regression, Random Forest, XGBoost, SVM**.
    - **XGBoost performed best** with **48.16% accuracy**.
    - **Challenges:** Overlapping job roles, imbalanced data.
    
    ## **Feature Importance**
    - **Education (PhD)** was the strongest predictor, indicating advanced degrees have a clear impact on role classification.
    - **Programming Languages (Python, R, Java, JavaScript)** were also highly influential, showing that language proficiency strongly correlates with specific roles.
    - **ML Algorithms (CNNs, Gradient Boosting, Transformers)** had a notable impact, emphasizing the significance of specialized ML skills.
    - **Company size & country had a smaller impact than expected**, meaning role assignments were less dependent on organization size compared to salary predictions.
    """)
    
    # Feature Importance Visualization
    role_feature_importance = pd.DataFrame({
        "Feature": role_model.feature_names_in_,
        "Importance": role_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_role_importance = px.bar(role_feature_importance, x="Feature", y="Importance", title="Feature Importance for Role Prediction")
    st.plotly_chart(fig_role_importance)
    
    st.markdown("""
    ## **Conclusion**
    - **Education level (especially PhD) is the most significant factor** in role prediction.
    - **Programming languages & ML frameworks play a major role** in differentiating job titles.
    - **Company size & country influence role assignment but are not the strongest determinants**.
    - Improving predictions would require **better-defined role classifications** and additional industry-specific variables.
    """)


# ----- ROLE PREDICTION -----
elif page == "Role Prediction":
    st.title("🧑‍💻 Role Prediction")
    st.write("Find out which job best suits your skills!")

    # Load trained model & label encoder
    role_model = joblib.load("role_model.pkl")
    label_encoder = joblib.load("role_label_encoder.pkl")
    role_feature_list = role_model.feature_names_in_

    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox(
            "Education Level",
            ["High School", "Some College", "Bachelor", "Master", "PhD", "Professional", "Professional Doctorate"]
        )
        country = st.selectbox("Country", df["Country"].unique())
        prog_exp = st.slider("Programming Experience (Years)", 0, 20, 3)
        ml_exp = st.slider("ML Experience (Years)", 0, 10, 2)
        age = st.slider("Age", 18, 70, 30)

        st.subheader("🖥️ Preferred IDEs:")
        ide_options = ['Jupyter Notebook', 'RStudio', 'VSCode', 'PyCharm', 'Spyder', 'Notepad++', 'MATLAB']
        selected_ides = {f'IDE - {ide}': st.checkbox(ide, key=f'ide_{ide}') for ide in ide_options}

        st.subheader("🤖 ML Frameworks Used:")
        framework_options = ['Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'Xgboost', 'LightGBM', 'CatBoost']
        selected_frameworks = {f'Framework - {fw}': st.checkbox(fw, key=f'fw_{fw}') for fw in framework_options}

    with col2:
        st.subheader("📌 Programming Languages Used:")
        lang_options = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB', 'C#', 'PHP']
        selected_langs = {f'Language - {lang}': st.checkbox(lang, key=f'lang_{lang}') for lang in lang_options}

        st.subheader("📊 ML Algorithms Used:")
        algo_options = ['Linear or Logistic Regression', 'Decision Trees or Random Forests', 'Gradient Boosting Machines', 'Bayesian Approaches', 'Neural Networks', 'Transformers']
        selected_algos = {f'Algorithm - {algo}': st.checkbox(algo, key=f'algo_{algo}') for algo in algo_options}

        st.subheader("🎓 Learning Platforms Used:")
        platform_options = ['Coursera', 'edX', 'Kaggle Learn', 'DataCamp', 'Udacity', 'Udemy', 'LinkedIn Learning']
        selected_platforms = {f'Learning - {platform}': st.checkbox(platform, key=f'platform_{platform}') for platform in platform_options}
    
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
        user_input[key] = int(value)

    # Ensure user input matches the trained model's features
    for col in role_feature_list:
        if col not in user_input.columns:
            user_input[col] = 0  

    # Reorder columns to match model input
    user_input = user_input[role_feature_list]

    # role percentage dictionary for summary
    role_percentages = {
        "Student": 20.62,
        "Data Scientist": 15.81,
        "Software Engineer": 10.21,
        "Data Analyst": 9.49,
        "Unemployed": 8.73,
        "Other": 8.05,
        "Research Scientist": 6.12,
        "Machine Learning Engineer": 5.95,
        "Business Analyst": 2.90,
        "Data Engineer": 2.76,
        "Project Manager": 2.68,
        "Teacher/Professor": 1.50,
        "Manager": 1.46,
        "Statistician": 1.29,
        "Engineer": 0.76,
        "Product Manager": 0.54,
        "Database Engineer": 0.54,
        "Developer Advocate": 0.30,
        "Data Architect": 0.19,
        "Data Administrator": 0.11
    }

    if st.button("Predict My Role"):
        role_prediction_encoded = role_model.predict(user_input)[0]
        predicted_role = label_encoder.inverse_transform([role_prediction_encoded])[0]
        st.markdown(f"## 🎯 Recommended Role: **{predicted_role}**")

         # Calculate the percentage of respondents with this role
        predicted_role_percentage = role_percentages.get(predicted_role, 0)
        
        # Generate summary paragraph
        summary = f"<div style='font-size:22px'>You are a <b><u>{gender}</u></b> with a <b><u>{education}</u></b> degree from <b><u>{country}</u></b>. "
        summary += f"You have <b><u>{prog_exp}</u></b> years of programming experience and <b><u>{ml_exp}</u></b> years of ML experience. "
        summary += f"You are <b><u>{age}</u></b> years old. "
        
        if any(selected_langs.values()):
            summary += f"You use: <b><u>{', '.join([key.split(' - ')[1] for key, val in selected_langs.items() if val])}</u></b>. "
        if any(selected_ides.values()):
            summary += f"Preferred IDEs: <b><u>{', '.join([key.split(' - ')[1] for key, val in selected_ides.items() if val])}</u></b>. "
        if any(selected_frameworks.values()):
            summary += f"ML Frameworks: <b><u>{', '.join([key.split(' - ')[1] for key, val in selected_frameworks.items() if val])}</u></b>. "
        if any(selected_algos.values()):
            summary += f"ML Algorithms: <b><u>{', '.join([key.split(' - ')[1] for key, val in selected_algos.items() if val])}</u></b>. "
        if any(selected_platforms.values()):
            summary += f"Learning Platforms: <b><u>{', '.join([key.split(' - ')[1] for key, val in selected_platforms.items() if val])}</u></b>. "
        summary += f"<br><br>Your predicted role is **{predicted_role}**, which you share with <b><u>{predicted_role_percentage:.2f}%</u></b> of survey respondents."

        summary += "</div>"
        st.markdown(summary, unsafe_allow_html=True)
        


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
