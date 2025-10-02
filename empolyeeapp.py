import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 🎯 Page setup
# -------------------------------
st.set_page_config(page_title="Employee Insights Dashboard", layout="wide")
st.title("📊 Employee Attrition & Performance Prediction Dashboard")

st.markdown("""
This dashboard helps HR teams to:
- Predict **Employee Attrition** (who may leave)
- Predict **Performance Rating**
- Visualize insights from HR data
""")

# -------------------------------
# ⚙️ Load trained models
# -------------------------------
@st.cache_resource
def load_models():
    # ✅ Use forward slashes to avoid errors
    attrition_model = joblib.load(r"D:\my_third_project_employee\attrition_model.pkl")
    performance_model = joblib.load(r"D:\my_third_project_employee\performance_model.pkl")

    return attrition_model, performance_model

attrition_model, performance_model = load_models()

# -------------------------------
# 📄 Load Cleaned CSV
# -------------------------------
DATA_PATH = "C:/Users/itsve/Downloads/cleaned_attrition.csv"
df = pd.read_csv(DATA_PATH)

st.success("✅ Cleaned dataset loaded successfully!")
st.subheader("🔍 Data Preview")
st.dataframe(df.head())

# -------------------------------
# 🔮 Predictions
# -------------------------------
st.subheader("📈 Predictions")

# --- Attrition Prediction ---
features_attr = ['Age','Department','MonthlyIncome','JobSatisfaction',
                 'YearsAtCompany','MaritalStatus','OverTime','Gender','BusinessTravel']

if all(col in df.columns for col in features_attr):
    X_attr = df[features_attr]
    df['Predicted_Attrition'] = attrition_model.predict(X_attr)
    df['Attrition_Probability'] = attrition_model.predict_proba(X_attr)[:, 1]
else:
    st.error("❌ Missing required columns for Attrition prediction.")

# --- Performance Rating Prediction ---
features_perf = ['Education','JobInvolvement','JobLevel','MonthlyIncome',
                 'YearsAtCompany','YearsInCurrentRole','TrainingTimesLastYear','WorkLifeBalance']

if all(col in df.columns for col in features_perf):
    X_perf = df[features_perf]
    df['Predicted_PerformanceRating'] = performance_model.predict(X_perf)
else:
    st.error("❌ Missing required columns for Performance Rating prediction.")

# -------------------------------
# 🧭 Top At-Risk Employees
# -------------------------------
if 'Attrition_Probability' in df.columns:
    st.subheader("🧭 Top 10 Employees at Risk of Leaving")
    top_risk = df.sort_values('Attrition_Probability', ascending=False).head(10)
    st.dataframe(top_risk[['Age','Department','MonthlyIncome','OverTime','JobSatisfaction','Attrition_Probability']])

# -------------------------------
# ⬇️ Download Predictions
# -------------------------------
st.download_button(
    label="⬇️ Download Predictions CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="predictions.csv",
    mime="text/csv"
)

# -------------------------------
# 📊 EDA & Charts
# -------------------------------
st.subheader("📊 Insights & Visualizations")

col1, col2 = st.columns(2)

with col1:
    if 'Attrition' in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='Attrition', data=df, ax=ax)
        ax.set_title("Attrition Distribution")
        st.pyplot(fig)

    if 'OverTime' in df.columns and 'Attrition' in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='OverTime', hue='Attrition', data=df, ax=ax)
        ax.set_title("OverTime vs Attrition")
        st.pyplot(fig)

with col2:
    if 'JobSatisfaction' in df.columns and 'Attrition' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x='Attrition', y='JobSatisfaction', data=df, ax=ax)
        ax.set_title("Job Satisfaction vs Attrition")
        st.pyplot(fig)

    if 'PerformanceRating' in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='PerformanceRating', data=df, ax=ax)
        ax.set_title("Performance Rating Distribution")
        st.pyplot(fig)

st.success("✅ Dashboard is ready! Explore predictions and insights above.")

