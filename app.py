# =========================================================
# 🚀 EMPLOYEE PERFORMANCE INTELLIGENCE SYSTEM (ADVANCED)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 🎨 PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Employee Performance Predictor",
    layout="wide",
    page_icon="📊"
)

# =========================================================
# 📁 PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "models", "encoders.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "employee_data.csv"))

dept_encoder = encoders["dept"]
perf_labels = encoders["perf"].classes_

# =========================================================
# 🎨 CUSTOM CSS (UI BOOST)
# =========================================================
st.markdown("""
<style>
.big-title {font-size: 40px; font-weight: bold; color: #4CAF50;}
.card {padding: 20px; border-radius: 10px; background-color: #f5f5f5;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🧠 FUNCTIONS
# =========================================================
def calculate_score(df):
    return (
        0.3*df["Experience"] +
        0.2*df["Training_Hours"] +
        0.3*df["Attendance"] +
        0.2*df["Projects_Completed"]
    ).astype(int)

def recommendation(score):
    if score >= 80:
        return "🏆 Promote Immediately"
    elif score >= 60:
        return "📈 Training Recommended"
    else:
        return "⚠️ Performance Plan Needed"

# =========================================================
# 🧭 SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🚀 Employee Performance Predictor")

page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard",
    "📊 Analytics",
    "🤖 Prediction",
    "📋 Scorecard",
    "📈 Drivers",
    "🏆 Top Performers",
    "📉 Low Performers",
    "📂 Bulk Intelligence",
    "⚙️ About"
])

# =========================================================
# 🏠 DASHBOARD
# =========================================================
if page == "🏠 Dashboard":
    st.markdown('<p class="big-title">📊 Performance Dashboard</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Employees", len(df))
    col2.metric("💰 Avg Salary", int(df["Salary"].mean()))
    col3.metric("📊 Avg Attendance", int(df["Attendance"].mean()))
    col4.metric("📈 Avg Experience", round(df["Experience"].mean(),1))

    st.subheader("📊 Performance Distribution")
    st.bar_chart(df["Performance_Label"].value_counts())

# =========================================================
# 📊 ANALYTICS
# =========================================================
elif page == "📊 Analytics":
    st.title("📊 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Salary vs Performance")
        fig, ax = plt.subplots()
        sns.boxplot(x="Performance_Label", y="Salary", data=df, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

# =========================================================
# 🤖 SINGLE PREDICTION
# =========================================================
elif page == "🤖 Prediction":
    st.title("🤖 Predict Employee Performance")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 60)
        exp = st.slider("Experience", 1, 35)
        salary = st.number_input("Salary", value=50000)

    with col2:
        training = st.slider("Training Hours", 0, 100)
        attendance = st.slider("Attendance", 50, 100)
        projects = st.slider("Projects Completed", 1, 20)

    dept = st.selectbox("Department", dept_encoder.classes_)
    dept_encoded = dept_encoder.transform([dept])[0]

    if st.button("Predict Performance"):
        features = np.array([[age, exp, salary, training, attendance, projects, dept_encoded]])
        pred = model.predict(features)[0]

        score = int(0.3*exp + 0.2*training + 0.3*attendance + 0.2*projects)

        st.success(f"🎯 Performance: {perf_labels[pred]}")
        st.info(f"📊 Score: {score}/100")
        st.warning(f"💡 Recommendation: {recommendation(score)}")

# =========================================================
# 📋 SCORECARD
# =========================================================
elif page == "📋 Scorecard":
    st.title("📋 Employee Scorecard")

    idx = st.number_input("Select Employee Index", 0, len(df)-1)
    row = df.iloc[idx]

    score = calculate_score(row)

    st.dataframe(row.to_frame().T)

    col1, col2 = st.columns(2)
    col1.metric("Performance Score", score)
    col2.success(recommendation(score))

# =========================================================
# 📈 DRIVERS
# =========================================================
elif page == "📈 Drivers":
    st.title("📈 Performance Drivers")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(
        x=[0.3,0.2,0.3,0.2],
        y=["Experience","Training","Attendance","Projects"],
        ax=ax
    )
    st.pyplot(fig)

# =========================================================
# 🏆 TOP PERFORMERS
# =========================================================
elif page == "🏆 Top Performers":
    st.title("🏆 Top Performers")

    top = df[df["Performance_Label"]=="High"]

    if len(top) == 0:
        top = df.sort_values(by="Experience", ascending=False).head(5)
        st.warning("No High performers → Showing top 5 based on experience")

    st.dataframe(top)

# =========================================================
# 📉 LOW PERFORMERS
# =========================================================
elif page == "📉 Low Performers":
    st.title("📉 Low Performers")

    low = df[df["Performance_Label"]=="Low"]

    if len(low) == 0:
        low = df.sort_values(by="Experience").head(5)
        st.warning("No Low performers → Showing lowest 5")

    st.dataframe(low)

# =========================================================
# 📂 BULK INTELLIGENCE (ADVANCED)
# =========================================================
elif page == "📂 Bulk Intelligence":
    st.title("📂 Bulk HR Intelligence")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df_new = pd.read_csv(file)

        try:
            if "Performance_Label" in df_new.columns:
                df_new = df_new.drop("Performance_Label", axis=1)

            df_new["Department"] = dept_encoder.transform(df_new["Department"])

            cols = ["Age","Experience","Salary","Training_Hours","Attendance","Projects_Completed","Department"]
            df_new = df_new[cols]

            preds = model.predict(df_new)

            df_new["Performance"] = [perf_labels[p] for p in preds]
            df_new["Score"] = calculate_score(df_new)
            df_new["Recommendation"] = df_new["Score"].apply(recommendation)

            # KPIs
            st.subheader("📊 Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Employees", len(df_new))
            col2.metric("Avg Score", int(df_new["Score"].mean()))
            col3.metric("High Performers", (df_new["Performance"]=="High").sum())

            # Top performers fallback
            top = df_new[df_new["Performance"]=="High"]
            if len(top) == 0:
                top = df_new.sort_values(by="Score", ascending=False).head(5)

            # Low performers fallback
            low = df_new[df_new["Performance"]=="Low"]
            if len(low) == 0:
                low = df_new.sort_values(by="Score").head(5)

            st.subheader("🏆 Top Performers")
            st.dataframe(top)

            st.subheader("📉 Low Performers")
            st.dataframe(low)

            st.subheader("📋 Full Report")
            st.dataframe(df_new)

        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# ⚙️ ABOUT
# =========================================================
elif page == "⚙️ About":
    st.title("⚙️ About System")

    st.markdown("""
    ### 🚀 Employee Performance Intelligence System

    🔹 Predicts employee performance using Machine Learning  
    🔹 Generates performance score  
    🔹 Provides HR recommendations  
    🔹 Supports bulk employee analysis  

    Built with:
    - Python
    - XGBoost
    - Streamlit
    """)