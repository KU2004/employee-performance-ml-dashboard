# 🚀 Employee Performance Intelligence System

## 📌 Overview
This project is an **AI-powered Employee Performance Intelligence System** designed to help HR teams analyze, predict, and improve employee performance using Machine Learning.

It goes beyond basic prediction by providing:
- 📊 Performance scoring
- ⚠️ Risk detection
- 💡 Improvement suggestions
- 📈 Business insights for HR decision-making

---

## 🎯 Problem Statement
Organizations struggle to:
- Identify high-performing employees
- Detect low performers early
- Make data-driven HR decisions

This system solves that using **data analytics + machine learning**.

---

## 💡 Key Features

### 🔹 1. Performance Prediction
- Predicts employee performance (High / Medium / Low)
- Uses **XGBoost Machine Learning model**

### 🔹 2. Performance Score
- Calculates score (0–100)
- Based on:
  - Experience
  - Training hours
  - Attendance
  - Projects completed

### 🔹 3. Confidence Score
- Shows prediction confidence %

### 🔹 4. Risk Detection
- Identifies employees at risk:
  - 🔴 High Risk
  - 🟡 Medium Risk
  - 🟢 Low Risk

### 🔹 5. AI Recommendations
- Suggests actions:
  - Promote
  - Train
  - Improve

### 🔹 6. Bulk Employee Analysis
- Upload CSV file
- Get:
  - Predictions
  - Scores
  - Recommendations
  - Risk analysis

### 🔹 7. Department Insights
- Performance analysis by department

### 🔹 8. Interactive Dashboard
- Built using **Streamlit**
- Real-time prediction UI

---

## 🏗️ Project Architecture
Data → Preprocessing → Model Training → Prediction → Insights → HR Decisions

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib, Seaborn

---

## 📁 Folder Structure
Employee-Performance-Predictor/
│
├── data/
├── src/
├── models/
├── outputs/
├── app.py
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1. Clone repositorygit clone https://github.com/your-username/employee-performance-intelligence-system.git

cd employee-performance-intelligence-system

### 2. Create virtual environment
python -m venv venv

### 3. Activate environment
venv\Scripts\activate (Windows)
source venv/bin/activate (Mac/Linux)


### 4. Install dependencies

pip install -r requirements.txt


---

## ▶️ Run Project

### Step 1: Train model

python src/train_model.py


### Step 2: Run app

streamlit run app.py


---

## 📊 Outputs

- Performance prediction
- Score visualization
- Risk detection
- HR insights dashboard

---

## 🧠 How It Works

1. Synthetic employee data is generated
2. Data is preprocessed and encoded
3. XGBoost model is trained
4. Model predicts performance
5. System generates insights for HR

---

## 💼 Business Impact

This system helps HR teams:
- Improve productivity
- Identify talent
- Reduce performance issues
- Make data-driven decisions

---

## 🧠 Interview Explanation

“I built an AI-powered employee performance intelligence system that not only predicts performance but also provides scoring, risk detection, and actionable insights for HR decision-making.”

---

## 🚀 Future Improvements

- SHAP explainability
- Real-time API (FastAPI)
- Deployment (AWS / Render)
- Real HR dataset integration

---

## 👨‍💻 Author

Kunal Patil
