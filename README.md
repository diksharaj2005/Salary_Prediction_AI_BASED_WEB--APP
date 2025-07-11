# 💼 Salary Prediction Web App

A modern and responsive **Machine Learning-powered web application** built with **Streamlit**, capable of predicting a person's salary based on demographic and professional attributes. This app uses a trained **XGBoost regression model** and features a **blurred background UI** for enhanced aesthetics.

---

## 🧭 Table of Contents

- [🔍 Overview](#-overview)
- [🎯 Features](#-features)
- [📊 Model Insights](#-model-insights)
- [🛠️ Tech Stack](#️-tech-stack)
- [📦 Installation](#-installation)
- [▶️ Run the App](#️-run-the-app)
- [🗂️ Project Structure](#️-project-structure)
- [☁️ Deployment](#️-deployment)
- [📃 License](#-license)
- [👩‍💻 Author](#-author)

---

## 🔍 Overview

This app allows users to input a few personal and professional details such as age, gender, education level, job title, and years of experience, and receive a **real-time salary prediction**. It is ideal for:
- Job seekers
- HR professionals
- Salary benchmarking
- Educational or demo purposes

---

## 🎯 Features

✅ Predicts annual salary using:
- Age  
- Gender  
- Education level  
- Job title  
- Years of experience  

✅ Engineered features:
- `Experience_per_Age`: Ratio of experience to age  
- `Seniority`: Binary feature derived from job titles like *Manager*, *Lead*, *CEO*, etc.

✅ Interface:
- Built with Streamlit for fast UI deployment
- Responsive layout
- Full-screen **blurred background image**
- Minimal, clean, user-friendly design

✅ Model:
- XGBoost Regressor
- Log-transformed predictions for better accuracy
- Outputs real-world dollar value predictions

---

## 📊 Model Insights

- 📈 **Model Used**: XGBoost Regressor
- 🔁 **Target Variable**: Log-transformed annual salary
- 🎯 **Score (R²)**: ~0.87 – 0.95 depending on feature set
- ⚙️ **Feature Engineering**:
  - Experience-per-age
  - Seniority from job title
  - One-hot encoded categorical variables
- 🧪 **Evaluation Metrics**:
  - RMSE
  - MAE
  - R² Score

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)
- **Backend**: Python, XGBoost
- **Libraries**:
  - pandas, numpy
  - joblib
  - scikit-learn
- **Styling**:
  - Custom CSS
  - Background blur using `filter: blur(...)`
- **Model**: Trained XGBoost regression model (`salary_model.pkl`)

---

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/salary-predictor-app.git
cd ibm

### Step 2: Set Up Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

### Step 3: Install Dependencies
``` bash

pip install -r requirements.txt


### Step 4: Ensure Assets Are in Place
Trained model: models/salary_model.pkl


### step 5 run the app
streamlit run app.py


🗂️ Project Structure
salary-predictor-app/
├── app.py                  # Main Streamlit app script
├── models/
│   └── salary_model.pkl    # Trained XGBoost model
├── static/
│   ├── bg.png              # Background image for UI
│   └── screenshot.png      # App preview image
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation


📃 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute.

👩‍💻 Author
Diksha Raj
Built with ❤️ using Streamlit and Machine Learning.
