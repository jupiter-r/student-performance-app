# ======================================
# Student Performance Prediction System
# Author: JUPITER R
# Tools: pandas, numpy, scikit-learn, streamlit
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# STEP 1: PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("Student Performance Prediction System")
st.write("This app predicts whether a student is likely to **Pass** or **Fail** based on academic and personal factors.")

# -------------------------------
# STEP 2: LOAD & PREPROCESS DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('student-mat.csv', sep=';')
    df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(['G3'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()

X = df.drop('pass', axis=1)
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 3: TRAIN MODELS
# -------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

# -------------------------------
# STEP 4: DISPLAY MODEL PERFORMANCE
# -------------------------------
st.subheader("Model Performance")
st.write(f"**Logistic Regression Accuracy:** {acc_log*100:.2f}%")
st.write(f"**Decision Tree Accuracy:** {acc_tree*100:.2f}%")

st.info("Both models are trained on the same dataset. You can use either one below to make predictions.")

# -------------------------------
# STEP 5: USER INPUT SECTION
# -------------------------------
st.subheader("Enter Student Details")

studytime = st.slider("Weekly Study Time (1–4)", 1, 4, 2)
failures = st.slider("Number of Past Failures", 0, 4, 0)
absences = st.slider("Absences", 0, 75, 5)
G1 = st.slider("First Period Grade (0–20)", 0, 20, 10)
G2 = st.slider("Second Period Grade (0–20)", 0, 20, 10)

model_choice = st.radio("Select Model", ["Logistic Regression", "Decision Tree"])

# -------------------------------
# STEP 6: PREDICTION LOGIC
# -------------------------------
if st.button("Predict Performance"):
    # Create an input array aligned to X's columns
    input_data = np.zeros((1, X.shape[1]))
    # Fill only known features
    feature_list = list(X.columns)
    for feature, value in zip(['studytime', 'failures', 'absences', 'G1', 'G2'], 
                              [studytime, failures, absences, G1, G2]):
        if feature in feature_list:
            idx = feature_list.index(feature)
            input_data[0, idx] = value
    
    # Predict
    if model_choice == "Logistic Regression":
        prediction = log_model.predict(input_data)[0]
        prob = log_model.predict_proba(input_data)[0][1]
    else:
        prediction = tree_model.predict(input_data)[0]
        prob = tree_model.predict_proba(input_data)[0][1]
    
    # Result Display
    if prediction == 1:
        st.success(f" The student is likely to PASS! (Confidence: {prob*100:.2f}%)")
    else:
        st.error(f" The student is likely to FAIL. (Confidence: {(1-prob)*100:.2f}%)")


st.caption("Developed by JUPITER R")