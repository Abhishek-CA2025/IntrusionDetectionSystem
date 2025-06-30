
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="AI-Based Intrusion Detection System", layout="wide")
st.title("🚨 AI-Based Intrusion Detection System")

uploaded_file = st.file_uploader("Upload NSL-KDD CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split features and labels
    X = df.drop('attack', axis=1)
    y = df['attack']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    model_choice = st.selectbox("Choose Classifier", ("Random Forest", "SVM", "Naive Bayes", "Neural Network"))

    if st.button("Train and Evaluate"):
        if model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "SVM":
            model = SVC()
        elif model_choice == "Naive Bayes":
            model = GaussianNB()
        else:
            model = Sequential()
            model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            score = model.evaluate(X_test, y_test, verbose=0)
            st.success(f"Neural Network Accuracy: {score[1]*100:.2f}%")
            st.stop()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        st.success(f"Model Accuracy: {acc * 100:.2f}%")
        st.text("Classification Report:")
        st.text(classification_report(y_test, predictions))
