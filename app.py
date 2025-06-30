import streamlit as st
import pandas as pd
import numpy as np
import datetime, random
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Intrusion Detection System", layout="centered")
st.title("🚨 Intrusion Detection Dashboard")

model = load_model("intrusion_model.h5")
scaler = StandardScaler()

uploaded_file = st.file_uploader("📂 Upload test data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Uploaded Data:")
    st.dataframe(df.head())

    if 'label' in df.columns:
        X = df.drop('label', axis=1)
    else:
        X = df

    X_scaled = scaler.fit_transform(X)
    preds = model.predict(X_scaled)
    labels = np.argmax(preds, axis=1)

    st.subheader("🔍 Detection Results")
    for i, label in enumerate(labels[:10]):
        attack = "Normal" if label == 11 else "Attack"
        src_ip = f"192.168.1.{random.randint(1,254)}"
        dst_ip = f"10.0.0.{random.randint(1,254)}"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        st.markdown(f"""
        #### 🔔 {attack} Detected
        - 🛠 **Label**: {label}  
        - 📡 **Source IP**: {src_ip}  
        - 🎯 **Target IP**: {dst_ip}  
        - 🕒 **Time**: {timestamp}
        ---
        """)
