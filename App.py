import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="SVM Klasifikasi Nganjuk", layout="wide")

st.title("📊 Klasifikasi Kepatuhan BPJS - SVM")
st.info("Kabupaten Nganjuk - Data Perusahaan 2024")

# Fungsi Load Data
def load_data():
    file_path = 'data.csv' # Pastikan nama file ini sesuai
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        # Preprocessing Kolom
        df['JML TK'] = pd.to_numeric(df['JML TK'], errors='coerce').fillna(0)
        df['JUMLAH BPJS KETENAGAKERJAAN'] = pd.to_numeric(df['JUMLAH BPJS KETENAGAKERJAAN'], errors='coerce').fillna(0)
        
        # Hitung Rasio
        df['Rasio'] = (df['JUMLAH BPJS KETENAGAKERJAAN'] / df['JML TK'] * 100).fillna(0)
        
        # Pelabelan 5 Kelompok
        def buat_label(x):
            if x <= 20: return "0%-20%"
            elif x <= 40: return "20%-40%"
            elif x <= 60: return "40%-60%"
            elif x <= 80: return "60%-80%"
            else: return "80%-100%"
            
        df['Kategori'] = df['Rasio'].apply(buat_label)
        return df
    else:
        return None

data = load_data()

if data is not None:
    st.success("File Berhasil Dimuat!")
    
    # --- PROSES SVM ---
    # Fitur: Jumlah TK dan Jumlah BPJS
    X = data[['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN']]
    y = data['Kategori']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model SVM
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)

    # --- TAMPILAN DASHBOARD ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Akurasi Model")
        st.metric(label="Accuracy Score", value=f"{akurasi*100:.2f}%")
        
        st.subheader("📈 Visualisasi Kelompok")
        fig, ax = plt.subplots()
        order = ["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]
        sns.countplot(data=data, x='Kategori', order=order, palette="magma", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("📝 Tabel Hasil Analisis")
        st.write(data[['PERUSAHAAN', 'JML TK', 'JUMLAH BPJS KETENAGAKERJAAN', 'Kategori']].head(10))
        
        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
        st.pyplot(fig_cm)

else:
    st.error("File 'data.csv' tidak ditemukan. Silakan unggah file dengan nama 'data.csv' ke repository Anda.")
