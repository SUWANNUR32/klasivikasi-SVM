import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analisis SVM Nganjuk", layout="wide")

# Fungsi mencari file CSV di folder secara otomatis
def find_csv():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    return files[0] if files else None

st.title("🚀 Klasifikasi Kepatuhan BPJS Ketenagakerjaan")
st.markdown("Aplikasi ini mendeteksi tingkat kepatuhan berdasarkan rasio jumlah tenaga kerja.")

# Coba cari file
target_file = find_csv()

if target_file:
    # Membaca data
    df = pd.read_csv(target_file)
    
    # Preprocessing Data (Berdasarkan Struktur File Anda)
    # Pastikan nama kolom sesuai: 'JML TK' dan 'JUMLAH BPJS KETENAGAKERJAAN'
    df['JML TK'] = pd.to_numeric(df['JML TK'], errors='coerce').fillna(0)
    df['JUMLAH BPJS KETENAGAKERJAAN'] = pd.to_numeric(df['JUMLAH BPJS KETENAGAKERJAAN'], errors='coerce').fillna(0)
    
    # Hitung Rasio untuk Labeling
    df['Rasio'] = (df['JUMLAH BPJS KETENAGAKERJAAN'] / df['JML TK'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Definisi 5 Kelompok
    def kelomopok_kepatuhan(x):
        if x <= 20: return "0%-20%"
        elif x <= 40: return "20%-40%"
        elif x <= 60: return "40%-60%"
        elif x <= 80: return "60%-80%"
        else: return "80%-100%"
        
    df['Label'] = df['Rasio'].apply(kelomopok_kepatuhan)

    # --- BAGIAN ANALISIS SVM ---
    # Fitur X (JML TK & BPJS), Target y (Label)
    X = df[['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN']]
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model SVM (Linear agar cepat)
    model_svm = SVC(kernel='linear')
    model_svm.fit(X_train, y_train)
    y_pred = model_svm.predict(X_test)
    
    # Hitung Akurasi
    score = accuracy_score(y_test, y_pred)

    # --- TAMPILAN DASHBOARD ---
    st.success(f"Berhasil memuat file: {target_file}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Akurasi Model SVM")
        st.metric("Akurasi", f"{score*100:.2f}%")
        
        # Plot Distribusi Kelompok
        st.subheader("Distribusi Perusahaan")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Label', order=["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"], palette="viridis")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("📋 Sampel Data Hasil Klasifikasi")
        st.write(df[['PERUSAHAAN', 'JML TK', 'JUMLAH BPJS KETENAGAKERJAAN', 'Label']].head(10))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig_cm)

else:
    st.error("❌ File CSV tidak ditemukan di folder!")
    st.info("Pastikan file CSV hasil ekspor dari Excel sudah berada di folder yang sama dengan app.py")
