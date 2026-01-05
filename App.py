import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Kepatuhan BPJS", layout="wide")

st.title("📊 Aplikasi Analisis Kepatuhan BPJS Ketenagakerjaan")
st.markdown("Klasifikasi menggunakan **Support Vector Machine (SVM)** berdasarkan data perusahaan di Kabupaten Nganjuk.")

# 1. Load Data
@st.cache_data
def load_data():
    # Pastikan nama file sesuai dengan yang ada di folder Anda
    df = pd.read_csv("Daftar Data Perusahaan Menengah ke Atas dengan lokasinya Tahun 2024 (Nganjuk).xlsx - Perusahaan.csv")
    
    # Preprocessing sederhana: Menghitung persentase
    # Mengonversi kolom ke numerik untuk menghindari error
    df['JML TK'] = pd.to_numeric(df['JML TK'], errors='coerce').fillna(0)
    df['JUMLAH BPJS KETENAGAKERJAAN'] = pd.to_numeric(df['JUMLAH BPJS KETENAGAKERJAAN'], errors='coerce').fillna(0)
    
    # Hitung Rasio Kepatuhan
    df['Rasio'] = (df['JUMLAH BPJS KETENAGAKERJAAN'] / df['JML TK']) * 100
    df['Rasio'] = df['Rasio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Pelabelan berdasarkan 5 Kelompok (0-20%, dst)
    def label_kepatuhan(x):
        if x <= 20: return "0%-20% (Sangat Kurang)"
        elif x <= 40: return "20%-40% (Kurang)"
        elif x <= 60: return "40%-60% (Cukup)"
        elif x <= 80: return "60%-80% (Baik)"
        else: return "80%-100% (Sangat Baik)"
    
    df['Kategori'] = df['Rasio'].apply(label_kepatuhan)
    return df

try:
    data = load_data()
    
    # Sidebar untuk navigasi
    menu = st.sidebar.selectbox("Pilih Tampilan", ["Dataset", "Hasil Analisis & SVM"])

    if menu == "Dataset":
        st.subheader("Data Perusahaan")
        st.dataframe(data)
        st.write(f"Total Data: {len(data)} Perusahaan")

    else:
        # 2. Proses SVM
        st.subheader("Hasil Klasifikasi SVM")
        
        # Fitur (X) dan Target (y)
        # Menggunakan JML TK dan Jumlah BPJS sebagai fitur
        X = data[['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN']]
        y = data['Kategori']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Inisialisasi Model (Menggunakan RBF Kernel sebagai default yang stabil)
        model = SVC(kernel='linear') 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Tampilkan Akurasi
        acc = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Akurasi Model", f"{acc*100:.2f}%")
        
        # 3. Menampilkan 5 Kelompok
        st.write("---")
        st.subheader("Distribusi 5 Kelompok Kepatuhan")
        
        counts = data['Kategori'].value_counts().reindex([
            "0%-20% (Sangat Kurang)", "20%-40% (Kurang)", 
            "40%-60% (Cukup)", "60%-80% (Baik)", "80%-100% (Sangat Baik)"
        ]).fillna(0)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Jumlah Perusahaan")
        st.pyplot(fig)
        
        # Confusion Matrix
        st.write("---")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig_cm)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.info("Pastikan file CSV berada di folder yang sama dengan app.py")
