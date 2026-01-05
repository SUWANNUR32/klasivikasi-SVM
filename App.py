import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analisis SVM Nganjuk", layout="wide")

# Fungsi mencari file CSV dengan penanganan encoding
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        return None, None
    
    target_file = files[0]
    try:
        # Menambahkan encoding='latin1' untuk mengatasi UnicodeDecodeError
        df = pd.read_csv(target_file, encoding='latin1')
        return df, target_file
    except Exception as e:
        st.error(f"Gagal membaca file {target_file}: {e}")
        return None, None

st.title("🚀 Klasifikasi Kepatuhan BPJS Ketenagakerjaan")
st.markdown("Analisis tingkat kepatuhan perusahaan di Kabupaten Nganjuk menggunakan algoritma SVM.")

df, target_file = load_data()

if df is not None:
    # --- PREPROCESSING ---
    # Membersihkan nama kolom dari spasi berlebih
    df.columns = df.columns.str.strip()
    
    # Pastikan kolom yang dibutuhkan ada
    cols_needed = ['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN', 'PERUSAHAAN']
    if all(col in df.columns for col in cols_needed):
        
        # Konversi ke numerik dan bersihkan data kosong
        df['JML TK'] = pd.to_numeric(df['JML TK'], errors='coerce').fillna(0)
        df['JUMLAH BPJS KETENAGAKERJAAN'] = pd.to_numeric(df['JUMLAH BPJS KETENAGAKERJAAN'], errors='coerce').fillna(0)
        
        # Hitung Rasio (Persentase)
        df['Rasio'] = (df['JUMLAH BPJS KETENAGAKERJAAN'] / df['JML TK'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Pengelompokan 5 Kategori
        def kelomopok_kepatuhan(x):
            if x <= 20: return "0%-20%"
            elif x <= 40: return "20%-40%"
            elif x <= 60: return "40%-60%"
            elif x <= 80: return "60%-80%"
            else: return "80%-100%"
            
        df['Label'] = df['Rasio'].apply(kelomopok_kepatuhan)

        # --- ANALISIS SVM ---
        X = df[['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN']]
        y = df['Label']
        
        # Cek jika data cukup untuk split
        if len(df) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_svm = SVC(kernel='linear', C=1.0)
            model_svm.fit(X_train, y_train)
            y_pred = model_svm.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            # --- TAMPILAN ---
            st.success(f"Berhasil memuat dan menganalisis: **{target_file}**")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Performa Model")
                st.metric("Akurasi SVM", f"{score*100:.2f}%")
                
                # Plot Distribusi
                fig, ax = plt.subplots()
                order = ["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]
                sns.countplot(data=df, x='Label', order=order, palette="viridis", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                st.subheader("📋 Sampel Hasil")
                st.dataframe(df[['PERUSAHAAN', 'JML TK', 'JUMLAH BPJS KETENAGAKERJAAN', 'Label']].head(10))
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=order)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=order, yticklabels=order)
                plt.xlabel('Prediksi')
                plt.ylabel('Aktual')
                st.pyplot(fig_cm)
        else:
            st.warning("Data terlalu sedikit untuk melakukan klasifikasi SVM.")
    else:
        st.error(f"Kolom tidak sesuai! Pastikan file memiliki kolom: {cols_needed}")
        st.write("Kolom yang terdeteksi:", df.columns.tolist())

else:
    st.error("❌ File tidak ditemukan atau tidak bisa dibaca.")
    st.info("Pastikan file CSV hasil download dari Excel ada di folder yang sama.")
