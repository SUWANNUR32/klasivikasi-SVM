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

# Fungsi mencari file CSV
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        return None, None
    
    target_file = files[0]
    try:
        # PERBAIKAN UTAMA: Menambahkan sep=';' karena file anda menggunakan titik koma
        df = pd.read_csv(target_file, sep=';', encoding='latin1')
        return df, target_file
    except Exception as e:
        st.error(f"Gagal membaca file {target_file}: {e}")
        return None, None

st.title("🚀 Klasifikasi Kepatuhan BPJS Ketenagakerjaan")
st.markdown("Analisis tingkat kepatuhan perusahaan menggunakan algoritma SVM.")

df, target_file = load_data()

if df is not None:
    # Bersihkan nama kolom dari spasi atau karakter aneh
    df.columns = df.columns.str.strip()
    
    # Kolom yang dibutuhkan
    cols_needed = ['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN', 'PERUSAHAAN']
    
    if all(col in df.columns for col in cols_needed):
        # Konversi data ke numerik (menghapus koma jika ada format ribuan)
        for col in ['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN']:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        # Hitung Rasio
        df['Rasio'] = (df['JUMLAH BPJS KETENAGAKERJAAN'] / df['JML TK'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Labeling 5 Kelompok
        def buat_label(x):
            if x <= 20: return "0%-20%"
            elif x <= 40: return "20%-40%"
            elif x <= 60: return "40%-60%"
            elif x <= 80: return "60%-80%"
            else: return "80%-100%"
            
        df['Label'] = df['Rasio'].apply(buat_label)

        # --- SVM ---
        X = df[['JML TK', 'JUMLAH BPJS KETENAGAKERJAAN']]
        y = df['Label']
        
        if len(df) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Gunakan linear kernel sesuai analisis Anda
            model = SVC(kernel='linear', C=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            # --- DISPLAY ---
            st.success(f"Berhasil memuat: {target_file}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Akurasi Model", f"{acc*100:.2f}%")
                st.subheader("Distribusi Kelompok")
                fig, ax = plt.subplots()
                order = ["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]
                sns.countplot(data=df, x='Label', order=order, palette="magma")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with c2:
                st.subheader("Data Hasil Klasifikasi")
                st.dataframe(df[['PERUSAHAAN', 'JML TK', 'JUMLAH BPJS KETENAGAKERJAAN', 'Label']].head(10))
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=order)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=order, yticklabels=order)
                st.pyplot(fig_cm)
    else:
        st.error(f"Kolom tidak ditemukan. Kolom yang ada: {df.columns.tolist()}")
else:
    st.error("File data.csv tidak ditemukan di GitHub.")
