import os
import pandas as pd
import streamlit as st

st.title("Streamlit - Load Dataset Otomatis")

possible_files = [
    "Supermarket Sales Cleaned.csv",
    "dataset.csv",
    "data.csv"
]

df = None

# Coba load file yang tersedia
for file in possible_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        st.success(f"Loaded: {file}")
        break

# Jika tidak ada dataset
if df is None:
    st.error("Dataset tidak ditemukan. Pastikan salah satu file CSV berikut berada di folder proyek:")
    st.write(possible_files)
    st.stop()

# Tampilkan dataset
st.subheader("Preview Dataset")
st.dataframe(df)

st.write("Jumlah baris:", df.shape[0])
st.write("Jumlah kolom:", df.shape[1])
