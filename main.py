import os
import pandas as pd

possible_files = [
    "Supermarket Sales Cleaned.csv",
    "dataset.csv",
    "data.csv"
]

df = None

for file in possible_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        print("Loaded:", file)
        break

if df is None:
    st.error("Dataset tidak ditemukan. Pastikan file CSV ada dalam folder proyek Streamlit.")
    st.stop()
