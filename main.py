import pandas as pd
from preprocessing import preprocess_dataframe
from model import train_model

# --- 1. Load dataset ---
df = pd.read_csv("Supermarket Sales Cleaned.csv")

# --- 2. Tentukan kolom teks & label ---
TEXT_COL = "Review"        # ganti sesuai dataset
LABEL_COL = "Rating"       # ganti sesuai dataset

# --- 3. Preprocessing ---
df = preprocess_dataframe(df, TEXT_COL)

# --- 4. Training model ---
train_model(df, TEXT_COL, LABEL_COL)
