import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_dataframe(df):
    """ Preprocessing untuk dataset Supermarket Sales """

    df = df.copy()

    # ---- 1. Drop kolom yang tidak dibutuhkan ----
    drop_cols = ["Invoice ID", "Date", "Time"]  # tidak relevan untuk prediksi rating
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # ---- 2. Handle missing value ----
    df = df.fillna(method="ffill")

    # ---- 3. Pisahkan target ----
    if "Rating" not in df.columns:
        raise ValueError("Kolom 'Rating' tidak ditemukan dalam dataset")

    y = df["Rating"]
    X = df.drop("Rating", axis=1)

    # ---- 4. Encode kolom kategori ----
    le_dict = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le

    # ---- 5. Normalisasi fitur numerik ----
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, le_dict, scaler
