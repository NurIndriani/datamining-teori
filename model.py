import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import preprocess_dataframe  # pastikan nama file benar

# ======================================================
#   FUNGSI TRAINING (2 MODEL + ENSEMBLE)
# ======================================================

def train_model(dataset_path, text_col, label_col):

    # --- LOAD DATA ---
    df = pd.read_csv(dataset_path)

    # --- PREPROCESSING ---
    df = preprocess_dataframe(df, text_col)

    X = df[text_col]
    y = df[label_col]

    # --- TF-IDF VECTORIZER ---
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # --- TRAIN-TEST SPLIT ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # -----------------------------------------------------
    #   MODEL A: Multinomial Naive Bayes
    # -----------------------------------------------------
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # -----------------------------------------------------
    #   MODEL B: SVM + Calibrator (biar bisa soft voting)
    # -----------------------------------------------------
    svm_base = LinearSVC()
    svm = CalibratedClassifierCV(svm_base)
    svm.fit(X_train, y_train)

    # -----------------------------------------------------
    #   VOTING CLASSIFIER (MODEL C)
    # -----------------------------------------------------
    voting_model = VotingClassifier(
        estimators=[("nb", nb), ("svm", svm)],
        voting="soft"
    )
    voting_model.fit(X_train, y_train)

    # --- EVALUASI ---
    pred_voting = voting_model.predict(X_test)
    acc = accuracy_score(y_test, pred_voting)

    print("======================================")
    print("AKURASI ENSEMBLE:", acc)
    print("======================================")
    print(classification_report(y_test, pred_voting))

    # --- SIMPAN MODEL ---
    with open("model.pkl", "wb") as f:
        pickle.dump(voting_model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model & vectorizer berhasil disimpan!")
    return acc
