import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

def train_model(df, text_col, label_col):
    X = df[text_col]
    y = df[label_col]

    # Vectorizer TF-IDF
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    # Model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Simpan model & vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model & vectorizer berhasil disimpan!")
