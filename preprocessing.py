import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- 1. Buat stemmer & stopword ---
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)     # hapus URL
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)        # hapus simbol
    text = re.sub(r'\s+', ' ', text).strip()        # hapus spasi ganda
    
    # Buang stopword
    text = stopword.remove(text)

    # Stemming
    text = stemmer.stem(text)

    return text

def preprocess_dataframe(df, text_column):
    """ Membersihkan seluruh kolom text """
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df
