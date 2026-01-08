import streamlit as st
import pickle
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ===============================
# LOAD MODEL & TF-IDF
# ===============================
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    model = pickle.load(f)

# ===============================
# PREPROCESSING
# ===============================
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊")

st.title("📊 Analisis Sentimen Media Sosial")
st.write("Model: **Random Forest + TF-IDF**")

user_input = st.text_area("Masukkan teks:")

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        clean_text = preprocess(user_input)
        vector = tfidf.transform([clean_text])
        prediction = model.predict(vector)[0]

        label_map = {
            0: "Negatif ❌",
            1: "Netral ⚖️",
            2: "Positif ✅"
        }

        st.success(f"Hasil Sentimen: **{label_map[prediction]}**")
