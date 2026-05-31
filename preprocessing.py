import pandas as pd
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("data/imdb_labelled.txt", sep="\t", header=None)
df.columns = ["sentence", "label"]

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    doc = nlp(" ".join(words))
    return " ".join([token.lemma_ for token in doc])


df["clean"] = df["sentence"].apply(clean_text)

df[["clean", "label"]].to_csv("data/clean_imdb.csv", index=False)

print("Preprocessing Finished")
