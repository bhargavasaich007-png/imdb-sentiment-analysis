import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load cleaned dataset
df = pd.read_csv("data/clean_imdb.csv")

# Ensure no empty text
df = df.dropna(subset=["clean"])

texts = df["clean"].astype(str)

# Convert to BoW
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

# LDA
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    print("\nTopic", i+1)
    print([words[j] for j in topic.argsort()[-10:]])
