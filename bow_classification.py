import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
df = pd.read_csv("data/clean_imdb.csv")

# Remove empty rows
df = df.dropna(subset=["clean"])

X = df["clean"]
y = df["label"]

# Bag of Words
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_bow, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": LinearSVC()
}

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n====================")
    print(name)
    print("====================")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
