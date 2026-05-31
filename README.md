# 🎬 IMDB Sentiment Analysis and Topic Detection

## 🔍 Overview
A Natural Language Processing project that performs sentiment classification
and topic detection on IMDB movie reviews. The project compares traditional
machine learning approaches (Bag of Words) with a deep learning model (BERT)
to classify reviews as positive or negative.

University: University of Portsmouth
Module: Intelligent Data and Text Analytics (M33147)
Programme: MSc Data Analytics
Year: 2024-2025
Team: UP2302123, UP2312756, UP2301714

## 📊 Dataset
| File | Rows | Description |
|------|------|-------------|
| `imdb_labelled.txt` | 1,000 | Raw IMDB reviews with sentiment labels |
| `clean_imdb.csv` | 748 | Cleaned and preprocessed reviews |

**Target Variable:** `label` (0 = Negative, 1 = Positive)

## 🛠️ Tools & Technologies
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, SpaCy, Transformers (HuggingFace)
- **Models:** Naive Bayes, Logistic Regression, SVM, BERT
- **Environment:** Python Scripts

## ⚙️ Approach
1. Preprocessing — punctuation removal, number removal, lowercase conversion,
   stopword removal, lemmatization using SpaCy
2. Bag of Words Classification — Naive Bayes, Logistic Regression, SVM
   with CountVectorizer and 80:20 train-test split
3. BERT Classification — Fine-tuned bert-base-uncased for binary
   sentiment classification using HuggingFace Trainer API
4. Topic Detection — LDA with 10 topics using CountVectorizer

## 📈 Results
| Model | Accuracy |
|-------|----------|
| Naive Bayes (BoW) | 72% |
| Logistic Regression (BoW) | 72% |
| SVM (BoW) | 71% |
| BERT (Fine-tuned, 2 epochs) | 86.67% |

## 📁 Files
| File | Description |
|------|-------------|
| `preprocessing.py` | Text cleaning and lemmatization |
| `bow_classification.py` | Bag of Words with 3 classifiers |
| `topic_detection.py` | LDA topic modelling with 10 topics |
| `bert_classification.py` | BERT fine-tuning and evaluation |
| `data/imdb_labelled.txt` | Raw dataset |
| `data/clean_imdb.csv` | Preprocessed dataset |
| `report/idta_2.pdf` | Full project report |
| `requirements.txt` | Python libraries to install |

## 🚀 How to Run
1. Install dependencies
