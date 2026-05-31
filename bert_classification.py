import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Load cleaned IMDb dataset
df = pd.read_csv("data/clean_imdb.csv")
df = df.dropna()

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize(batch):
    return tokenizer(batch["clean"], padding=True, truncation=True)


train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()


# Get predictions
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
print("BERT Accuracy:", accuracy)