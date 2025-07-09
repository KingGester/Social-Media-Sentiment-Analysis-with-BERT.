

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import Dataset
import numpy as np

from .tokenizer import tokenize_function
from .model import build_model
from .metrics import compute_metrics


def prepare_data(df, text_col="Text", label_col="Sentiment", test_size=0.2):
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[label_col])

    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df[[text_col, "label"]],
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))

    return train_dataset, test_dataset, label_encoder


def train_transformer_model(dataset_dict, label_encoder, model_path="./bert-sentiment"):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    
    train_dataset = dataset_dict["train"].map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = dataset_dict["test"].map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = build_model(num_labels=len(label_encoder.classes_))

    training_args = TrainingArguments(
        output_dir=model_path,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{model_path}/logs",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer, tokenizer, model, test_dataset



def evaluate_model(trainer, test_dataset, label_encoder):
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    print("\n[INFO] Classification Report:")
    print(classification_report(labels, preds, target_names=label_encoder.classes_))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    print("\n[INFO] Confusion Matrix:")
    print(cm)
