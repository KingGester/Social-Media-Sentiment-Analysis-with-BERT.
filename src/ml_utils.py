from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np


def vectorize_texts(texts, max_features=10000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


# Logistic Regression
def train_classical_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_classical_model(model, X_test_vec, y_test, label_encoder):
    y_pred = model.predict(X_test_vec)

 
    labels_in_test = np.unique(y_test)
    class_names = label_encoder.inverse_transform(labels_in_test)

    print("\n[INFO] Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels_in_test,
        target_names=class_names,
        zero_division=0  
    ))

    print("\n[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels_in_test))


def prepare_classical_data(df, text_col="Text", label_col="Sentiment", test_size=0.2):
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[label_col])

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df["label"],
        test_size=test_size,
        random_state=42,
        stratify=df["label"]
    )

    return X_train, X_test, y_train, y_test, label_encoder