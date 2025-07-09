from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns


def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))  # remove URLs
    text = re.sub(r"[^\w\s]", "", text)      # remove punctuation
    text = re.sub(r"\d+", "", text)           # remove numbers
    text = text.lower().strip()
    return text

def preprocess_df(df, text_col="Text"):
    df[text_col] = df[text_col].apply(clean_text)
    return df


def normalize_label(label):
    return str(label).lower().strip()

def map_to_emotion_category(label):
    label = normalize_label(label)

    positive = [
        "joy", "admiration", "excitement", "inspiration", "inspired",
        "serenity", "elation", "euphoria", "fulfillment", "grateful", "happy",
        "proud", "contentment", "satisfaction", "vibrancy", "zest"
    ]
    negative = [
        "anger", "disgust", "sadness", "grief", "frustration", "hate",
        "heartbreak", "devastated", "loneliness", "bitter", "regret",
        "betrayal", "fear", "overwhelmed", "desolation", "isolation"
    ]
    neutral = [
        "curiosity", "reflection", "contemplation", "calmness", "neutral",
        "connection", "coziness", "serenity", "elegance", "stillness"
    ]
    mixed = [
        "bittersweet", "ambivalence", "nostalgia", "melancholy", "confusion",
        "envious", "lostlove", "numbness", "miscalculation"
    ]
    
    if label in positive:
        return "positive"
    elif label in negative:
        return "negative"
    elif label in neutral:
        return "neutral"
    elif label in mixed:
        return "mixed"
    else:
        return "other"

def simplify_sentiment_labels(df, label_col="Sentiment"):
    df[label_col] = df[label_col].apply(map_to_emotion_category)
    return df


def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_model(X, y):
    model = LogisticRegression(max_iter=10,multi_class='multinomial',class_weight='balanced', solver='lbfgs',random_state=10)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, preds))
    print("\n[INFO] Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def predict_sentiment(model, vectorizer, texts):
    texts_clean = [clean_text(t) for t in texts]
    X = vectorizer.transform(texts_clean)
    return model.predict(X)


def plot_sentiment_distribution(y):
    plt.figure(figsize=(6, 6))
    y.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    plt.title("Sentiment Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()