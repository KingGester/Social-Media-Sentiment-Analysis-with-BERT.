# 🧠 Social Media Sentiment Analysis with BERT

## 📌 Task Overview

تحلیل مجموعه داده‌ای از محتوای کاربران در شبکه‌های اجتماعی با هدف استخراج بینش‌های کلیدی:

### 🔍 موارد خواسته‌شده:

1. **تحلیل میزان مشارکت کاربران** با استفاده از ویژگی‌های `Retweets` و `Likes` برای شناسایی محتوای محبوب.
2. **تحلیل روند زمانی محتوا** با استفاده از ستون `Timestamp` برای کشف الگوها، نوسانات و مضامین تکرارشونده.
3. **تحلیل احساسات (Sentiment Analysis)** با استفاده از مدل‌های NLP و طبقه‌بندی احساسات در دسته‌هایی مانند:

   * Excitement
   * Contentment
   * Admiration
   * Surprise
   * Thrill
   * یا هر برچسب دیگر مرتبط با داده‌ها و مدل.

## 🧪 Solution Summary

در این پروژه، با تمرکز بر اجرای مدل از پیش آموزش‌دیده **BERT** (مدل `bert-base-uncased`) روی داده‌ها، تحلیل احساسات کاربران با دقت بالا انجام شد.

## 🧱 Project Structure

```
.
├── data /
│   ├── sentimentdataset.csv                # فایل اصلی دیتاست
│   ├── sentiment_predictions.csv           # خروجی BERT شامل احساسات پیش‌بینی‌شده
├── notebooks/
│   ├── 01_engagement_analysis.ipynb
│   ├── 02_time_analysis.ipynb
│   ├── 03_sentiment_analysis.ipynb 
│   ├── 04_transformer_sentiment.ipynb
├── src/
│   ├── utils.py
│   ├── time_analysis.py
│   ├── sentiment_analysis.py        
├── outputs/                              # نمودارهای تحلیلی
│   ├── Distribution of Predicted Sentiments by BERT.png
│   ├── plot_engagement_by_hour.png
│   ├── plot_engagement_by_month.png
│   ├── plot_engagement_by_weekday.png
│   ├── plot_engagement_distributions.png
│   ├── plot_posts_over_time.png
│   ├── plot_sentiment_distribution.png
└── README.md                           # توضیحات و مستندات پروژه
```

## ⚙️ Steps Performed

### 1. 📊 Exploratory Data Analysis (EDA)

* تحلیل آماری تعداد لایک و ریتوییت
* بررسی همبستگی بین محبوبیت محتوا و احساسات کاربران
* مصور سازی توزیع احساسات در داده‌ها

### 2. 🧼 Data Preprocessing

* پاکسازی متن (حذف URL، اعداد، نمادها)
* نرمال‌سازی و استانداردسازی برچسب‌های احساسی

### 3. 🤖 Sentiment Analysis with BERT

* استفاده از مدل `BertForSequenceClassification`
* طبقه‌بندی احساسات به سه دسته اصلی: **Positive**, **Negative**, **Neutral**
* اضافه کردن ستون `PredictedSentiment` به خروجی CSV

## 📈 Visualizations

### 📌 Distribution of Predicted Sentiments

![sentiment\_distribution](plots/sentiment_distribution.png)

### 📌 Likes and Retweets Distribution

![likes\_retweets\_distribution](plots/likes_retweets_distribution.png)

### 📌 Average Engagement by Sentiment

![engagement\_by\_sentiment](plots/engagement_by_sentiment.png)

## 📂 Output

فایل نهایی `sentiment_predictions.csv` شامل:

* متن کاربران
* برچسب اصلی (در صورت وجود)
* برچسب پیش‌بینی‌شده توسط BERT
* تعداد لایک و ریتوییت

## 🚀 Run the Code (Colab Friendly)

برای اجرای این پروژه در Colab:

```python
!pip install transformers scikit-learn
```

سپس:

```python
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd

# Load CSV
df = pd.read_csv("sentimentdataset.csv")

# Encode labels
df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentiment'])

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Inference
inputs = tokenizer(df['Text'].tolist(), return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    preds = torch.argmax(model(**inputs).logits, dim=1)

df['PredictedSentiment'] = label_encoder.inverse_transform(preds.numpy())
df.to_csv("sentiment_predictions.csv", index=False)
```

## 💡 Future Suggestions

* استفاده از مدل‌های پیشرفته‌تر مانند `RoBERTa` یا `DistilBERT`
* انجام تحلیل Topic Modeling برای درک بهتر محتوا
* افزودن ویژگی‌های زبانی یا تصویری در صورت وجود
* Fine-tune کردن مدل BERT برای دامنه خاص شبکه اجتماعی مورد بررسی

## 🙌 Team Note

از اعتماد شما متشکریم. این پروژه آماده ارائه به کارفرما یا استفاده در رزومه شماست.

**تهیه‌شده توسط:** سجاد پرچم

---

📁 برای دریافت خروجی، نمودارها و کدهای اجراشده، فایل‌ها را از مخزن پروژه دریافت نمایید یا خروجی نهایی `sentiment_predictions.csv` را بررسی فرمایید.
