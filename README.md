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

### 1. Analysis of user participation
* آمار از بیشترین میانگین کمترین (likesوRetweets)
* برسی ترند های کلی
* آیا تعامل در حال افزایش است ؟ یا الگوی خاصی دارد؟

### 2.Check the time trend
* رسم نمودار زمانی برای تعداد پست ها ,لایک ها و ریتیوت ها در بازده ها مختلف مقلا روزانه هفتگی
* تحلیل دورهای مثلا اخر هفته ها تعامل بیشتر است ؟

### 3. 📊 Exploratory Data Analysis (EDA)

* تحلیل آماری تعداد لایک و ریتوییت
* بررسی همبستگی بین محبوبیت محتوا و احساسات کاربران
* مصور سازی توزیع احساسات در داده‌ها

### 4.  Data Preprocessing

* پیش پردازش متن برای مدل کردن
* نرمال سازی متن و علایم نگارشی

### 5.Classical machine learning model
* استفاده از الگوریتم ماشین لرنینگ LogisticRegression
* 🧼و استفاده ازش برای تحلیل احساس متن کاربران
## Classification Report:

              precision    recall  f1-score   support

       mixed       0.86      0.75      0.80         8
    negative       0.85      0.73      0.79        15
     neutral       0.80      0.44      0.57         9
    positive       0.74      0.89      0.81        36

    accuracy                           0.78        68
   macro avg       0.81      0.70      0.74        68
weighted avg       0.79      0.78      0.77        68

### 6. 🤖 Sentiment Analysis with BERT

* استفاده از مدل `BertForSequenceClassification`
* طبقه‌بندی احساسات به سه دسته اصلی: **Positive**, **Negative**, **Neutral**
* اضافه کردن ستون `PredictedSentiment` به خروجی CSV
* طبقه بندی احساسات 
* تحلیل اینکه چه احساسی بیتشتر دیده می شود و آیا زمان یا میزان تعامل رابطه ای دارند یا نه ؟

## 📈 Visualizations


### 📌 Likes and Retweets Distribution

![likes\_retweets\_distribution](https://github.com/KingGester/Social-Media-Sentiment-Analysis-with-BERT./blob/main/outputs/plot_engagement_distributions.png)


### 📌 Engagement By Weekday


![Engagement By Weekday](https://github.com/KingGester/Social-Media-Sentiment-Analysis-with-BERT./blob/main/outputs/plot_engagement_by_weekday.png)

### Engagement By Month

![Engagement By Month.png](https://github.com/KingGester/Social-Media-Sentiment-Analysis-with-BERT./blob/main/outputs/plot_engagement_by_month.png)

### Engagement By Hours
![Engagement By Hours](https://github.com/KingGester/Social-Media-Sentiment-Analysis-with-BERT./blob/main/outputs/plot_engagement_by_hour.png)

### Posts Over Time
![Posts Over Time](https://github.com/KingGester/Social-Media-Sentiment-Analysis-with-BERT./blob/main/outputs/plot_posts_over_time.png)

### Social-Media-Sentiment-Analysis-with-BERT.
![Social-Media-Sentiment-Analysis-with-BERT.](https://github.com/KingGester/Social-Media-Sentiment-Analysis-with-BERT./blob/main/outputs/Distribution%20of%20Predicted%20Sentiments%20by%20BERT.png)

## 📂 data

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
