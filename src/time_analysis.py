
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_time_features(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['Weekday'] = df['Timestamp'].dt.day_name()
    return df

def plot_posts_over_time(df):
    hourly_counts = df['Hour'].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette="Blues_d")
    plt.title("Number of Posts per Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Posts")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.show()

def plot_engagement_by_hour(df):
    hourly_engagement = df.groupby('Hour')[['Likes', 'Retweets']].mean()
    plt.figure(figsize=(10, 5))
    hourly_engagement.plot(kind='line', marker='o')
    plt.title("Average Engagement by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Likes / Retweets")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_engagement_by_weekday(df):
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_engagement = df.groupby('Weekday')[['Likes', 'Retweets']].mean().reindex(weekday_order)
    weekday_engagement.plot(kind='bar', figsize=(10, 5))
    plt.title("Average Engagement by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Likes / Retweets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_engagement_by_month(df):
    monthly_engagement = df.groupby('Month')[['Likes', 'Retweets']].mean()
    monthly_engagement.plot(kind='bar', figsize=(10, 5))
    plt.title("Average Engagement by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Likes / Retweets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()