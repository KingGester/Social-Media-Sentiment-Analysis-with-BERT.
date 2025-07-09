import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def load_dataet(path):
    return pd.read_csv(path)

def basic_info(df):
    print("\n[INFO] Dataset Info: ")
    print(df.info())
    print("\n[INFO] First 5 Rows:")
    print(df.head())

def engagement_summary(df):
    print("\n[INFO] Engagement Summary: ")
    print("Likes - Mean: {:.2f}, Max: {}, Min: {}".format(df["Likes"].mean(), df["Likes"].max(), df["Likes"].min()))
    print("Retweets - Mean: {:.2f}, Max: {}, Min: {}".format(df['Retweets'].mean(), df['Retweets'].max(), df['Retweets'].min()))

def plot_engagement_distributions(df):

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Likes'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Likes')

    plt.subplot(1, 2, 2)
    sns.histplot(df['Retweets'], bins=20, kde=True, color='orange')
    plt.title('Distribution of Retweets')

    plt.tight_layout()
    plt.show()

def top_engaged_posts(df, top_n=5):

    df['Total_Engagement'] = df['Likes'] + df['Retweets']
    top_posts = df.sort_values(by='Total_Engagement', ascending=False).head(top_n)
    print("\n[INFO] Top {} Engaged Posts:".format(top_n))
    print(top_posts[['Text', 'Likes', 'Retweets', 'Total_Engagement']])