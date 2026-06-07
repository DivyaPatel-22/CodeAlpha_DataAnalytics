import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re

# Load Netflix Dataset
df = pd.read_csv("netflix_titles.csv")

# Keep only descriptions
df = df[['title', 'type', 'country', 'release_year', 'rating', 'description']]

# Remove missing descriptions
df.dropna(subset=['description'], inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_description'] = df['description'].apply(clean_text)

# Calculate Sentiment Polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['polarity'] = df['clean_description'].apply(get_sentiment)

# Classify Sentiment
def classify_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['polarity'].apply(classify_sentiment)

# Display results
print(df[['title', 'sentiment']].head())

# -------------------------
# Chart 1: Sentiment Count
# -------------------------
plt.figure(figsize=(7,5))
sns.countplot(x='sentiment', data=df)
plt.title("Netflix Content Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# -------------------------
# Chart 2: Pie Chart
# -------------------------
sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(7,7))
plt.pie(sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=90)

plt.title("Netflix Content Sentiment Percentage")
plt.show()

# -------------------------
# Chart 3: Polarity Histogram
# -------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['polarity'], bins=30, kde=True)

plt.title("Sentiment Polarity Distribution")
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.show()

# -------------------------
# Chart 4: Sentiment by Type
# -------------------------
plt.figure(figsize=(8,5))
sns.countplot(x='type', hue='sentiment', data=df)

plt.title("Sentiment by Content Type")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# Save Results
df.to_csv("Netflix_Sentiment_Output.csv", index=False)

print("Sentiment Analysis Completed Successfully!")