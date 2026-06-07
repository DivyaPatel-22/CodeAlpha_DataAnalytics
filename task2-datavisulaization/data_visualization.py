import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("netflix_titles.csv")

# Data Cleaning
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Unknown')
df['director'] = df['director'].fillna('Unknown')

# Convert Date
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

sns.set_style("whitegrid")

# ==================================
# 1. BAR CHART
# Movies vs TV Shows
# ==================================
plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df)
plt.title("Movies vs TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.show()

# ==================================
# 2. PIE CHART
# Content Distribution
# ==================================
content = df['type'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(content,
        labels=content.index,
        autopct='%1.1f%%',
        startangle=90)

plt.title("Netflix Content Distribution")
plt.show()

# ==================================
# 3. LINE CHART
# Content Added Per Year
# ==================================
plt.figure(figsize=(10,5))

year_data = df['year_added'].value_counts().sort_index()

plt.plot(year_data.index,
         year_data.values,
         marker='o')

plt.title("Netflix Content Added Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.grid(True)

plt.show()

# ==================================
# 4. HEATMAP
# Correlation Analysis
# ==================================
df['release_year'] = pd.to_numeric(df['release_year'])

heatmap_df = df[['release_year','year_added']].dropna()

plt.figure(figsize=(6,4))

sns.heatmap(
    heatmap_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ==================================
# 5. TOP 10 COUNTRIES
# ==================================
plt.figure(figsize=(10,5))

df['country'].value_counts().head(10).plot(kind='bar')

plt.title("Top 10 Countries Producing Netflix Content")
plt.xlabel("Country")
plt.ylabel("Number of Titles")

plt.show()

# ==================================
# 6. TOP RATINGS
# ==================================
plt.figure(figsize=(10,5))

sns.countplot(
    y='rating',
    data=df,
    order=df['rating'].value_counts().index
)

plt.title("Content Ratings Distribution")

plt.show()

print("Netflix Data Visualization Dashboard Completed!")