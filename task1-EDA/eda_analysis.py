# CodeAlpha Task 1: Exploratory Data Analysis - Netflix Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("netflix_titles.csv")

# Display first rows
print("First 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Data information
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include="all"))

# Fill missing values
df['director'] = df['director'].fillna("Unknown")
df['cast'] = df['cast'].fillna("Unknown")
df['country'] = df['country'].fillna("Unknown")
df['rating'] = df['rating'].fillna("Unknown")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Movies vs TV Shows
plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df)
plt.title("Movies vs TV Shows")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# Top 10 countries
plt.figure(figsize=(10,5))
top_country = df['country'].value_counts().head(10)
top_country.plot(kind='bar')
plt.title("Top 10 Content Producing Countries")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.show()

# Release year trend
plt.figure(figsize=(10,5))
df['release_year'].value_counts().sort_index().plot(kind='line')
plt.title("Netflix Content Release Trend")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.grid()
plt.show()

# Ratings distribution
plt.figure(figsize=(10,5))
sns.countplot(y='rating', data=df, order=df['rating'].value_counts().index)
plt.title("Ratings Distribution")
plt.xlabel("Count")
plt.ylabel("Rating")
plt.show()

# Top 10 genres
plt.figure(figsize=(10,5))
genres = df['listed_in'].str.split(', ', expand=True).stack()
genres.value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Content added by year
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

plt.figure(figsize=(10,5))
df['year_added'].value_counts().sort_index().plot(kind='bar')
plt.title("Content Added to Netflix by Year")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
plt.show()

# Key Insights
print("\nKEY INSIGHTS:")
print("1. Netflix has more Movies compared to TV Shows.")
print("2. United States produces the highest number of Netflix titles.")
print("3. Netflix content increased significantly after 2015.")
print("4. TV-MA is one of the most common ratings.")
print("5. Dramas, International Movies, and Comedies are common genres.")

print("\nEDA Completed Successfully!")