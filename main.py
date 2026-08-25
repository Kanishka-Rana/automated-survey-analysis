import pandas as pd
import matplotlib.pyplot 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
file_path = "DATA/automated_survey_analysis_dataset.csv"

df = pd.read_csv(file_path)

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nRating Summary:")
print(df["Rating"].describe())

print("\nAverage Rating:")
print(df["Rating"].mean())

print("\nCategory Counts:")
print(df["Category"].value_counts())

print("\nSample Customer Feedback:")
print(df["Feedback"].head(10))

# . SENTIMENT ANALYSIS

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return "Neutral"

    score = sia.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["Sentiment"] = df["Feedback"].apply(analyze_sentiment)

print("\n===== SENTIMENT RESULTS =====")
print(df[["Feedback", "Sentiment"]].head(10))

print("\n===== SENTIMENT COUNTS =====")
print(df["Sentiment"].value_counts())

# ==============================
#  BUSINESS INSIGHTS
# ==============================

total_responses = len(df)

positive_count = (df["Sentiment"] == "Positive").sum()
negative_count = (df["Sentiment"] == "Negative").sum()
neutral_count = (df["Sentiment"] == "Neutral").sum()

positive_percentage = (positive_count / total_responses) * 100
negative_percentage = (negative_count / total_responses) * 100
neutral_percentage = (neutral_count / total_responses) * 100

print("\n===== BUSINESS INSIGHTS =====")

print("Total Responses:", total_responses)

print(f"Positive Feedback: {positive_count} ({positive_percentage:.1f}%)")
print(f"Neutral Feedback: {neutral_count} ({neutral_percentage:.1f}%)")
print(f"Negative Feedback: {negative_count} ({negative_percentage:.1f}%)")

print(f"Average Customer Rating: {df['Rating'].mean():.2f}")

# Most common category
most_common_category = df["Category"].value_counts().idxmax()

print("Most Common Feedback Category:", most_common_category)

# Category with most negative feedback
negative_feedback = df[df["Sentiment"] == "Negative"]

if len(negative_feedback) > 0:
    most_negative_category = negative_feedback["Category"].value_counts().idxmax()
    print("Category With Most Negative Feedback:", most_negative_category)

# Overall sentiment
sentiment_counts = df["Sentiment"].value_counts()

overall_sentiment = sentiment_counts.idxmax()

print("Overall Customer Sentiment:", overall_sentiment)

# ==============================
# 8. DATA VISUALIZATION
# ==============================

import matplotlib.pyplot as plt

# 1. Sentiment Distribution
sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(7, 5))
sentiment_counts.plot(kind="bar")
plt.title("Customer Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Responses")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 2. Rating Distribution
plt.figure(figsize=(7, 5))
df["Rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Responses")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 3. Category Distribution
plt.figure(figsize=(8, 5))
df["Category"].value_counts().plot(kind="bar")
plt.title("Feedback by Category")
plt.xlabel("Category")
plt.ylabel("Number of Responses")
plt.xticks(rotation=45)
plt.tight_layout()


print("\n===== NEXT STEP STARTED =====")
print("Code is running after Business Insights")
#  CATEGORY-WISE SENTIMENT

category_sentiment = pd.crosstab(
    df["Category"],
    df["Sentiment"]
)

print("\n===== CATEGORY-WISE SENTIMENT =====")
print(category_sentiment)

# Negative feedback by category
negative_by_category = (
    df[df["Sentiment"] == "Negative"]["Category"]
    .value_counts()
)

print("\n===== NEGATIVE FEEDBACK BY CATEGORY =====")
print(negative_by_category)

# ==============================
#  RATING VS SENTIMENT ANALYSIS
# ==============================

rating_sentiment = pd.crosstab(df["Rating"],df["Sentiment"])

print("\n===== RATING VS SENTIMENT =====")
print(rating_sentiment)

# Average rating for each sentiment
average_rating_by_sentiment = df.groupby("Sentiment")["Rating"].mean()

print("\n===== AVERAGE RATING BY SENTIMENT =====")
print(average_rating_by_sentiment)

# 11. CATEGORY-WISE AVERAGE RATING
# ==========================================

average_rating_by_category = df.groupby("Category")["Rating"].mean().sort_values(ascending=False)

print("\n===== CATEGORY-WISE AVERAGE RATING =====")
print(average_rating_by_category)


# ==========================================
# 12. CATEGORY-WISE AVERAGE RATING CHART
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
average_rating_by_category.plot(kind="bar")

plt.title("Average Customer Rating by Category")
plt.xlabel("Category")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("category_average_rating.png")



# ==========================================
# 13. TOP NEGATIVE FEEDBACK
# ==========================================

negative_feedback = df[df["Sentiment"] == "Negative"]

print("\n===== TOP NEGATIVE FEEDBACK =====")
print(negative_feedback[["Category", "Rating", "Feedback"]].head(10))
