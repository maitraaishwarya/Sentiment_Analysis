import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')

# Load the dataset from CSV file
data = pd.read_csv('Facebook1.csv')

# Extract the text column from the dataset
texts = data['text']

# Create an instance of the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each text
for text in texts:
    sentiment_scores = sid.polarity_scores(text)
    print(f"Text: {text}")
    print(f"Sentiment Scores: {sentiment_scores}")
    print("Sentiment: ", end="")
    if sentiment_scores['compound'] >= 0.05:
        print("Positive")
    elif sentiment_scores['compound'] <= -0.05:
        print("Negative")
    else:
        print("Neutral")
    print()

# Perform sentiment analysis on each text
sentiment_scores = []

for text in texts:
    sentiment_score = sid.polarity_scores(text)['compound']
    sentiment_scores.append(sentiment_score)


# Plot the sentiment scores - Histogram
plt.figure(figsize=(8, 6))
plt.hist(sentiment_scores, bins=10, edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis - Histogram')
plt.show()

# Calculate sentiment distribution for pie chart
positive_count = sum(score >= 0.05 for score in sentiment_scores)
negative_count = sum(score <= -0.05 for score in sentiment_scores)
neutral_count = len(sentiment_scores) - positive_count - negative_count

# Plot the sentiment distribution - Pie chart
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_count, negative_count, neutral_count]
colors = ['green', 'red', 'gray']
explode = (0.1, 0.1, 0)  # Explode the positive and negative slices
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
plt.title('Sentiment Analysis - Pie Chart')
plt.axis('equal')
plt.show()