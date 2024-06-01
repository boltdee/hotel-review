import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/mnt/data/b_fixed.csv')

# Function to clean the text
def clean_text(text):
    text = re.sub(r'\n', ' ', text)  # Remove new lines
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

# Apply the cleaning function to the text column
df['cleaned_text'] = df['text_column'].apply(clean_text)  # Replace 'text_column' with the actual column name

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get the sentiment scores
def get_sentiment_scores(text):
    return analyzer.polarity_scores(text)

# Apply the function to get sentiment scores
df['sentiment'] = df['cleaned_text'].apply(get_sentiment_scores)

# Extract the compound score for simplicity
df['compound'] = df['sentiment'].apply(lambda score_dict: score_dict['compound'])

# Set the title of the Streamlit app
st.title('Sentiment Analysis on Text Data')

# Display the dataset
st.write(df.head())

# Plot the sentiment distribution
st.subheader('Sentiment Distribution')
fig, ax = plt.subplots()
df['compound'].hist(bins=20, ax=ax)
ax.set_title('Sentiment Distribution')
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Display the average sentiment score
average_sentiment = df['compound'].mean()
st.subheader('Average Sentiment Score')
st.write(average_sentiment)
