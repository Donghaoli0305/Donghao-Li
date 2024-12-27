import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find all CSV files in the same folder as the script
csv_files = [os.path.join(script_dir, file) for file in os.listdir(script_dir) if file.endswith('.csv')]

# Load datasets
datasets = [pd.read_csv(file) for file in csv_files]

# Concatenate all datasets into one DataFrame
data = pd.concat(datasets, ignore_index=True)

# Preprocess 'created_at' column to ensure it's in datetime format
data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')

# Remove rows with invalid or missing dates
data = data.dropna(subset=['created_at'])

# Focus on the 'original_text' column
tweets = data[['created_at', 'original_text']].dropna()

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Preprocess tweets and group by week
tweets['tokens'] = tweets['original_text'].apply(preprocess_text)
tweets['week'] = tweets['created_at'].dt.to_period('W')

# Analyze common words and emotions over time
weekly_analysis = {}
for period, group in tweets.groupby('week'):
    # Analyze common words
    all_tokens = [token for tokens in group['tokens'] for token in tokens]
    fdist = FreqDist(all_tokens)
    common_words = fdist.most_common(10)

    # Analyze emotions
    sentiment_scores = group['original_text'].apply(sia.polarity_scores)
    emotion_summary = {
        'positive': sum(1 for score in sentiment_scores if score['compound'] > 0.05),
        'negative': sum(1 for score in sentiment_scores if score['compound'] < -0.05),
        'neutral': sum(1 for score in sentiment_scores if -0.05 <= score['compound'] <= 0.05)
    }
    
    # Store results
    weekly_analysis[period] = {
        'common_words': common_words,
        'emotion_summary': emotion_summary
    }

# Display results
for week, analysis in weekly_analysis.items():
    print(f"Week {week}:")
    print("  Most Common Words:")
    for word, count in analysis['common_words']:
        print(f"    {word}: {count}")
    print("  Emotion Summary:")
    print(f"    Positive: {analysis['emotion_summary']['positive']}")
    print(f"    Negative: {analysis['emotion_summary']['negative']}")
    print(f"    Neutral: {analysis['emotion_summary']['neutral']}")
    print()
