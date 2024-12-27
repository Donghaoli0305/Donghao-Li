import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Find all CSV files in the same folder as the script
csv_files = [os.path.join(script_dir, file) for file in os.listdir(script_dir) if file.endswith('.csv')]

# Load datasets
datasets = [pd.read_csv(file) for file in csv_files]

# Concatenate all datasets into one DataFrame
data = pd.concat(datasets, ignore_index=True)

# Focus on the 'original_text' and 'place' columns
tweets = data[['original_text', 'place']].dropna()

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

# Pandemic-related keywords
pandemic_keywords = {'pandemic', 'virus', 'lockdown', 'quarantine', 'infection', 
                     'covid', 'coronavirus', 'outbreak', 'hospital', 'death'}

# Function to determine if a tweet has high "pandemic level"
def is_high_pandemic_level(text):
    tokens = preprocess_text(text)
    return any(word in pandemic_keywords for word in tokens)

# Apply pandemic-level detection and sentiment analysis
tweets['High_Pandemic_Level'] = tweets['original_text'].apply(is_high_pandemic_level)

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
tweets['Sentiment'] = tweets['original_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
tweets['Negative_Sentiment'] = tweets['Sentiment'] < -0.05

# Aggregating pandemic level and sentiment by region
region_pandemic_level = tweets.groupby('place').agg(
    Pandemic_Level_Count=('High_Pandemic_Level', 'sum'),
    Total_Tweets=('original_text', 'count'),
    Negative_Sentiment_Count=('Negative_Sentiment', 'sum')
)

# Calculate percentage of high pandemic level tweets and negative sentiment
region_pandemic_level['Pandemic_Level_Percentage'] = (
    region_pandemic_level['Pandemic_Level_Count'] / region_pandemic_level['Total_Tweets'] * 100
)
region_pandemic_level['Negative_Sentiment_Percentage'] = (
    region_pandemic_level['Negative_Sentiment_Count'] / region_pandemic_level['Total_Tweets'] * 100
)

# Flag regions with high pandemic level
region_pandemic_level['Pandemic_Flag'] = region_pandemic_level['Pandemic_Level_Percentage'] > 50

# Get top 10 regions with the highest pandemic level percentage
top_10_places = region_pandemic_level.nlargest(10, 'Pandemic_Level_Percentage')

# Save the top 10 results to a CSV
output_file = os.path.join(script_dir, "Top_10_Pandemic_Regions.csv")
top_10_places.to_csv(output_file)

# Display the top 10 regions
print("Top 10 Places with the Highest Pandemic Level:\n")
print(top_10_places)

print(f"\nTop 10 results saved to {output_file}")
