import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

df = pd.read_csv('twitter_training.csv')

def preprocess_tweets(tweets):
    processed_tweets = []
    for tweet in tweets:
        tweet = re.sub(r'[^\w\s]', '', tweet)
        tweet = re.sub(r'#\w+', '', tweet)
        tweet = re.sub(r'@\w+', '', tweet)
        processed_tweets.append(tweet)
    return processed_tweets

def analyze_sentiment(tweets):
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for tweet in tweets:
        words = [word.lower() for word in word_tokenize(tweet) if word.isalpha()]
        words = [word for word in words if word not in stopwords.words('english')]
        
        sentiment_score = sia.polarity_scores(' '.join(words))['compound']
        
        if sentiment_score >= 0.05:
            sentiment = 'positive'
        elif sentiment_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        sentiments.append(sentiment)
    return sentiments

def plot_sentiment_distribution(sentiments):
    plt.figure(figsize=(8, 6))
    

    bins = [-0.5, 0.5, 1.5, 2.5]  
    
    plt.hist(sentiments, bins=bins, color='skyblue', align='left', rwidth=0.8)
 
    bin_labels = ['negative', 'neutral', 'positive']
    plt.xticks(bins[:-1], bin_labels)
    
    plt.title('Sentiment Distribution of Tweets')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.show()


def main():
    
    print(df.columns)
    
   
    tweets = df.iloc[:, 1].tolist()
    
    processed_tweets = preprocess_tweets(tweets)
 
    sentiments = analyze_sentiment(processed_tweets)
   
    plot_sentiment_distribution(sentiments)

if name == "main":
    main()
