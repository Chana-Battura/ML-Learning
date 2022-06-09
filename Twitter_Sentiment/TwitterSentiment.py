import tweepy
from textblob import TextBlob
import csv

consumer_key = "_____KEY_____"
consumer_secret = "_____KEY_____"

access_token = "_____KEY_____"
access_token_secret = "_____KEY_____"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.Client("AAAAAAAAAAAAAAAAAAAAAITodQEAAAAA3R1b2fnWyUG98I0AOtx5KFW1gJo%3DCzSlgSIdt8EkDWdnFIkKOl2CaAcKV2Kd9rgnZ8n9flUD155JVC", auth)

f = open("tweets.csv", "w", encoding="UTF8")
writer = csv.writer(f)
writer.writerow(["Tweet", "Sentiment"])

public_tweets = api.search_recent_tweets("from:elonmusk")
for tweet in public_tweets.data:
    analysis = TextBlob(tweet.text)
    sentiment = "Positive"
    if analysis.sentiment.polarity == 0.0:
        sentiment = "Neutral"
    elif analysis.sentiment.polarity < 0:
        sentiment = "Negative"
    writer.writerow([tweet.text, sentiment])
    print(analysis.sentiment)

f.close()