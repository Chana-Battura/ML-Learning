import csv
import math
from operator import neg
from statistics import mode
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tweepy
from textblob import TextBlob

dates = []
prices = []
data = []

def get_data(filename):
    global data
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dates.append(np.datetime64(row[0]))
            prices.append(float(row[1]))
    data = [list(a) for a in zip(dates, prices)]
    f.close()

def plot_price():
    print(data)
    x = np.array(dates)
    y = np.array(prices)
    plt.plot(x, y, color="black", label="DATA")
    plt.xlabel("DATE")
    plt.ylabel("PRICE")
    plt.title("MAY STOCK: TSLA")

def create_database(data, l=1):
    dataX, dataY = [], []
    for i in range(len(data)-l-1):
        a = data[i+l][1]
        dataX.append(a)
        dataY.append(data[i+l+1][1])
    return (np.array(dataX), np.array(dataY))


def neural_prediction():
    train_size = int(len(data) * 0.67)
    test_size = len(data) - train_size
    train, test = data[0:train_size], data[train_size:len(data)]
    trainX, trainY = create_database(train)
    testX, testY = create_database(test)
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=400, verbose=1)

    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredictPlot = np.empty_like(data)
    trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(1*2)+1:len(data)-1, :] = testPredict
    plt.plot(data, color = "red")
    plt.plot(trainPredict, color = "blue")
    plt.plot(testPredict, color = "green")
    plot_price()
    plt.show()

def sentiment_calculate():
    positive = 0
    negative = 0
    consumer_key = "_____KEY_____"
    consumer_secret = "_____KEY_____"

    access_token = "_____KEY_____"
    access_token_secret = "_____KEY_____"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.Client("_____KEY_____", auth)

    f = open("tweets.csv", "w", encoding="UTF8")
    writer = csv.writer(f)
    writer.writerow(["Tweet", "Sentiment"])

    public_tweets = api.search_recent_tweets("TSLA")
    for tweet in public_tweets.data:
        analysis = TextBlob(tweet.text)
        if analysis.sentiment.polarity > 0.0:
            positive += 1
        elif analysis.sentiment.polarity != 0:
            negative += 1
    f.close()
    
    return(positive>negative)
get_data("data/TSLA.csv")
neural_prediction()
