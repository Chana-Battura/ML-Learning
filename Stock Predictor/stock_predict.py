import csv
from operator import neg
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tweepy
from textblob import TextBlob

dates = []
prices = []

def get_data(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dates.append(int(row[0].split("-")[2]))
            prices.append(float(row[1]))

def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel="linear", C=1e3)
    svr_poly = SVR(kernel="poly", C=1e3, degree = 2)
    svr_rbf = SVR(kernel = "rbf", C=1e3, gamma=0.1)

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color="black", label="DATA")
    plt.plot(dates, svr_rbf.predict(dates), color="red", label = "RBF")
    plt.plot(dates, svr_lin.predict(dates), color="green", label = "LINEAR")
    plt.plot(dates, svr_poly.predict(dates), color="blue", label = "POLYNOMIAL")

    plt.xlabel("DATE")
    plt.ylabel("PRICE")
    plt.title("SVR")
    plt.legend()
    plt.show()

    return svr_rbf.predict(np.array([x]).reshape(1,1))


get_data("data/TSLA.csv")

predicted_price = predict_price(dates, prices, 10)

print(predicted_price)