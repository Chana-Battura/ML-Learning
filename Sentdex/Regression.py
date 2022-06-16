##############################################################
# Regression: Find best fit line -- Model data with function #
# Example: Linear Regression looks for liner line to model   #
# find what m and b is, example with stock prices prediction #
# Since Stock prices are continuous.                         #  
# Features: attributes, labels: values                       #
##############################################################

import math
import pandas as pd
import quandl, datetime
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')

df = quandl.get("WIKI/TSLA")

##############################################################
# Not all features are necessary for pattern recognition     #
# - Adjusted means adjusted with stock split compensating    #
# Look for relationship between the various features to pick #
# For example, we picked these features                      #
##############################################################

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume" ]]

##############################################################
# Added new coloumn called percent change for volatility     #
##############################################################

df["HL_PCT"] = (df["Adj. High"]-df["Adj. Low"]) / df["Adj. Close"] * 100.0

##############################################################
# Added new coloumn called daily change with new minus the   #
# old all divided by the old price                           #
##############################################################

df["PCT_Change"] = (df["Adj. Close"]-df["Adj. Open"]) / df["Adj. Open"] * 100.0


##############################################################
# new dataframe with colloumns we care about                 # 
# Our label will be the price of stock not adj. close as     #
# the other values require it to be a feature to find them   #
##############################################################

# Features We Chose Must Impact Price
# Thus      Price           X            X           X
#So honestly, just adjusted close is truly the only needed feature
df = df[["Adj. Close", "Adj. Volume", "HL_PCT", "PCT_Change"]]


forecast_col = "Adj. Close" #predict next adj. close as that is the price
df.fillna(-99999, inplace=True) #This just will be seen as outlier better than deleting col

forecast_out = int(math.ceil(0.10*len(df))) #Predicting ten percent of the dataframe
print(forecast_out)

df["Label"] = df[forecast_col].shift(-forecast_out) #each label will be adjusted close price of ten percent days later


#Featurs and labels: X is features, Y is labels
X = np.array(df.drop(["Label"],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True) #Removing the NaN values
Y = np.array(df["Label"])

X_train, X_test, Y_train, Y_Test = model_selection.train_test_split(X, Y, test_size=0.1 )

'''clf = LinearRegression(n_jobs=-1) #n_jobs=-1 means it will use all the threads
clf.fit(X_train, Y_train)

with open("Sentdex\linearregression.pickle", "wb") as f:
    pickle.dump(clf, f)
'''
pickle_in = open("Sentdex/linearregression.pickle", "rb")
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, Y_Test)
#print(accuracy)

forecast_set = clf.predict(X_lately) #Predicting the future values from forecast_out shift
df["Forecast"] = np.nan


last_date = df.iloc[-1].name #Last date in the dataframe
last_unix = last_date.timestamp() #Last unix time
one_day = 86400 #seconds in a day
next_unix = last_unix + one_day #Next unix time

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix) #Next date
    next_unix += one_day #Next unix time
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #Adding the forecasted value to the dataframe

df["Adj. Close"].plot() #Plotting the graph
df["Forecast"].plot() #Plotting the graph
plt.legend(loc=4) #Legend location
plt.xlabel("Date") #X label
plt.ylabel("Price") #Y label
plt.show()
