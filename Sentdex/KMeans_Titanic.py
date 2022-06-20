import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

'''
    ###################################
    We will test to see whether the 
    algorithm is able to cluster the data
    to survival.
    ###################################

    PClass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
    survived: Survival (0 = No; 1 = Yes)
    name: Name ---This most likely wont be used in the algorithm---
    sex Sex
    age Age
    sibsp Number of Siblings/Spouses Aboard
    parch Number of Parent/Children Aboard
    ticket Ticket Number
    fare Passenger Fare (British Pound)
    cabin Cabin
    embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
    boat Lifeboat
    body Body Identification Number
    home.dest Home/Destination
'''

df = pd.read_excel("Sentdex/titanic.xls")
#print(df.head())
df.drop(['body', 'name'], axis=1, inplace=True) #Drops the body and name columns
df.fillna(0, inplace=True) #Fills the NaN values with 0

def handle_non_numerical_data(df):
    columns = df.columns.values

    for col in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[col].dtype != np.int64 and df[col].dtype !=np.float64:
            column_contents = df[col].values.tolist() #Converts the column to a list
            unique_elements = set(column_contents) #Gets the unique elements in the column
            x = 0 #Counter
            for unique in unique_elements: #Iterates through the unique elements
                if unique not in text_digit_vals: #If the unique element is not in the dictionary
                    text_digit_vals[unique] = x #Adds the unique element to the dictionary with the counter value
                    x+=1                      #Increments the counter
            df[col] = list(map(convert_to_int, df[col])) #Maps the column to the dictionary
    return df

df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived', 'fare'], axis=1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df["survived"])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct += 1

print("Accuracy: ", correct/len(X))