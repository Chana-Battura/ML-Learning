import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
style.use('ggplot')
import numpy as np
import pandas as pd 

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11], 
              [1, 3]])
'''
plt.scatter(X[:, 0], X[:, 1], s=150)
plt.show()'''


colors = ["r", "g", "b", "c", "m", "y"]

class K_Means:

    def __init__(self, k = 2, tol = 0.001, max_iter = 300) -> None:
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]
        
        for i in range(self.max_iter):
            self.classifications = {} #centroids and classifications

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data: #iterate through all the data
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids] #calculate the distance from each centroid
                classification = distances.index(min(distances)) #find the closest centroid
                self.classifications[classification].append(featureset) #add the data to the classification

            prev_centroids = dict(self.centroids) #store the previous centroids

            for classification in self.classifications: #iterate through all the classifications
                self.centroids[classification] = np.average(self.classifications[classification], axis=0) #calculate the new centroid

            optimized = True #optimized is true if the centroids have not changed

            for c in self.centroids: #iterate through all the centroids
                original_centroid = prev_centroids[c] #get the original centroid
                current_centroid = self.centroids[c] #get the current centroid
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol: #if the centroids have changed
                    optimized = False #set optimized to false

            if optimized: #if optimized is true
                break #break out of the loop

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids] #calculate the distance from each centroid
        classification = distances.index(min(distances)) #find the closest centroid
        return classification #return the classification
    
'''for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="X", c=color, s=150, linewidths=2)
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

clf = K_Means()
clf.fit(X)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == Y[i]:
        correct += 1
print("Accuracy: ", correct/len(X))