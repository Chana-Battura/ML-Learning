from operator import ne
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import pickle
df = pd.read_csv("Sentdex/breast-cancer-wisconsin.data")
#Algorithms recognize -99999 as outlier
df.replace("?", -99999, inplace=True)

#Here we dont really need id to figure out
#the danger of cancer so remove ID
df.drop(["id"], axis=1, inplace=True)

X = np.array(df.drop(["class"], axis=1))
Y = np.array(df["class"])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)
'''
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
with open("Sentdex\KNN_BC.pickle", "wb") as f:
    pickle.dump(clf, f)'''

pickle_in = open("Sentdex\KNN_BC.pickle", "rb")
clf = pickle.load(pickle_in)
accuracy = clf.score(X_test, Y_test)
#print(accuracy)

example_measures = np.array([[3, 2, 1, 1, 1, 4, 2, 3, 2], [10, 2, 1, 1, 1, 4, 2, 3, 2]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)


print(prediction)

