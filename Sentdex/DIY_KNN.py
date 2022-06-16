#######################################################################
# KNN needs a distance calculation, which we use Euclidean Distance   #
# Euclidean Distance ==>                                              #
# Square Root(Sigma of i = 1 to number of dimensions of (Qi-Pi)^2     #
# Example, q = (1,3) and p = (2,5) ===> Sqrt((1-2)^2 + (3-5)^2))      #
#######################################################################
import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data ,predict, k = 3):
    if len(data) >= k:
        warnings.warn("ERROR: K is set to value less that total voting groups!")
    
    #We have to compare to all points, making it harder
    #We could do radius, but like still need to check with others

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence

df = pd.read_csv("Sentdex/breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace=True)
df.drop(["id"], axis=1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        else:
            pass
            print(confidence)
        total += 1

print("Accuracy:" ,(correct/total))

#######################################################################
# Confidence Vs. Accuracy                                             #
# How confident you are with the cancer, commpared to how accurate    #
#######################################################################
