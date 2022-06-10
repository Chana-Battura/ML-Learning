from sklearn import metrics, tree, neighbors, neural_network
#Descision Tree from Sklearn, checks for patterns
#via adding more and more nodes, eventually
#classfying it

#Height, Weight, Shoe Size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40]]

#Labels for Each of the Body Metrics from Above
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female']

#Descsion Tree Classifier
clf_tree = tree.DecisionTreeClassifier()

#K-Nearest Neighbors Classifier
clf_neighbors = neighbors.KNeighborsClassifier()

#MLP Classifier 
clf_MLP = neural_network.MLPClassifier()

#Training the classifiers with the data set
clf_tree = clf_tree.fit(X,Y)
clf_MLP = clf_MLP.fit(X,Y)
clf_neighbors = clf_neighbors.fit(X,Y)


#Using the classifier, here we score the 
#accuracy of the classifier based on its results
test = [[190,70,43], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
corr = ["male",'female', 'male', 'male']
acc_tree = metrics.accuracy_score(corr, clf_tree.predict(test))
acc_MLP = metrics.accuracy_score(corr, clf_MLP.predict(test))
acc_neighbors = metrics.accuracy_score(corr, clf_neighbors.predict(test))

print("Tree Accuracy: " + str(acc_tree))
print("MLP Accuracy: " + str(acc_MLP)) #Worst, Most Likely Due to Wrong Usage but Variable
print("KNN Accuracy: " + str(acc_neighbors))


