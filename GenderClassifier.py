from sklearn import tree
#Descision Tree from Sklearn, checks for patterns
#via adding more and more nodes, eventually
#classfying it

#Height, Weight, Shoe Size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#Labels for Each of the Body Metrics from Above
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#Descsion Tree Classifier
clf = tree.DecisionTreeClassifier()

#Trains the tree with the data set
clf = clf.fit(X,Y)

#Using the classifier, the tree predicts the 
#gender based on these values
prediction = clf.predict([[190,70,43]])
print(prediction[0])

