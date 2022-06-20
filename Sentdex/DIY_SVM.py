import pstats
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from pandas import NA
from regex import P

style.use("ggplot")

class SVM():
    def __init__(self, visualization=True):
        #Visualization calls for graphing data
        self.visualization = visualization
        self.colors = {1:"r", -1:"b"}
        if self.visualization:
            self.fig = plt.figure()
            self.axis = self.fig.add_subplot(1,1,1)
            #We declared the matplot graph space
    
    #training with data
    def fit(self, data):
        self.data = data
        opt_dict = {} #{||W|| : [w,b]}
        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]] #transformation matrix
        all_data = []
        #Iterate through all data
        for yi in self.data: #yi is the y value
            for featureset in self.data[yi]: #featureset is the x value
                for feature in featureset: #feature is the x value
                    all_data.append(feature) #append all x values to all_data
        
        self.max_feature_value = max(all_data) #max value of all x values
        self.min_feature_value = min(all_data) #min value of all x values
        all_data = None

        #Support vectors yi(xi.w+b) = 1
        #1.01 is the margin of error

        step_sizes = [self.max_feature_value * 0.1, 
        self.max_feature_value * 0.01, 
        #We are going to step through the data by 0.1, 0.01, 0.001, 0.0001
        self.max_feature_value * 0.001]
        
        #extremely expensive
        b_range_multiplier = 2  
        #We dont need to take as small of steps with b as we do with w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10 #we want to start with a high value
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            #we want to set the w value to the latest optimum
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiplier), self.max_feature_value*b_range_multiplier, step*b_multiple):
                    for transformation in transforms:
                        w_transformed = w*transformation
                        #transformation matrix
                        #we want to transform the w value
                        found_option = True
                        #we want to check if the data is classified correctly
                        #Weakest Link as SMO attempts to find the best hyperplane 
                        #Have to run through all data to check for fitting
                        for i in self.data:
                            for xi in self.data[i]: #xi is the x value
                                yi = i #yi is the y value
                                #yi(xi.w+b) >= 1 check for constraint
                                if not yi*(np.dot(w_transformed, xi)+b) >= 1: #if the data is not classified correctly
                                    found_option = False
                                    #if the data is not classified correctly
                        if found_option:
                            opt_dict[np.linalg.norm(w_transformed)] = [w_transformed, b]
                if w[0] < 0:
                    optimized = True
                    print("Optimized a step...")
                else:
                    #w = [5,5]
                    #step = 1
                    #w-[step,step] = [4,4]
                    #works same here as w-step=[4,4] for Computer Science
                    w = w - step
            
            norms = sorted([n for n in opt_dict]) #sort the values in opt_dict by their magnitude
            opt_choice = opt_dict[norms[0]] #choose the first value in opt_dict

            self.w = opt_choice[0] #set the w value to the first value in opt_dict
            self.b = opt_choice[1] #set the b value to the second value in opt_dict
            latest_optimum = opt_choice[0][0]+step*2 #set the latest optimum to the first value in opt_dict plus the step size


    def predict(self, features):
        #Sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        if classification!=0 and self.visualization:
            self.axis.scatter(features[0], features[1], s=200, marker="*", c=self.colors[classification])
        return classification
    
    def visualize(self):
        [[self.axis.scatter(x[0],x[1], s=100, color = self.colors[i]) for x in data_dict[i]] for i in data_dict]

        #hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        #PSV = Positive Support Vector Hyperplane
        #(W.X+b)=1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.axis.plot([hyp_x_min, hyp_x_max], [psv1, psv2], "k")
        
        #NSV = Negative Support Vector Hyperplane
        #(W.X+b)=-1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.axis.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], "k")
        
        #DB = Decision Boundary
        #(W.X+b)=0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.axis.plot([hyp_x_min, hyp_x_max], [db1, db2], "y--")
        
        plt.show()
        

data_dict = {-1:np.array([[1,7],[2,8],[3,8]]), 1:np.array([[5,1], [6,-1], [7,3]])}
predict_data = [
    [2,4],
    [3,5],
    [1,-3],
    [5,6],
    [6,-5]
]

clf = SVM()
clf.fit(data=data_dict)
for predict in predict_data:
    clf.predict(predict)
clf.visualize()
