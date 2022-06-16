################################################################
# Best Fit Line Slope Calculation:                             #
# y = mx + b                                                   #
# m = (mean(x)*mean(y)- mean(x*y) / mean(x)**2 - mean((x)**2)  #
# b = (mean(y) - m*mean(x))                                    #
################################################################

from cProfile import label
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

#Xs = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float64)
#Ys = np.array([120,135,145,140,165,155,160,175,173,170], dtype=np.float64)


def create_dataset(count, variance, step=5, correlation=False):
    init_value = 1
    Ys = []
    for i in range(count):
        Y = init_value + random.randrange(-variance, variance)
        Ys.append(Y)
        if correlation and correlation == "+":
            init_value += step
        elif correlation and correlation == "-":
            init_value -= step
    Xs = [i for i in range(len(Ys))]

    return np.array(Xs, dtype=np.float64), np.array(Ys, dtype=np.float64)

def best_fit_slope_and_intercept(Xs, Ys):
    m = (mean(Xs)*mean(Ys) - mean(Xs*Ys))/(mean(Xs)**2 - mean(Xs**2))
    b = mean(Ys) - m*mean(Xs)
    return m, b

################################################################
# r^2 = 1 - (Squared Error(Best Fit)/Squared Error(Mean(y)))   #
# ex. r^2 = 1 - (0.5/0.25) = 0.8                               #
# We want the best fit error to be as low as possible compared #
# to the mean error.                                           #
################################################################

def squared_error(Ys_orig, Ys_line):
    return sum((Ys_orig-Ys_line)**2)

def coefficent_of_determination(Ys_orig, Ys_line):
    Y_mean_line = [mean(Ys_orig) for y in Ys_orig]
    squared_error_regr = squared_error(Ys_orig, Ys_line)
    squared_error_ymean = squared_error(Ys_orig, Y_mean_line)
    return (1-(squared_error_regr/squared_error_ymean))

Xs, Ys = create_dataset(50, 40, 3, correlation="-")

m, b = best_fit_slope_and_intercept(Xs, Ys)

regression_line = [(m*x)+b for x in Xs]
Y_mean_line = [mean(Ys) for y in Ys]

predict_x = 55
predict_y = (m*predict_x)+b

r_squared = coefficent_of_determination(Ys, regression_line)
print("R Squared Value is: {}".format(r_squared))
plt.scatter(predict_x, predict_y, c="g", label="Prediction")
plt.plot(Xs, regression_line, c='r', label="Best Fit Line")
plt.plot(Xs, Y_mean_line, c="m", label="Mean Line")
plt.scatter(Xs,Ys, c="b", label="Data")
plt.legend()
plt.show()

