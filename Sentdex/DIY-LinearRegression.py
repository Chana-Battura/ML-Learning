################################################################
# Best Fit Line Slope Calculation:                             #
# y = mx + b                                                   #
# m = (mean(x)*mean(y)- mean(x*y) / mean(x)**2 - mean((x)**2)  #
# b = (mean(y) - m*mean(x))                                    #
################################################################

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

Xs = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float64)
Ys = np.array([5,4,6,5,3,5,7,5,6,5], dtype=np.float64)

def best_fit_slope_and_intercept(Xs, Ys):
    m = (mean(Xs)*mean(Ys) - mean(Xs*Ys))/(mean(Xs)**2 - mean(Xs**2))
    b = mean(Ys) - m*mean(Xs)
    return m, b

m, b = best_fit_slope_and_intercept(Xs, Ys)

regression_line = [(m*x)+b for x in Xs]

predict_x = 11
predict_y = (m*predict_x)+b
plt.scatter(predict_x, predict_y, c="g")

plt.plot(Xs, regression_line, c='r')
plt.scatter(Xs,Ys, c="b")
plt.show()