import numpy as np

# y = xw
# y: 1*3
# x: 1*4
# w: 4*3

# dy/dx = w
# dy/dw = 
# dy : y: 1*3

dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, -0.7, 0.87]]).T

dinputs = np.dot(dvalues, weights.T)

print(dinputs)
