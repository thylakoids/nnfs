import unittest
import numpy as np
# import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from thylakoids.time_code import code_timer

from nn.nn import (Layer_Dense, Activation_Softmax, Activation_Sigmoid,
                   Activation_ReLU, Loss_CategoricalCrossentropy,
                   Activation_Softmax_Loss_CategoricalCrossentropy,
                   Optimizer_SGD)

# np.random.seed(0)
# nnfs.init()


class Testnn(unittest.TestCase):
    # X:300*2, y:300*1
    X, y = spiral_data(samples=100, classes=3)
    plot = False

    def test_layer_checker(self):
        # define
        dense1 = Layer_Dense(2, 9)
        activation1 = Activation_ReLU()
        activation2 = Activation_Sigmoid()
        activation3 = Activation_Softmax()
        loss = Loss_CategoricalCrossentropy()

        # forward
        dense1.forward(self.X)
        activation1.forward(dense1.output)
        activation2.forward(dense1.output)
        activation3.forward(dense1.output)
        loss.forward([activation3.output, self.y])

        # check
        dense1.check_gradient()
        activation1.check_gradient()
        activation2.check_gradient()
        activation3.check_gradient()
        loss.check_gradient()


if __name__ == "__main__":
    unittest.main()
