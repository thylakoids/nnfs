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

np.random.seed(0)
# nnfs.init()


class Testnn(unittest.TestCase):
    # X:300*2, y:300*1
    X, y = spiral_data(samples=100, classes=3)
    plot = False

    def test_layer_gradient_checker(self):
        # define
        dense1 = Layer_Dense(2, 9)
        activation1 = Activation_ReLU()
        activation2 = Activation_Sigmoid()
        activation3 = Activation_Softmax()
        loss = Loss_CategoricalCrossentropy()
        softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()

        # forward
        dense1.forward(self.X)
        activation1.forward(dense1.output)
        activation2.forward(dense1.output)
        activation3.forward(dense1.output)
        loss.forward(activation3.output, self.y)
        softmax_loss.forward(dense1.output, self.y)

        # check
        dense1.check_gradient()
        activation1.check_gradient()
        activation2.check_gradient()
        activation3.check_gradient()
        loss.check_gradient()
        softmax_loss.check_gradient()

    def test_train_model(self):
        # create model
        dense1 = Layer_Dense(2, 64)
        activation1 = Activation_ReLU()

        dense2 = Layer_Dense(64, 3)
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

        # create optimizer
        optimizer = Optimizer_SGD()

        # Train in loop
        for epoch in range(10001):
            # perform forward pass
            dense1.forward(self.X)
            activation1.forward(dense1.output)

            dense2.forward(activation1.output)
            loss = loss_activation.forward(dense2.output, self.y)

            # calculate accuracy
            predictions = np.argmax(loss_activation.activation.output, axis=1)
            if len(self.y.shape) == 2:
                self.y = np.argmax(self.y, axis=1)
            accuracy = np.mean(predictions == self.y)

            if not epoch % 100:
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')

            # perform backward pass
            loss_activation.backward()
            dense2.backward(loss_activation.parameters['input'].dvalue)
            activation1.backward(dense2.parameters['input'].dvalue)
            dense1.backward(activation1.parameters['input'].dvalue)

            # update parameters
            optimizer.update_params([dense1, dense2])


if __name__ == "__main__":
    unittest.main()
