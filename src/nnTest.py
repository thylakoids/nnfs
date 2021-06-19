import unittest
import numpy as np
from nnfs.datasets import spiral_data

from nn.nn import (Layer_Dense, Activation_Softmax, Activation_Sigmoid,
                   Activation_ReLU, Loss_CategoricalCrossentropy,
                   Activation_Softmax_Loss_CategoricalCrossentropy,
                   Optimizer_Adadelta, Optimizer_Adagrad, Optimizer_Adam,
                   Optimizer_RMSprop, Optimizer_SGD)

np.random.seed(0)


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
        # optimizer = Optimizer_SGD(learning_rate=1, decay=1e-4)
        # optimizer = Optimizer_SGD(learning_rate=1,
        #                           decay=1e-3,
        #                           momentum=0.9,
        #                           nesterov=True)
        # optimizer = Optimizer_Adagrad(learning_rate=1, decay=0)
        # optimizer = Optimizer_RMSprop(learning_rate=0.02,
        #                               decay=1e-5,
        #                               rho=0.999)
        # optimizer = Optimizer_Adadelta()
        optimizer = Optimizer_Adam(learning_rate=0.05, decay=1e-7)

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
                print(
                    f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, learning_rate: {optimizer.current_learning_rate}'
                )

            # perform backward pass
            loss_activation.backward()
            dense2.backward(loss_activation.get_dx('input'))
            activation1.backward(dense2.get_dx('input'))
            dense1.backward(activation1.get_dx('input'))

            # update parameters
            optimizer.update_params([dense1, dense2])


if __name__ == "__main__":
    unittest.main()
