import unittest
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from thylakoids.time_code import code_timer

from nn.nn import (Layer_Dense, Activation_Softmax, Activation_Sigmoid,
                   Activation_ReLU, Loss_CategoricalCrossentropy,
                   Loss_Crossentropy,
                   Activation_Softmax_Loss_CategoricalCrossentropy,
                   Optimizer_SGD)

np.random.seed(0)
# nnfs.init()


class Testnn(unittest.TestCase):
    # X:300*2, y:300*1
    X, y = spiral_data(samples=100, classes=3)
    plot = False

    def test_crossEntropy(self):
        samples = self.X.shape[0]

        y_predict = np.random.randn(samples, 3)
        y_true = np.zeros([samples, 3])
        y_true[range(samples), self.y] = 1

        loss_Crossentropy = Loss_Crossentropy()
        loss_CategoricalCrossentropy = Loss_CategoricalCrossentropy()

        with code_timer('loss1'):
            loss1 = loss_Crossentropy.calculate(y_predict, y_true)
        with code_timer():
            loss2 = loss_CategoricalCrossentropy.calculate(y_predict, y_true)
        self.assertEqual(loss1, loss2)

    def test_backward_softmax_crossEntropy(self):
        softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4],
                                    [0.02, 0.9, 0.08]])
        class_targets = np.array([0, 1, 1])

        softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
        softmax_loss.backward(softmax_outputs, class_targets)
        dvalues1 = softmax_loss.dinputs

        activation = Activation_Softmax()
        activation.output = softmax_outputs
        loss = Loss_CategoricalCrossentropy()
        loss.backward(softmax_outputs, class_targets)
        activation.backward(loss.dinputs)
        dvalues2 = activation.dinputs

        self.assertAlmostEqual(dvalues1[0, 0], dvalues2[0, 0])

    def test_nn1(self):
        layer1 = Layer_Dense(2, 5)
        activation1 = Activation_Sigmoid()

        layer2 = Layer_Dense(5, 3)
        activation2 = Activation_Softmax()

        loss_function = Loss_CategoricalCrossentropy()

        layer1.forward(self.X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        loss = loss_function.calculate(activation2.output, self.y)
        print(f"loss: {loss}")

        predictions = np.argmax(activation2.output, axis=1)
        y = self.y
        if len(y.shape) == 2:
            y = np.argmax(self.y, axis=1)
        accuracy = np.mean(predictions == y)
        print('acc:', accuracy)

        self.assertEqual(activation2.output.shape, (300, 3))

    def test_nn2(self):
        """test_nn2
        1. combine softmax and cross entropy loss
        2. forward and backward
        :return:
        """

        layer1 = Layer_Dense(2, 5)
        activation1 = Activation_Sigmoid()

        layer2 = Layer_Dense(5, 3)
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

        # Forward pass
        layer1.forward(self.X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        loss = loss_activation.forward(layer2.output, self.y)
        print(f"loss: {loss}")

        # Accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        y = self.y
        if len(y.shape) == 2:
            y = np.argmax(self.y, axis=1)
        accuracy = np.mean(predictions == y)
        print('acc:', accuracy)

        # Backward pass
        loss_activation.backward(loss_activation.output, self.y)
        layer2.backward(loss_activation.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        self.assertEqual(layer1.dweights.shape, (2, 5))
        self.assertEqual(loss_activation.output.shape, (300, 3))

    def test_Activation_Sigmoid(self):
        X = np.linspace(-4, 4, 100)
        activation = Activation_Sigmoid()
        y = activation.forward(X)
        if self.plot:
            plt.plot(X, y)
            plt.xlabel("X")
            plt.ylabel("y")
            plt.title("sigmoid function")
            plt.show()
        self.assertEqual(y.shape, (100, ))

    def test_Activation_ReLU(self):
        X = np.linspace(-4, 4, 100)
        activation = Activation_ReLU()
        y = activation.forward(X)
        if self.plot:
            plt.plot(X, y)
            plt.xlabel("X")
            plt.ylabel("y")
            plt.title("ReLU function")
            plt.show()
        self.assertEqual(y.shape, (100, ))

    def test_train_model(self):
        # create model
        dense1 = Layer_Dense(2, 64)
        activation1 = Activation_ReLU()

        dense2 = Layer_Dense(64, 3)
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

        # create optimizer
        optimizer = Optimizer_SGD(2)
        accs = []
        losses = []

        # Train in loop
        for epoch in range(10001)[:10]:
            # perform forward pass
            dense1.forward(self.X)
            activation1.forward(dense1.output)

            dense2.forward(activation1.output)
            loss = loss_activation.forward(dense2.output, self.y)

            # calculate accuracy
            predictions = np.argmax(loss_activation.output, axis=1)
            if len(self.y.shape) == 2:
                self.y = np.argmax(self.y, axis=1)
            accuracy = np.mean(predictions == self.y)

            accs.append(accuracy)
            losses.append(loss)
            if not epoch % 100:
                print(
                    f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.3f}'
                )

            # perform backward pass
            loss_activation.backward(loss_activation.output, self.y)
            dense2.backward(loss_activation.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # update parameters
            optimizer.pre_update_params()
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.post_update_params()

        if self.plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(accs)
            plt.ylabel('acc')

            plt.subplot(212)
            plt.plot(losses)
            plt.xlabel('epoch')
            plt.ylabel('loss')

            plt.show()

    def test_optimizer_SGD(self):
        # create model
        dense1 = Layer_Dense(2, 64)
        activation1 = Activation_ReLU()

        dense2 = Layer_Dense(64, 3)
        loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

        # create optimizer
        optimizer = Optimizer_SGD()

        # perform forward pass
        dense1.forward(self.X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, self.y)

        print(f'loss: {loss}')

        # check gradient
        dense1.check_gradient()
        activation1.check_gradient()
        dense2.check_gradient()

        # calculate accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(self.y.shape) == 2:
            self.y = np.argmax(self.y, axis=1)
        accuracy = np.mean(predictions == self.y)

        print(f'acc: {accuracy}')

        # perform backward pass
        loss_activation.backward(loss_activation.output, self.y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # update parameters
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

        # perform forward pass #2
        dense1.forward(self.X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, self.y)

        print(f'loss: {loss}')

        # calculate accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(self.y.shape) == 2:
            self.y = np.argmax(self.y, axis=1)
        accuracy = np.mean(predictions == self.y)

        print(f'acc: {accuracy}')


if __name__ == "__main__":
    unittest.main()
