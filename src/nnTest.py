import unittest
import numpy as np
from nnfs.datasets import spiral_data


from nn.nn import Layer_Dense, Activation_Softmax, Activation_Sigmoid

np.random.seed(0)


class Testnn(unittest.TestCase):
    X, y = spiral_data(100, 3)

    def test_nn(self):
        layer1 = Layer_Dense(2, 5)
        activation1 = Activation_Sigmoid()

        layer2 = Layer_Dense(5, 3)
        activation2 = Activation_Softmax()

        layer1.forward(self.X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        self.assertEqual(activation2.output.shape, (300, 3))


if __name__ == "__main__":
    unittest.main()
