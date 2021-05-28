import numpy as np


class Layer:
    def __init__(self):
        # there might be multiple input, consider using a list to represent
        # inputs
        self.inputs = None
        self.output = None

    def forward(self, *inputs):
        # self.inputs = inputs
        # return self.output
        raise NotImplementedError

    def backward(self, doutput=1):
        raise NotImplementedError




class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = 0


class Activation_Sigmoid:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-1 * inputs))
        return self.output

    def backward(self, dvalues):
        self.dinputs = self.inputs * (1 - self.inputs)


class Activation_Softmax:
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output,
                    single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss_CategoricalCrossentropy(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        self.inputs = [y_pred, y_true]
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError(
                f"The shape of y_pred is not supported: {y_pred.shape}.")

        negative_log_likelihoods = -np.log(correct_confidences)
        self.output = np.mean(negative_log_likelihoods)
        return self.output

    def backward(self, dvalues=1):
        y_pred, y_true = self.inputs

        if len(y_true.shape) == 1:
            labels = len(y_pred[0])
            y_true = np.eye(labels)[y_true]

        self.dy_pred = -y_true / y_pred
        # Normalize gradient
        # so that, we don't need to normalize gradient when perform optimization
        self.dy_pred = self.dy_pred / len(y_pred)


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


# class Loss_Crossentropy(Loss):
#     """
#     my own implementation
#     """
#     def forward(self, y_pred, y_true):
#         y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
#         if y_pred.shape != y_true.shape:
#             raise ValueError(
#                 f"y_pred.shape != y_true.shape: {y_pred.shape}!={y_true.shape}"
#             )

#         return -np.sum(np.log(y_pred) * y_true, axis=1)
