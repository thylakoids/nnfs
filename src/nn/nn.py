import numpy as np


class Layer:
    def __init__(self):
        # there might be multiple input, consider using a list to represent
        # inputs
        self.inputs = None
        self.output = None

        self.dinputs = None

    def forward(self, *inputs):
        # self.inputs = inputs
        # ...
        # return self.output
        raise NotImplementedError

    def backward(self, doutput=1):
        raise NotImplementedError

    def _check_gradient(self, to_be_check_fun, doutput, lossfun):
        delta = 1e-7
        tollerance = 1e-4
        self.backward(doutput)

        loss = lossfun(self.output)
        to_be_check = [(x_fun(), dx_fun())
                       for x_fun, dx_fun in to_be_check_fun]
        for x, dx in to_be_check:
            x_shape = x.shape
            for i in range(x_shape[0]):
                for j in range(x_shape[1]):
                    x[i][j] += delta
                    self.forward(self.inputs)
                    loss_delta = lossfun(self.output) - loss
                    dx_ij_numerical = loss_delta / delta

                    # diff =  a-b or (a-b)/a when abs(a) > 1
                    diff = np.abs((dx_ij_numerical - dx[i][j]))
                    if diff > tollerance:
                        if abs(dx_ij_numerical) > 1:
                            diff = diff / abs(dx_ij_numerical)

                    if diff > tollerance:
                        print(doutput)
                        print('i, j, dx_ij_numerical, dx[i][j], diff:', i, j,
                              dx_ij_numerical, dx[i][j], diff)
                        raise AssertionError
                    # restore parameters and inputs
                    x[i][j] -= delta
        # restore output
        self.forward(self.inputs)

    def check_gradient(self, to_be_check_fun=None):
        """check_gradient
        Assume the next layer is: `f(x) = x` then `d(x) = 1` or `f(x) = x^2`
        then `d(x) = 2*x`.
        example:
        to_be_check = [[lambda: self.weights, lambda: self.dweights],
                       [lambda: self.biases, lambda: self.dbiases],
                       [lambda: self.inputs, lambda: self.dinputs]]
        super().check_gradient(to_be_check_fun)
        """
        to_be_check_fun = to_be_check_fun or [[
            lambda: self.inputs, lambda: self.dinputs
        ]]
        doutput_lossfun = [(np.ones_like(self.output), lambda x: np.sum(x)),
                           (2 * self.output, lambda x: np.sum(x * x)),
                           (3 * self.output**2, lambda x: np.sum(x**3))]
        for doutput, lossfun in doutput_lossfun:
            self._check_gradient(to_be_check_fun, doutput, lossfun)


class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()

        # TODO: how to initial parameters
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = 0.10 * np.random.randn(1, n_neurons)

        self.dweights = None
        self.dbiases = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, doutput):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, doutput)
        self.dbiases = np.sum(doutput, axis=0, keepdims=True)

        # Gradient on input
        self.dinputs = np.dot(doutput, self.weights.T)

    def check_gradient(self):
        to_be_check = [[lambda: self.weights, lambda: self.dweights],
                       [lambda: self.biases, lambda: self.dbiases],
                       [lambda: self.inputs, lambda: self.dinputs]]
        super().check_gradient(to_be_check)


class Activation_ReLU(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, doutput):
        self.dinputs = doutput.copy()
        self.dinputs[self.inputs < 0] = 0


class Activation_Sigmoid(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-1 * inputs))
        return self.output

    def backward(self, doutput):
        self.dinputs = doutput * self.output * (1 - self.output)


class Activation_Softmax(Layer):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, doutput):
        # Create uninitialized array
        self.dinputs = np.empty_like(doutput)

        # Enumerate outputs and gradients
        for index, (single_output,
                    single_dvalues) in enumerate(zip(self.output, doutput)):
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

    def forward(self, inputs):
        y_pred, y_true = inputs
        self.inputs = inputs
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

    def backward(self, doutput=1):
        y_pred, y_true = self.inputs

        if len(y_true.shape) == 1:
            labels = len(y_pred[0])
            y_true = np.eye(labels)[y_true]

        dy_pred = -y_true / y_pred
        # Normalize gradient
        # so that, we don't need to normalize gradient when perform optimization
        dy_pred = doutput * dy_pred / len(y_pred)
        self.dinputs = [dy_pred]

    def check_gradient(self):
        to_be_check_fun = [[lambda: self.inputs[0], lambda: self.dinputs[0]]]
        return super().check_gradient(to_be_check_fun=to_be_check_fun)


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
    def __init__(self, learning_rate=1.0, decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1
