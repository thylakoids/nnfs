import numpy as np


class Parameters:
    """
    Class to represent inputs and parameters for Layer
    """
    def __init__(self, name, value=None, dvalue=None):
        self.name = name
        self.value = value
        self.dvalue = dvalue


class Layer:
    """
    1. parameters (input + parameters) is stored in self.parameters
    2. able to call self.forward without argument which make it easy to check gradient
    """
    def __init__(self):
        # self.inputs = {}
        self.parameters = {}
        self.output = None

    def _set_parameter(self, name, value=None, dvalue=None):
        parameter = self.parameters.get(name)
        if parameter is None:
            self.parameters[name] = Parameters(name, value, dvalue)
            return
        if value is not None:
            self.parameters[name].value = value
        if dvalue is not None:
            self.parameters[name].dvalue = dvalue

    def _get_parameter(self, name):
        parameter = self.parameters.get(name)
        return parameter

    def _get_parameter_prop(self, name, derivative=False):
        """
        get parameter value/dvalue by name
        """
        parameter = self.parameters.get(name)
        if parameter is None:
            return None
        if derivative:
            return parameter.dvalue
        return parameter.value

    def get_x(self, name):
        return self._get_parameter_prop(name, derivative=False)

    def get_dx(self, name):
        return self._get_parameter_prop(name, derivative=True)

    def set_x(self, name, value):
        self._set_parameter(name, value=value)

    def set_dx(self, name, dvalue):
        self._set_parameter(name, dvalue=dvalue)

    def forward(self, input=None):
        # self.set_x('input', input)
        # ...
        # return self.output
        raise NotImplementedError

    def backward(self, doutput):
        raise NotImplementedError

    def _check_gradient(self, doutput, lossfun):
        delta = 1e-7
        tollerance = 1e-4
        self.backward(doutput)
        loss = lossfun(self.output)

        for _, parameter in self.parameters.items():
            dx = parameter.dvalue
            if dx is None:
                continue
            x = parameter.value
            # TODO: more smart way to traverse a muti-dimentional array
            x_shape = x.shape
            for i in range(x_shape[0]):
                for j in range(x_shape[1]):
                    x[i][j] += delta
                    self.forward()
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
        self.forward()

    def check_gradient(self):
        """check_gradient
        Assume the next layer is: `f(x) = x` then `d(x) = 1` or `f(x) = x^2`
        then `d(x) = 2*x`.
        """
        doutput_lossfun = [(np.ones_like(self.output), lambda x: np.sum(x)),
                           (2 * self.output, lambda x: np.sum(x * x)),
                           (3 * self.output**2, lambda x: np.sum(x**3))]
        for doutput, lossfun in doutput_lossfun:
            self._check_gradient(doutput, lossfun)


class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()

        # TODO: how to initial parameters
        weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        biases = 0.10 * np.random.randn(1, n_neurons)

        self.set_x('weights', weights)
        self.set_x('biases', biases)

    @property
    def weights(self):
        return self.get_x('weights')

    @weights.setter
    def weights(self, value):
        self.set_x('weights', value)

    def forward(self, input=None) -> np.ndarray:
        if input is not None:
            self.set_x('input', input)

        weights = self.get_x('weights')
        biases = self.get_x('biases')
        input = self.get_x('input')
        if input is None:
            raise ValueError("no input provided")

        # -------------------------
        self.output = np.dot(input, weights) + biases
        # -------------------------

        return self.output

    def backward(self, doutput):
        input = self.get_x('input')
        weights = self.get_x('weights')

        # -------------------------
        # Gradients on parameters
        dweights = np.dot(input.T, doutput)
        dbiases = np.sum(doutput, axis=0, keepdims=True)
        # Gradient on input
        dinput = np.dot(doutput, weights.T)
        # -------------------------

        self.set_dx('weights', dweights)
        self.set_dx('biases', dbiases)
        self.set_dx('input', dinput)


class Activation_ReLU(Layer):
    def forward(self, input=None) -> np.ndarray:
        if input is not None:
            self.set_x('input', input)
        input = self.get_x('input')
        if input is None:
            raise ValueError("no input provided")

        # -------------------------
        self.output = np.maximum(0, input)
        # -------------------------

        return self.output

    def backward(self, doutput):
        input = self.get_x('input')

        # ---------------------
        dinput = doutput.copy()
        dinput[input < 0] = 0
        # ---------------------

        self.set_dx('input', dinput)


class Activation_Sigmoid(Layer):
    def forward(self, input=None) -> np.ndarray:
        if input is not None:
            self.set_x('input', input)
        input = self.get_x('input')
        if input is None:
            raise ValueError("no input provided")

        # -------------------------
        self.output = 1 / (1 + np.exp(-1 * input))
        # -------------------------

        return self.output

    def backward(self, doutput):
        # -------------------------
        dinput = doutput * self.output * (1 - self.output)
        # -------------------------

        self.set_dx('input', dinput)


class Activation_Softmax(Layer):
    def forward(self, input=None) -> np.ndarray:
        if input is not None:
            self.set_x('input', input)
        input = self.get_x('input')
        if input is None:
            raise ValueError("no input provided")
        # -------------------------
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # -------------------------
        return self.output

    def backward(self, doutput):
        # ---------------------
        # Create uninitialized array
        dinput = np.empty_like(doutput)

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
            dinput[index] = np.dot(jacobian_matrix, single_dvalues)
        # ---------------------

        self.set_dx('input', dinput)


class Loss_CategoricalCrossentropy(Layer):
    def forward(self, y_pred=None, y_true=None):
        if y_pred is not None:
            self.set_x('y_pred', y_pred)
        if y_true is not None:
            self.set_x('y_true', y_true)
        y_pred = self.get_x('y_pred')
        y_true = self.get_x('y_true')
        if y_pred is None or y_true is None:
            raise ValueError("no y_pred and y_true provided")

        # -----------------------------
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
        # -----------------------------

        return self.output

    def backward(self, doutput=1):
        y_pred = self.get_x('y_pred')
        y_true = self.get_x('y_true')

        # -----------------------
        if len(y_true.shape) == 1:
            labels = len(y_pred[0])
            y_true = np.eye(labels)[y_true]

        dy_pred = -y_true / y_pred
        # Normalize gradient
        # so that, we don't need to normalize gradient when perform optimization
        dy_pred = doutput * dy_pred / len(y_pred)
        # ------------------------

        self.set_dx('y_pred', dy_pred)


class Activation_Softmax_Loss_CategoricalCrossentropy(Layer):
    def __init__(self):
        super().__init__()
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, input=None, y_true=None):
        if input is not None:
            self.set_x('input', input)
        if y_true is not None:
            self.set_x('y_true', y_true)

        input = self.get_x('input')
        y_true = self.get_x('y_true')

        # ---------------------
        self.activation.forward(input)
        self.output = self.loss.forward(self.activation.output, y_true)
        # ---------------------

        return self.output

    def backward(self, doutput=1):
        y_true = self.get_x('y_true')
        y_pred = self.activation.output

        # ----------------------------
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        samples = len(y_true)

        dinput = y_pred.copy()
        dinput[range(samples), y_true] -= 1
        dinput = doutput * dinput / samples
        # ----------------------------

        self.set_dx('input', dinput)


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def _update_params_one_layer(self, layer):
        weights = layer.get_x('weights')
        biases = layer.get_x('biases')

        weights += -self.current_learning_rate * layer.get_dx('weights')
        biases += -self.current_learning_rate * layer.get_dx('biases')

        layer.set_x('weights', weights)
        layer.set_x('biases', biases)

    def update_params(self, layers):
        self.pre_update_params()
        for layer in layers:
            self._update_params_one_layer(layer)
        self.post_update_params()

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1
