import numpy as np


class Parameters:
    """
    Class to represent inputs and parameters for Layer
    """
    def __init__(self,
                 name,
                 value=None,
                 dvalue=None,
                 momentum=None,
                 cache=None):
        self.name = name
        self.value = value
        self.dvalue = dvalue
        self.momentum = momentum
        self.cache = cache


class Layer:
    """
    1. parameters (input + parameters) is stored in self.parameters
    2. able to call self.forward without argument which make it easy to check gradient
    """
    def __init__(self):
        # self.inputs = {}
        self.parameters = {}
        self.output = None

    def _set_parameter(self,
                       name,
                       value=None,
                       dvalue=None,
                       momentum=None,
                       cache=None):
        parameter = self.parameters.get(name)
        if parameter is None:
            self.parameters[name] = Parameters(name, value, dvalue, momentum,
                                               cache)
            return
        if value is not None:
            self.parameters[name].value = value
        if dvalue is not None:
            self.parameters[name].dvalue = dvalue
        if momentum is not None:
            self.parameters[name].momentum = momentum
        if cache is not None:
            self.parameters[name].cache = cache

    def _get_parameter(self, name):
        parameter = self.parameters.get(name)
        return parameter

    def _get_parameter_prop(self, name, prop):
        """
        get parameter value/dvalue/momentum by name
        """
        parameter = self.parameters.get(name)
        if hasattr(parameter, prop):
            return getattr(parameter, prop)
        return None

    def get_x(self, name):
        return self._get_parameter_prop(name, prop='value')

    def get_dx(self, name):
        return self._get_parameter_prop(name, prop='dvalue')

    def get_momentum(self, name):
        return self._get_parameter_prop(name, prop='momentum')

    def get_cache(self, name):
        return self._get_parameter_prop(name, prop='cache')

    def set_x(self, name, value):
        self._set_parameter(name, value=value)

    def set_dx(self, name, dvalue):
        self._set_parameter(name, dvalue=dvalue)

    def set_momentum(self, name, momentum):
        self._set_parameter(name, momentum=momentum)

    def set_cache(self, name, cache):
        self._set_parameter(name, cache=cache)

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

        for _, parameter in self.parameters.items():
            dx = parameter.dvalue
            if dx is None:
                continue
            x = parameter.value
            # TODO: more smart way to traverse a muti-dimentional array
            x_shape = x.shape
            for i in range(x_shape[0]):
                for j in range(x_shape[1]):
                    # x - h
                    x[i][j] -= delta
                    self.forward()
                    loss_l = lossfun(self.output)

                    # x + h
                    x[i][j] += 2 * delta
                    self.forward()
                    loss_r = lossfun(self.output)

                    loss_delta = 0.5 * (loss_r - loss_l)
                    dx_ij_numerical = loss_delta / delta

                    # diff =  a-b or (a-b)/a when abs(a) > 1
                    diff = np.abs((dx_ij_numerical - dx[i][j]))
                    if diff > tollerance:
                        if abs(dx_ij_numerical) > 1:
                            diff = diff / abs(dx_ij_numerical)

                    if diff > tollerance:
                        # ReLU is non-differentiable at 0
                        if self.__class__ == Activation_ReLU and abs(
                                x[i][j]) < 2 * delta:
                            pass
                        else:
                            print(doutput)
                            print('i, j, dx_ij_numerical, dx[i][j], diff:', i,
                                  j, dx_ij_numerical, dx[i][j], diff)
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
        weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        biases = 0. * np.random.randn(1, n_neurons)

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


class Optimizer:
    def __init__(self, learning_rate=1.0, decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def _update_params_one_layer(self, layer):
        raise NotImplementedError

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


class Optimizer_SGD(Optimizer):
    def __init__(self,
                 learning_rate=1.0,
                 decay=0.,
                 momentum=0.,
                 nesterov=False):
        super().__init__(learning_rate=learning_rate, decay=decay)
        self.momentum = momentum
        self.nesterov = nesterov

    def _update_params_one_layer(self, layer):
        weights = layer.get_x('weights')
        biases = layer.get_x('biases')

        dweights = layer.get_dx('weights')
        dbiases = layer.get_dx('biases')

        current_learning_rate = self.current_learning_rate
        momentum = self.momentum
        nesterov = self.nesterov

        if momentum and nesterov:
            weight_momentums = layer.get_momentum('weights')
            bias_momentums = layer.get_momentum('biases')
            if weight_momentums is None:
                weight_momentums = np.zeros_like(weights)
            if bias_momentums is None:
                bias_momentums = np.zeros_like(biases)

            weight_momentums_new = momentum * weight_momentums - current_learning_rate * dweights
            bias_momentums_new = momentum * bias_momentums - current_learning_rate * dbiases

            weights_update = (
                1 +
                momentum) * weight_momentums_new - momentum * weight_momentums
            biases_update = (
                1 + momentum) * bias_momentums_new - momentum * bias_momentums

            layer.set_momentum('weights', weight_momentums_new)
            layer.set_momentum('biases', bias_momentums_new)

        elif momentum:
            weight_momentums = layer.get_momentum('weights')
            bias_momentums = layer.get_momentum('biases')
            if weight_momentums is None:
                weight_momentums = np.zeros_like(weights)
            if bias_momentums is None:
                bias_momentums = np.zeros_like(biases)

            # integrate velocity
            weights_update = momentum * weight_momentums - current_learning_rate * dweights
            biases_update = momentum * bias_momentums - current_learning_rate * dbiases
            layer.set_momentum('weights', weights_update)
            layer.set_momentum('biases', biases_update)
        else:
            weights_update = -current_learning_rate * dweights
            biases_update = -current_learning_rate * dbiases

        # integrate position
        layer.set_x('weights', weights + weights_update)
        layer.set_x('biases', biases + biases_update)


class Optimizer_Adagrad(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon

    def _update_params_one_layer(self, layer):
        weights = layer.get_x('weights')
        biases = layer.get_x('biases')

        dweights = layer.get_dx('weights')
        dbiases = layer.get_dx('biases')

        current_learning_rate = self.current_learning_rate
        epsilon = self.epsilon

        weight_caches = layer.get_cache('weights')
        bias_caches = layer.get_cache('biases')
        if weight_caches is None:
            weight_caches = np.zeros_like(weights)
        if bias_caches is None:
            bias_caches = np.zeros_like(biases)

        # ---------------------------------------------
        weight_caches += dweights**2
        bias_caches += dbiases**2
        weights += -current_learning_rate * dweights / (
            np.sqrt(weight_caches) + epsilon)
        biases += -current_learning_rate * dbiases / (np.sqrt(bias_caches) +
                                                      epsilon)
        # ---------------------------------------------

        layer.set_cache('weights', weight_caches)
        layer.set_cache('biases', bias_caches)
        layer.set_x('weights', weights)
        layer.set_x('biases', biases)


class Optimizer_RMSprop(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.rho = rho

    def _update_params_one_layer(self, layer):
        weights = layer.get_x('weights')
        biases = layer.get_x('biases')

        dweights = layer.get_dx('weights')
        dbiases = layer.get_dx('biases')

        current_learning_rate = self.current_learning_rate
        epsilon = self.epsilon
        rho = self.rho

        weight_caches = layer.get_cache('weights')
        bias_caches = layer.get_cache('biases')
        if weight_caches is None:
            weight_caches = np.zeros_like(weights)
        if bias_caches is None:
            bias_caches = np.zeros_like(biases)

        # ---------------------------------------------
        weight_caches = rho * weight_caches + (1 - rho) * dweights**2
        bias_caches = rho * bias_caches + (1 - rho) * dbiases**2
        weights += -current_learning_rate * dweights / (
            np.sqrt(weight_caches) + epsilon)
        biases += -current_learning_rate * dbiases / (np.sqrt(bias_caches) +
                                                      epsilon)
        # ---------------------------------------------

        layer.set_cache('weights', weight_caches)
        layer.set_cache('biases', bias_caches)
        layer.set_x('weights', weights)
        layer.set_x('biases', biases)


class Optimizer_Adadelta(Optimizer):
    def __init__(self, epsilon=1e-3, rho=0.95):
        super().__init__()
        self.epsilon = epsilon
        self.rho = rho

    def _update_params_one_layer(self, layer):
        weights = layer.get_x('weights')
        biases = layer.get_x('biases')

        dweights = layer.get_dx('weights')
        dbiases = layer.get_dx('biases')

        epsilon = self.epsilon
        rho = self.rho

        weight_caches = layer.get_cache('weights')
        bias_caches = layer.get_cache('biases')
        weight_delta_caches = layer.get_momentum('weights')
        bias_delta_caches = layer.get_momentum('biases')
        if weight_caches is None:
            weight_caches = np.zeros_like(weights)
        if bias_caches is None:
            bias_caches = np.zeros_like(biases)
        if weight_delta_caches is None:
            weight_delta_caches = np.zeros_like(weights)
        if bias_delta_caches is None:
            bias_delta_caches = np.zeros_like(biases)

        # ---------------------------------------------
        weight_caches = rho * weight_caches + (1 - rho) * dweights**2
        bias_caches = rho * bias_caches + (1 - rho) * dbiases**2

        weight_delta = -dweights * (np.sqrt(weight_delta_caches) + epsilon) / (
            np.sqrt(weight_caches) + epsilon)
        bias_delta = -dbiases * (np.sqrt(bias_delta_caches) +
                                 epsilon) / (np.sqrt(bias_caches) + epsilon)

        weight_delta_caches = rho * weight_delta_caches + (
            1 - rho) * weight_delta**2
        bias_delta_caches = rho * bias_delta_caches + (1 - rho) * bias_delta**2

        weights += weight_delta
        biases += bias_delta
        # ---------------------------------------------
        layer.set_momentum('weights', weight_delta_caches)
        layer.set_momentum('biases', bias_delta_caches)
        layer.set_cache('weights', weight_caches)
        layer.set_cache('biases', bias_caches)
        layer.set_x('weights', weights)
        layer.set_x('biases', biases)


class Optimizer_Adam(Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 decay=0.,
                 epsilon=1e-7,
                 beta_1=0.9,
                 beta_2=0.999):
        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def _update_params_one_layer(self, layer):
        weights = layer.get_x('weights')
        biases = layer.get_x('biases')

        dweights = layer.get_dx('weights')
        dbiases = layer.get_dx('biases')

        current_learning_rate = self.current_learning_rate
        iterations = self.iterations
        epsilon = self.epsilon
        beta_1 = self.beta_1
        beta_2 = self.beta_2

        weight_caches = layer.get_cache('weights')
        bias_caches = layer.get_cache('biases')
        weight_momentums = layer.get_momentum('weights')
        bias_momentums = layer.get_momentum('biases')
        if weight_caches is None:
            weight_caches = np.zeros_like(weights)
        if bias_caches is None:
            bias_caches = np.zeros_like(biases)
        if weight_momentums is None:
            weight_momentums = np.zeros_like(weights)
        if bias_momentums is None:
            bias_momentums = np.zeros_like(biases)

        # ---------------------------------------------
        # update momentums
        weight_momentums = beta_1 * weight_momentums + (1 - beta_1) * dweights
        bias_momentums = beta_1 * bias_momentums + (1 - beta_1) * dbiases

        # update caches
        weight_caches = beta_2 * weight_caches + (1 - beta_2) * dweights**2
        bias_caches = beta_2 * bias_caches + (1 - beta_2) * dbiases**2

        coef = np.sqrt(1 - beta_2**(iterations + 1)) / (1 - beta_1**
                                                        (iterations + 1))
        weights += -coef * current_learning_rate * weight_momentums / (
            np.sqrt(weight_caches) + epsilon)
        biases += -coef * current_learning_rate * bias_momentums / (
            np.sqrt(bias_caches) + epsilon)
        # ---------------------------------------------

        layer.set_momentum('weights', weight_momentums)
        layer.set_momentum('biases', bias_momentums)
        layer.set_cache('weights', weight_caches)
        layer.set_cache('biases', bias_caches)
        layer.set_x('weights', weights)
        layer.set_x('biases', biases)
