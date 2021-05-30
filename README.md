Neural Network From Scratch in Python
=====================================
Implement neural network from scratch in python.

Diagram of A Simple Neural Net
-----------------------------

``` mermaid
graph LR
X[sample X] --> x1
X --> x2
x1((x1))--*w11-->b1((+b1))
x1--*w12-->b2((+b2))
x2((x2))--*w21-->b1
x2--*w22-->b2

b1--Relu-->h1((h1))
b2--Relu-->h2((h2))

h1-->softmax((softmax))
h2-->softmax

softmax-->p1((p1))
softmax-->p2((p2))
p1 --> y_pred((y_pred))
p2 --> y_pred

y_pred-->cross_entropy((cross_entropy))
y_true-->cross_entropy

cross_entropy --> loss
```

Layer by Layer
--------------
We need to keep in mind the big picture here :

1. Each layer may have **one or more** inputs and a single output.
2. The derivative with respect to the variable should have the **same shape**
   to that variable.
3. When conducting back propagation, each layer would receive $\frac{
   \partial{loss} }{\partial{self.output}}$ from the previous layer.
4. We should implement **gradient checker** for each layer, because the process of
   back propagation is error prone.

The abstract class `Layer` looks something like this:
``` python
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # self.input = input
        # ...
        # return self.output
        raise NotImplementedError

    def backward(self, doutput=1):
        raise NotImplementedError

    def check_gradient(self, to_be_check):
        raise NotImplementedError
```

- [x] Dense_Layer
- [x] Activation_Relu
- [x] Activation_Sigmoid
- [x] Activation_Softmax
- [x] Loss_CategoricalCrossentropy
- [x] Activation_Softmax_Loss_CategoricalCrossentropy

Training a Neural Network
-------------------------
### Batch gradient decent
$$ w = w -  l*\frac{\partial{loss}}{\partial{w}} $$
``` python
weights += -self.current_learning_rate * dweights
biases += -self.current_learning_rate * dbiases
```

These elements affect the training process a lot:
1. **Learing rate**
2. How you choose to **initialize** the parameters

Referrences
-----------
1. [Neural Networks from Scratch in **X**](https://github.com/Sentdex/NNfSiX)
2. [tinynn](https://github.com/borgwang/tinynn)
