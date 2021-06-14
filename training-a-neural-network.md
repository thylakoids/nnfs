Gradient-Based Optimization Algorithms
======================================
Learning Rate Decay
-------------------
### Linear Decay
$$\eta_k = (1-\alpha)\eta_0 + \alpha \eta_{\tau}$$

Note: 

1. $\alpha =\frac{k}{\tau}$
2. usually set $\eta_{\tau} =1\small{\%} \eta_0$

### Explonential Decay
$$\eta_k = \frac{1}{1 + k\alpha}\eta_0$$

Momentum
--------
$$v_t = \gamma v_{t-1} + \eta \frac{\partial loss}{\partial w}$$
$$w = w_{t-1} - v_t$$

Note: The momentum term $\gamma$ is usually set to 0.9 or a similar value.

References
----------
1. [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
