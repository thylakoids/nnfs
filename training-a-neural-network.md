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

### Step decay



Momentum
--------
$$v_{t+1} = \gamma v_t - \eta \nabla_{\theta}J(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

Note: The momentum term $\gamma$ is usually set to 0.9 or a similar value.
>With Momentum update, the parameter vector will build up velocity($\frac{v}{1-\gamma}$) in any direction that has consistent gradient.


Nesterov Accelerated Gradient
-----------------------------
$$\begin{aligned}
v_t  &= \gamma v_{t-1} - \eta \nabla_\theta J(\theta - \gamma v_{t-1}) \\
\theta  &= \theta + v_t
\end{aligned}$$

### Implementation
$$\begin{aligned}
\theta^*_t &= \theta_t + \gamma v_t \\
v_{t+1} &= \gamma v_t - \eta \nabla_{\theta}J(\theta^*_t) \\
\theta_{t+1} &= \theta_t + v_{t+1}
\end{aligned}
$$

However, in practice people prefer to express the update to look as similar to
vanilla SGD or to the previous momentum update as possible. This is possible to
achieve by manipulating the update above with a variable transform $\theta =
\theta^* - \gamma v$, and then expressing the update in terms of $\theta^*$
instead of $\theta$:

$$\begin{aligned}
v_{t+1} &= \gamma v_t - \eta \nabla_{\theta}J(\theta^*_t) \\
\theta^*_{t+1} &= \theta^*_t - \gamma v_t + (1+\gamma)v_{t+1}
\end{aligned}
$$

Then rename $\theta^*$ back to $\theta$:

$$\begin{aligned}
v_{t+1} &= \gamma v_t - \eta \nabla_{\theta}J(\theta_t) \\
\theta_{t+1} &= \theta_t - \gamma v_t + (1+\gamma)v_{t+1} \\
&= \theta_t  + v_{t+1} + \gamma(v_{t+1} - v_t)
\end{aligned}
$$

That is, the parameter vector we are actually storing is always the ahead version.


Per-Parameter Adaptive Learning Rate Methods
--------------------------------------------

### Adagrad
> adaptive gradient

Let
$$g_{t,i} = \nabla_\theta J(\theta_{t,i})$$

Then SGD becomes:
$$\theta_{t+1, i} = \theta_{t, i} - \eta \cdot g_{t,i}$$

In its update rule, Adagrad modifies the general learning rate $\eta$ at each time
step $t$ for every parameter $\theta_i$ based on the past gradients that have been
computed for $\theta_i$ :
$$\theta_{t+1, i} = \theta_{t, i} - \dfrac{\eta}{\sqrt{G_{t, ii}} + \epsilon} \cdot g_{t, i}$$

$G_{t} \in \mathbb{R}^{d \times d}$ here is a diagonal matrix where each
diagonal element $i,j$ is the sum of the squares of the gradients w.r.t.
$\theta_i$ up to time step $t$. We can then vectorize the implementation by
performing a matrix-vector product $\odot$ between $G_t$ and $g_t$:

$$\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t}} + \epsilon} \odot g_{t}$$

### RMSProp
> Root Mean Square Propagation

$$\begin{aligned}
G_t &= \rho G_{t-1} + (1-\rho)g_t^2 \\
\theta_{t+1} &= \theta_{t} - \dfrac{\eta}{\sqrt{G_{t}} + \epsilon} \odot g_{t}
\end{aligned}
$$

Note: $\rho$ is a hyperparameter and typical values are [0.9, 0.99, 0.999].
### Adam
> Adaptive Moment


References
----------
1. [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
2. [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/neural-networks-3/#sgd)
