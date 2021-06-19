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

### Adadelta
$$\begin{aligned}
G_t &= \rho G_{t-1} + (1-\rho)g_t^2 \\

\Delta\theta_{t} &= - \dfrac{\sqrt{\Delta\Theta_{t-1}}+\epsilon}{\sqrt{G_{t}} + \epsilon} \odot g_{t} \\

\Delta\Theta_t &= \rho \Delta\Theta_{t-1} + (1 - \rho)\Delta\theta_t^2 \\
\theta_{t} &= \theta_{t-1} + \Delta\theta_t
\end{aligned}
$$

> Correct Units with Hessian Approximation

Since second order methods are correct, we rearrange Newton’s method (assuming
a diagonal Hessian) for the inverse of the second derivative to determine the
quantities involved:
$$
\Delta x = \frac{\frac{\partial f}{\partial x}}{\frac{\partial^2 f}{\partial x^2}} \Rightarrow \frac{1}{\frac{\partial^2 f}{\partial x^2}} = \frac{\Delta x}{\frac{\partial f}{\partial x}}
$$

### Adam
> Adaptive Moment Estimation

> Adam : RMSProp + momentum

Firstly compute the decaying averages of past and past squared gradients $m_t$
and $v_t$ respectively as follows:
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
\end{aligned}
$$

$m_t$ and $v_t$ are estimates of the first moment (the mean) and the second
moment (the uncentered variance) of the gradients respectively, hence the name
of the method. As $m_t$ and $v_t$ are initialized as vectors of 0's, the
authors of Adam observe that they are biased towards zero, especially during
the initial time steps, and especially when the decay rates are small (i.e.
$\beta_1$ and $\beta_2$ are close to 1).

They counteract these biases by computing bias-corrected first and second
moment estimates:

$$
\begin{aligned} 
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \\
\hat{v}_t &= \dfrac{v_t}{1 - \beta^t_2}
\end{aligned}
$$


They then use these to update the parameters just as we have seen in Adadelta
and RMSprop, which yields the Adam update rule: 
$$\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

> Note that the efficacy of the algorithm can, at the expense of clarity, be improved upon
by changing the order of computation, e.g. by replacing the last three line in the 
loop with the following lines:


$$\begin{aligned}
\hat{\eta} &= \eta \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
\theta_{t+1} &= \theta_{t} - \dfrac{\hat{\eta}}{\sqrt{v_t} + \epsilon} m_t
\end{aligned}
$$

### AdaMax
The $v_t$ factor in the Adam update rule scales the gradient inversely
proportionally to the $\ell_2$ norm of the past gradients (via the $v_{t-1}$
term) and current gradient $|g_t|^2$:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) |g_t|^2$$

We can generalize this update to the $\ell_p$ norm. Note that Kingma and Ba
also parameterize $\beta_2$ as $\beta_2^p$:

$$v_t = \beta_2^p v_{t-1} + (1 - \beta_2^p) |g_t|^p$$

Norms for large $p$ values generally become numerically unstable, which is why
$\ell_1$ and $\ell_2$ norms are most common in practice. However, $\ell_\infty$ also
generally exhibits stable behavior. For this reason, the authors propose AdaMax
(Kingma and Ba, 2015) and show that $v_t$ with $\ell_\infty$ converges to the
following more stable value. To avoid confusion with Adam, we use $u_t$ to
denote the infinity norm-constrained $v_t$:
$$\begin{aligned}
u_t &= \lim\limits_{p \to \infty}(v_t)^{1/p} \\
& = \max(\beta_2 \cdot u_{t-1}, |g_t|)
\end{aligned}
$$


### Nadam

### AMSGrad

References
----------
1. [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
2. [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/neural-networks-3/#sgd)















