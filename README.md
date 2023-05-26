# vq-gan-101

PyTorch Implementation of the VQ-GAN model.

## The Swish Activation Function

### References

- *Searching for Activation Functions* by Prajit Ramachandran, Barret Zoph, Quoc V. Le at Google Brain, 2017.

**Swish** is an activation function defined as,
$$f(x) = x \cdot \mathrm{sigmoid} (\beta x),$$
where $\beta$ is a learnable parameter and
$$\sigma(z) = (1 + \mathrm{exp}(-z))^{-1}.$$
The activation function is proposed to replace  ReLU with automated search techniques and has shown impressive performance in various image classification and machine translation tasks as compared to the popular ReLU.

In practice, nearly all implementations do not use the learnable parameter $\beta$, in which case the activation function is $x \sigma(x)$ ("Swish-$1$"), which is also equivalent to the Sigmoid-weighted Linear Unit (SiLU) as proposed by Elfwing *et al.* Note that if $\beta = 0$, Swish becomes the scaled linear function $f(x) = x / 2$, and if $\beta \to \infty$, the sigmoid component approaches a $0-1$ function, so Swish behaves like a ReLU function. Thus,
> the Swish activation function can be loosely viewed as a smooth function which nonlinearly interpolates between the linear function and the ReLU function, and the trainable parameter $\beta$ controls the degree of that interpolation.

Similar to ReLU, the Swish activation function is unbounded above and bounded below, but smooth and non-monotonic. The non-monotonicity is of the distinguishing properties of the Swish activation functions as compared other common activation functions. The derivative of the Swish activation function is,
$$
\begin{aligned}
f'(x) &= \sigma(\beta x) + \beta x \cdot \sigma (\beta x) (1 - \sigma(\beta x)) \\
&= \sigma(\beta x) + \beta x \cdot \sigma(\beta x) - \beta x \cdot \sigma(\beta x)^2 \\
&= \beta x \cdot \sigma(x) + \sigma(\beta x)(1 - \beta x \cdot \sigma(\beta x)) \\
&= \beta f(x) + \sigma(\beta x)(1 - \beta f(x)).
\end{aligned}
$$
