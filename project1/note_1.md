# project1

When activation function is step function:
$$
\begin{cases}
  \mathrm{net} = w_1 x_1 + w_2 x_2 + w_3 x_3 \\
  i = 1, 2, 3
\end{cases}
$$

$$
\mathrm{net}=\sum_{i=1}^3 w_i x_i\quad  E = \frac{1}{2}(d - o)^2\quad \frac{\partial E}{\partial w_i} = \frac{\partial E}{\partial o} \frac{\partial o}{\partial net} \frac{\partial net}{\partial w_i}
$$
When activation function is step function:
$$
\sigma(x) = \begin{cases}
           0 & \text{if } x < 0 \\
           1 & \text{if } x \geq 0
       \end{cases}
\quad \quad  \frac{\partial E}{\partial w_i}=(d - o)x_i \\
\\
\delta = (d[p][i] - o[i]) / 2
$$


When activation function is  Sigmoid function：
$$
\sigma(x) = 2\frac{1}{1 + e^{-x}} - 1 \quad \quad  \frac{\partial E}{\partial w_i}=(d - o)(1 - o^2)x_i\\
\\
\delta = (d[p][i] - o[i])(1 - o[i] * o[i]) / 2
$$

这个函数的值域是 $(-1, 1)$，而不是标准 Sigmoid 函数的 $(0, 1)$。因此，我们需要重新计算这个函数的导数。

$$
\sigma'(x) = \frac{d\sigma(x)}{dx} = 2\frac{e^{-x}}{(1 + e^{-x})^2} = (1 - \sigma(x)^2)\\
$$

现在我们可以根据这个新的 Sigmoid 函数重新计算权重更新公式。损失函数保持不变：

$$
E = \frac{1}{2}(d - o)^2
$$

损失函数关于权重 $w_i$ 的梯度为：

$$
\frac{\partial E}{\partial w_i} = \frac{\partial E}{\partial o} \frac{\partial o}{\partial net} \frac{\partial net}{\partial w_i} = (d - o)(1 - o^2)x_i
$$

更新权重：

$$
w_i = w_i - \eta \frac{\partial E}{\partial w_i} = w_i + \eta (d - o)(1 - o^2)x_i
$$

这里的公式与程序中的一致：

$$
\delta = (d[p][i] - o[i])(1 - o[i] * o[i]) / 2
$$

这个 $\delta$ 其实就是损失函数关于权重梯度的一半。在程序中，它用于更新权重。

## Team Project I: Part 1

- Write a computer program to realize the perceptron learning rule and the delta learning rule.
- Train a neuron using your program to realize the AND gate. The input pattern and their teacher signals are given as follows:

– Data: (0,0,-1); (0,1,-1); (1,0,-1); (1,1,-1) – Teacher signals: -1, -1, -1, 1

• Program outputs:
 – Weights of the neuron, and
 – Neuronoutputforeachinputpattern.

## Team Project I: Part 2

- Extend the program written in the first step to learning of single layer neural networks.
- The program should be able to design

– Case 1: A single layer neural network with discrete neurons.

– Case2:Asinglelayerneuralnetworkwithcontinuous neurons.

• Test your program using the following data – Inputs: (10,2,-1), (2,-5,-1), (-5,5,-1).
 – Teacher signals: (1,-1,-1), (-1,1,-1), and (-1,-1,1)

