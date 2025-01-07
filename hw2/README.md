# Lecture 6

# 牛顿法 

$$
\theta _{t+1}=\theta _{t}-\mu \left[\nabla ^{2}f(\theta )\right]^{-1}\nabla f(\theta )
$$

牛顿法可以直接找到正确的最优点。但是需要求解二阶导矩阵(Hessian矩阵)，但是神经网络中的矩阵往往非常大，计算这个矩阵是完全不切实际的。

# 动量更新

动量更新大部分是前一时刻的模型参数，少部分是梯度更新。介于牛顿法和梯度下降法中间。

动量的数学表达式为：

$$
u_{t+1} = \beta u_t + (1 - \beta) \nabla_\theta f(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha u_{t+1}
$$

其中：

- $\theta_t$ 是当前参数
- $\alpha$ 是学习率（step size）
- $\beta$ 是动量系数（momentum averaging parameter），如常用值 $0.9$
- $\nabla_\theta f(\theta_t)$ 是损失函数关于参数的梯度

在算法初始化时，通常将动量项 $u_t$ 初始化为零向量，即：

$$
u_0 = \mathbf{0}
$$

这意味着在第一次更新时，动量项仅由当前的梯度贡献：

$$
u_1 = (1 - \beta) \nabla_\theta f(\theta_0)
$$

随着迭代进行，动量会逐渐累积之前的梯度信息，使得在下降方向更加稳定，有助于加速收敛并减少震荡。

# Unbias
加入了**unbias**的动量修正是为了在梯度下降的初始阶段消除由于动量初始化为零所带来的偏差（bias）。在常规动量方法中，由于 $u_0$ 通常初始化为零，前几次更新的动量量值较小，可能导致更新不稳定或收敛速度减慢。

为了解决这个问题，可以使用**修正后的无偏动量**，其更新公式为：

$$
\theta_{t+1} = \theta_t - \alpha \frac{u_{t+1}}{1 - \beta^{t+1}}
$$

其中：

- $\beta^{t+1}$ 是动量衰减参数随时间的幂次，修正了初始动量的偏差。
- 该修正项 $\frac{1}{1 - \beta^{t+1}}$ 在初期会放大动量项，以补偿前几次更新的低幅度。

# Nesterov

Nesterov加速梯度是一种改进的动量方法，它通过在计算梯度时引入“前瞻性”，从而提升了优化的收敛速度和稳定性。

### Nesterov 动量的优化思想
Nesterov 动量通过在**计算梯度前**进行一次前向预测，从而在更新参数时考虑未来的变化趋势。具体公式为：

$$
u_{t+1} = \beta u_t + (1 - \beta) \nabla_\theta f(\theta_t - \alpha u_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha u_{t+1}
$$

#### 核心区别：
- 在计算梯度时，不是使用当前位置  $\theta_t$，而是提前使用**预测位置** $\theta_t - \alpha u_t$。
- 这种前瞻性的计算可以更准确地反映下降方向，从而加速收敛。

# Adam
Adam是一种自适应梯度优化算法，它结合了动量方法和自适应学习率的思想，能够在训练深度神经网络时提供更稳定且快速的收敛效果。它通过计算梯度的一阶矩（均值）和二阶矩（方差）进行参数更新。

Adam在每次迭代时更新三个变量：

1. **动量项（梯度的一阶矩估计）**：

$$
u_{t+1} = \beta_1 u_t + (1 - \beta_1) \nabla_\theta f(\theta_t)
$$

- $\beta_1$ 通常取 $0.9$，控制梯度的指数加权平均。
- 这是类似于传统动量的更新，保留了梯度的历史信息。

2. **梯度方差项（二阶矩估计）**：

$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2)(\nabla_\theta f(\theta_t))^2
$$
- $\beta_2\$ 通常取 $0.999$，用于估计梯度平方的移动平均。
- 该项用于估计梯度的方差，能够帮助自适应调整每个参数的学习率。

3. **参数更新**：

$$
\theta_{t+1} = \theta_t - \alpha \frac{u_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
$$
- $\alpha$ 是学习率。
- $\epsilon$ 是一个极小的常数，用于防止分母为零。
- 分母 $\sqrt{v_{t+1}}$ 表示对大梯度方向进行更小的更新，有助于控制学习率随梯度变化的适应性。

---

### **无偏校正（Bias Correction）**
由于 $u_t$ 和 $v_t$ 在初始时被初始化为零，会导致前几次迭代产生偏差。为了解决这个问题，Adam 使用了无偏校正：

动量校正:
$$
\hat{u}_{t+1} = \frac{u_{t+1}}{1 - \beta_1^{t+1}}
$$

梯度方差校正:
$$
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
$$

最终公式:
$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{u}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}
$$

# Lecture 8

- Caffe是原地反向传播
- TensorFlow是静态的计算图，工业界使用，方便优化，可以选择想要输出的节点
- Pytorch是动态计算图，可以根据节点的数值决定网络结构，灵活，但是输出是固定的（无法指定输出节点）

损失函数+优化器+正则化

- 优化器需要保存需要优化的权重，如果使用带动量的更新方式，还需要保存额外的状态（动量等）。
- 正则化有两种方式：1. 直接加在损失函数里 2.优化器内weight decay

在模型更新的时候，为了防止继续生成计算图，更新w的时候使用.data。

计算softmax时，可以对每个输入减去max，防止上溢出。


# Normalization
- Layernorm:对每一个样本归一化（减去均值除以标准差）
- Batchnrom:对一个批次中的样本每一个特征进行归一化。
- Batchnorm的问题：一个样本会受到其他样本影响。改进方法：使用全部数据集的均值和方差。running average/variance。在运行过程中不断更新计算均值和方差。测试的时候使用整体的均值和方差归一化。


# Regularization
- implicit regularization:隐式正则化。现有的部分算法和架构有的可以防止过拟合（SGD）
- Explicit regularization:显示正则化。在神经网络中加入一些特有的层，专门用于regular网络。

1. L2正则化/weight decay
在损失函数中加入正则化项，通过求导可以计算得到，结果就是在更新时，对原来的梯度乘以$(1-\alpha \times \lambda)$，所以一般都写在优化器的weight_dacay里。

2. Dropout
随机将一定比例的输出置零。


# HomeWork

## 1. 初始化
分别实现了`xavier_uniform`,`xavier_normal`和`kaiming_uniform`,`kaiming_normal`。

`kaiming`初始化适用于`ReLU`激活的神经网络。

## 2. Linear
注意初始化时，使用`kaiming`初始化对应维度的输入。需要输入正确的维度，再转置回来。

## 3. LogSumExp
```python
      z = node.inputs[0]
      max_z = z.realize_cached_data().max(self.axes, keepdims=True)
      exp_z = exp(z - max_z)
      sum_exp_z = summation(exp_z, self.axes)
      grad_sum_exp_z = out_grad / sum_exp_z
      expand_shape = list(z.shape)
      axes = range(len(expand_shape)) if self.axes is None else self.axes
      for axis in axes:
         expand_shape[axis] = 1
      grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
      return grad_exp_z * exp_z
```
$$
L = \alpha + \log\left(\sum_{i=1}^{n} e^{z_i - \alpha}\right)
$$

对 $z_k$ 求导：

$$
\frac{\partial L}{\partial z_k} = \frac{e^{z_k - \alpha}}{\sum_{j} e^{z_j - \alpha}}
$$

即

$$
\frac{\partial L}{\partial z_k} = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
$$

我们可以按照上面一样，一项一项求，但是我们可以发现，我们计算得到node的数值，放在exp()内就是指数的求和。因此可以使用node的cache_value来简化代码。

## 4. LogSoftMax

**LogSoftMax** 的梯度公式为：

$$
\frac{\partial L}{\partial z_k} = \frac{\partial L}{\partial \text{LogSoftMax}(z_k)} - \text{SoftMax}(z_k) \cdot \sum_{i=1}^{n} \frac{\partial L}{\partial \text{LogSoftMax}(z_i)}
$$

或者，使用向量化表示：

$$
\frac{\partial L}{\partial \mathbf{Z}} = \mathbf{g} - \mathbf{s} \cdot (\mathbf{g}^T \mathbf{1})
$$

## 5. Laynorm & BatchNorm

BatchNorm时不需要reshape添加维度，因为broadcast会自动右端对齐，在前面广播。
简而言之就是，如果需要往后填充维度，需要人为reshape，在后方添加维度1，然后再广播。如果需要往前填充，直接广播即可。

## 6. Optimizer

weight decay是在梯度更新时，对原来的梯度乘以$(1-\alpha \times \lambda)$

## 7. Dataset & DataLoader

需要reshape读入的数据，重写`__getitem__`和`__len__`即可。


# Summary

- 实现了Xavier和Kaiming初始化（uniform和normal）
- 实现了LogSumExp和LogSoftMax运算符
- Module基类可以获取参数列表，子模块列表，以及设置train和eval模式，通过__call__调用继承的子类的`forward`方法。
- 实现了nn里的各个模块（继承自Module基类）
- 实现了SoftMaxLoss模块（损失函数也是一个Module）
- 实现了batchnorm和layernorm（维度区别/batchnorm running mean/variance）running mean/variance通过动量更新实现
- 实现了各种模块（继承自Module）
- 实现了SGD和Adam优化器
- 优化器继承自Optimizer，包含需要优化的params，`step`更新params，`reset_grad`将grad重置。
- 优化器内加入weight_dacay正则以及动量更新。
- 实际实现是`grad = grad + weight_dacy * param`
- 优化器更新梯度时直接替换param.data，防止引入额外计算图
- 优化器构成：动量，二阶矩（Adam），unbias，weight_decay，学习率，除数加上epsilon。
- 数据集包括Dataset基类和DataLoader基类。
- Dataset子类需要重载`__getitem__`和`__len__`
- DataLoader迭代输出Dataset内容。

