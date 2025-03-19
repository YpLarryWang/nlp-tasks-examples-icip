# `PyTorch`基础知识备忘录

`PyTorch`详细教程请参考[Stanford CS@224N PyTorch Tutorial CoLab](https://colab.research.google.com/drive/1Pz8b_h-W9zIBk1p2e6v-YFYThG1NkYeS?usp=sharing)（**推荐**）和[PyTorch官网教程](https://pytorch.org/tutorials/)，但注意PyTorch官网的教程内容主要是由社区成员贡献的，因此质量参差不齐，参考时需谨慎。

本备忘录不会罗列知识，而是记录一些容易忘记和混淆的细节。

## FAQ

**Q**: `torch`的`float32`, `float64`和`Python`的`float`, `double`有什么区别？

**A**: 在PyTorch中，`torch.float32` 和 `torch.float64` 是用于张量的两种常用浮点数数据类型，而 Python 的 `float` 和 `double` 与它们稍有不同。以下是它们之间的主要区别：

<!-- --- -->

### 1. `torch.float32` vs. `torch.float64`
- **`torch.float32`**:
  - 对应 **32位**（4字节）的浮点数格式。
  - 通常符合 IEEE 754 标准的单精度浮点数。
  - 精度较低，但计算速度较快，且占用内存更少。
  - 常用于深度学习和其他计算密集型任务。

- **`torch.float64`**:
  - 对应 **64位**（8字节）的浮点数格式。
  - 符合 IEEE 754 标准的双精度浮点数。
  - 精度较高，但计算速度慢，占用内存较大。
  - 通常用于需要高精度的科学计算。

在计算上，`torch.float32` 和 `torch.float64` 分别相当于 NumPy 的 `numpy.float32` 和 `numpy.float64`。

<!-- --- -->

### 2. Python 的 `float`
- 在 Python 中，内置的 `float` 是 **64位** IEEE 754 双精度浮点数，与 C 语言的 `double` 相同。
- **`float` 在 Python 中的特性**：
  - 默认的小数点数值会被解释为 64位浮点数。
  - 等价于 `numpy.float64` 和 `torch.float64`。
  - Python 没有单独的 `float32` 或 `float16`，如果需要这些位宽的浮点数，需借助 NumPy、PyTorch 等库。

注意Python 中没有名为 `double` 的数据类型，因为 `float` 已经是 64位（双精度）。

<!-- --- -->

### 3. 异同总结


| **特性**               | **torch.float32**    | **torch.float64**    | **Python float**    |
|------------------------|----------------------|----------------------|---------------------|
| **位宽**              | 32位（4字节）         | 64位（8字节）         | 64位（8字节）        |
| **标准**              | IEEE 754 单精度浮点数 | IEEE 754 双精度浮点数 | IEEE 754 双精度浮点数 |
| **精度**              | 较低                  | 较高                  | 高                   |
| **计算效率**          | 较快                  | 较慢                  | 相当于 `float64`     |
| **内存占用**          | 少                   | 多                   | 多                    |


---

**Q**: 一整个`torch.tensor`的`dtype`是一样的吗？如果是的话，如何查看整个`tensor`的`dtype`?

**A**: 在 PyTorch 中，一个张量 (`torch.tensor`) 的所有元素的数据类型 (`dtype`) 是一致的。我们不能在同一个张量中混合使用不同的数据类型。这是为了确保在数值运算时有一致的行为和性能优化。

如果我们想检查一个 PyTorch 张量的 `dtype`，可以使用张量对象的 `.dtype` 属性。以下是一个简单的例子：

```python
import torch

# 创建一个 PyTorch 张量
data = torch.tensor([
                     [0.11111111, 1],
                     [2, 3],
                     [4, 5]
                    ], dtype=torch.float32)

# 查看张量的 dtype
print(data.dtype)  # 输出：torch.float32
```

如上所示，通过调用 `data.dtype`，我们可以看到该张量的数据类型是 `torch.float32`。

<!-- 这种统一的数据类型设计不仅简化了计算过程，也提高了在设备上执行的效率。在需要使用不同数据类型的时候，一般采用多个张量分别存储不同数据类型的数值。 -->

---

**Q**: 如何理解张量的维度以及维度$\geq 3$的张量？

**A**: 在 PyTorch 中，张量的维度（或者称为轴，`dimension` 或 `dim`）是通过张量的形状确定的。每一个维度对应于张量的某一层级的数据组织结构。张量的维度从 0 开始计数，`dim=0` 是张量最外层的维度。

### 理解 PyTorch 张量维度

1. **标量（0维张量）**：没有维度，仅仅是一个数值。例如，`torch.tensor(5)`。

2. **向量（1维张量）**：包含一个轴，如一个列表。例如，`torch.tensor([1, 2, 3])`，这个张量的形状是 `(3,)`，表示它有3个元素且只有一个维度。

3. **矩阵（2维张量）**：有两个轴，例如一个二维数组。`torch.tensor([[1, 2, 3], [4, 5, 6]])` 形状为 `(2, 3)`，表示有2行3列。

4. **更高维度张量（3维及以上）**：如一个三维数组。考虑下面的例子：

   ```python
   import torch

   # 创建一个三维张量
   data = torch.tensor([
       [
           [1, 2, 3],
           [4, 5, 6]
       ],
       [
           [7, 8, 9],
           [10, 11, 12]
       ]
   ])

   print(data.shape)  # 输出：torch.Size([2, 2, 3])
   ```

   - 第一个维度 (`dim=0`) 有2个子数组，这可以视为一个“块”。
   - 第二个维度 (`dim=1`) 每个块包含2行。
   - 第三个维度 (`dim=2`) 每行包含3个元素。

### 直观理解多维张量

- **`dim=0`**：这个通常代表批次（batch），在深度学习中用得比较多，它是最外层的维度。
- **`dim=1`**：通常是数据条目的数量。
- **`dim=2`**：可能是特征或时间步长。
- **维度超过3层时**：可以想象是更复杂的嵌套容器。比如对于图像数据，可能有 `(batch_size, channels, height, width)`，非常常见于图像处理。

总之，张量的维度表示的是数据的结构和排列方式。当维度较多时，可以想象这些数据是嵌套的多层容器，彼此之间以一定规律排列。

---

**Q**: 在使用下面的代码时：
```python
# Create an example tensor
# requires_grad parameter tells PyTorch to store gradients
x = torch.tensor([2.], requires_grad=True)

# Print the gradient if it is calculated
# Currently None since x is a scalar
pp.pprint(x.grad)

# Calculating the gradient of y with respect to x
y = x * x * 3 # 3x^2
y.backward()
pp.pprint(x.grad) # d(y)/d(x) = d(3x^2)/d(x) = 6x = 12
```
PyTorch为什么能够求出梯度？它为什么知道$y$是$x$的函数？


**A**: 在PyTorch中，当我们执行上述代码时，PyTorch能够计算梯度的原因在于其自动微分（Autograd）机制。让我来解释这是如何工作的：

### PyTorch的自动微分机制

1. **计算图的构建**：
   当我们使用PyTorch的张量（Tensor）进行运算时，如果这些张量的`requires_grad`属性设置为`True`，PyTorch会在背后构建一个计算图，记录所有操作及其输入输出关系。

2. **操作的追踪**：
   在我们的例子中，`x * x * 3`这个操作被记录下来。PyTorch知道`y`是关于`x`的函数，因为`y`是通过对`x`执行数学运算得到的。

3. **梯度计算**：
   当我们调用`y.backward()`时，PyTorch使用链式法则沿着计算图反向传播，计算每个需要梯度的张量的梯度。

### 在本例中

假设`x`的值是2，那么：
- `y = x * x * 3` 计算得 `y = 2 * 2 * 3 = 12`
- 当调用`y.backward()`时，PyTorch计算`dy/dx = 6x = 6*2 = 12`

<!-- ### 完整代码示例

```python
import torch
import pprint as pp

# 创建一个需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)

# 定义计算
y = x * x * 3  # 3x^2

# 计算梯度
y.backward()

# 打印梯度值
pp.pprint(x.grad)  # 输出: tensor(12.)
``` -->

注意，需要确保`x`是一个设置了`requires_grad=True`的PyTorch张量，这样PyTorch才会跟踪对它的操作并计算梯度。如果没有设置这个属性，PyTorch不会为该张量构建计算图。