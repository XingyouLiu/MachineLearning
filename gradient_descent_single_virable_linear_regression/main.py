"""
以一个简单的线性回归问题为例来演示梯度下降。假设有以下数据点：
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
"""

# 导入numpy库并为其创建别名np
import numpy as np

# 使用np.array()创建两个一维的numpy数组来表示数据点的x和y坐标
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 获取x数组的长度，这代表数据点的数量
m = len(x)

# 初始化模型参数w和b为0
w = 0
b = 0

# 设置学习率alpha和迭代次数iterations
alpha = 0.01
iterations = 10000

#定义损失函数的计算：
J=0
def cost_function(x, y, m, w, b, J):
    for i in range(m):
        xi = x[i]
        yi = y[i]
        J += 1 / m * (w * xi + b - yi) ** 2
    return J

# 开始梯度下降的迭代过程
def gradient_descent(x, y, m, w, b, alpha, iterations):
    for i in range(iterations):
        """
        本函数中运用了矢量化方法：
        矢量化是一种编程技术，它允许你在不使用明显的循环和迭代的情况下对数组或序列执行操作。
        操作都是在整个数组上一次性执行的，而不是在循环中一个接一个地执行。
        """
        # 对于每个数据点x，计算模型的预测值y_pred
        y_pred = w * x + b

        # 计算预测值y_pred与真实值y之间的误差
        error = y_pred - y

        # 计算损失函数J关于w的偏导数
        dw = (1 / m) * np.dot(x, error)  #np.dot()是计算两个数组（向量）的点积

        # 计算损失函数J关于b的偏导数
        db = (1 / m) * np.sum(error)

        # 使用计算出的偏导数来更新模型参数w和b
        w -= alpha * dw
        b -= alpha * db

    return w, b

#也可以不用numpy中的数组运算方法：
def gradient_descent_primary_version(x, y, m, w, b, alpha, iterations):
    for i in range(iterations):
        dw = 0
        db = 0
        for j in range(m):
            y_pred_j = w * x[j] + b
            error_j = y_pred_j - y[j]
            dw_j = x[j] * error_j
            db_j = error_j
            dw += dw_j
            db += db_j
        dw *= 1/m
        db *= 1/m
        w -= alpha * dw
        b -= alpha * db
    return w, b

"""
在第一种方法中，你使用了NumPy的向量化操作和点积（np.dot()）来计算参数更新。这种方法非常高效，因为NumPy在底层使用了优化过的C语言和Fortran库来进行数值计算。
在第二种方法中，你使用了Python的原生for循环来计算参数更新。这种方法在计算上相对较慢，因为Python的循环操作没有C语言那么高效。
虽然两种方法在数学上是等价的，但由于浮点数运算的复杂性，它们在计算机上的实现可能会导致细微的数值差异。这些差异通常是由于舍入误差和数值计算的不稳定性造成的。在实际应用中，这种微小的差异通常可以忽略不计。
"""


w, b = gradient_descent(x, y, m, w, b, alpha, iterations)
w1, b1 = gradient_descent_primary_version(x, y, m, w, b, alpha, iterations)
# 输出优化后的模型参数w和b
print("Optimized w:", w)
print("Optimized b:", b)
print("Optimized w1:", w1)
print("Optimized b1:", b1)

#用此w、b计算cost function：J（w，b）
J_w_b = cost_function(x, y, m, w, b, J)
print(J_w_b)

J_w_b = cost_function(x, y, m, w1, b1, J)
print(J_w_b)