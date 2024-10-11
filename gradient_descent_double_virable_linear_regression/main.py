"""
虑多元线性回归，即有多个特征的情况。假设我们有以下数据点：

特征1 (x1): 房子的大小（单位：平方英尺）
特征2 (x2): 房子的卧室数量

我们希望预测房价 (y)。

以下是一些示例数据：
x1 = [1500, 1600, 1700, 1800, 1900]
x2 = [3, 3, 4, 4, 5]
y  = [6450, 6750, 7600, 8000, 8500]

模型定义为：y=w1 * x1 + w2 * x2 + b

我们的目标是通过梯度下降找到最佳的权重 w1 w2 和偏置 b
"""
import numpy as np

# 数据
x1 = np.array([1500, 1600, 1700, 1800, 1900])
x2 = np.array([3, 3, 4, 4, 5])
y = np.array([6450, 6750, 7600, 8000, 8500])
m = len(x1)

# 初始化参数
w = np.zeros(2)
b=0

# 学习率和迭代次数
alpha = 0.00000001
iterations = 100000

# 梯度下降
for _ in range(iterations):
    # 计算预测值
    y_pred = w[0] * x1 + w[1] * x2 + b

    # 计算误差
    error = y_pred - y

    # 计算梯度
    dw1 = (2 / m) * np.dot(x1, error)
    dw2 = (2 / m) * np.dot(x2, error)
    dw = np.array([dw1, dw2])
    db = (2 / m) * np.sum(error)

    # 更新参数
    w -= alpha * dw
    b -= alpha * db

print("Optimized w1:", w[0])
print("Optimized w2:", w[1])
print("Optimized b:", b)