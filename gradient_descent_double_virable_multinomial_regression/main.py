"""
考虑多项式回归，具体来说，我们试图拟合一个二次函数。
"""
import numpy as np

# 数据
x = np.array([-2, -1, 0, 1, 2])
y = np.array([4, 1, 0, 1, 4])
m = len(x)

# 初始化参数
w1, w2, b = 0, 0, 0

# 学习率和迭代次数
alpha = 0.01
iterations = 10000

# 梯度下降
for _ in range(iterations):
    # 计算预测值
    y_pred = w1 * x + w2 * x ** 2 + b

    # 计算误差
    error = y_pred - y

    # 计算梯度
    dw1 = (2 / m) * np.dot(x, error)
    dw2 = (2 / m) * np.dot(x ** 2, error)
    db = (2 / m) * np.sum(error)

    # 更新参数
    w1 -= alpha * dw1
    w2 -= alpha * dw2
    b -= alpha * db

print("Optimized w1:", w1)
print("Optimized w2:", w2)
print("Optimized b:", b)
