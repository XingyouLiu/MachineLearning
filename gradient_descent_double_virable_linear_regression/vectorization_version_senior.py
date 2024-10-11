import numpy as np

# 数据
# 将特征合并为一个5x2的矩阵，每行表示一个数据点，每列表示一个特征。
# 这样可以让我们利用矩阵运算来简化计算。
X = np.array([[1500, 3], [1600, 3], [1700, 4], [1800, 4], [1900, 5]], dtype=np.float64)
# 目标值，即房价
y = np.array([6450, 6750, 7600, 8000, 8500], dtype=np.float64)
# 数据点的数量
m = len(y)

# 初始化参数
# 权重初始化为0
w = np.zeros(2)
# 偏置初始化为0
b = 0

# 学习率和迭代次数
# 学习率决定了参数更新的步长
alpha = 0.000001
# 迭代次数决定了优化过程的总步数
iterations = 100000

# 梯度下降
for _ in range(iterations):
    # 计算预测值
    # 使用矩阵乘法来一次计算所有数据点的预测值
    y_pred = X.dot(w) + b   #X.dot(w)等同于np.dot(X,w)

    # 计算误差
    # 误差是预测值和实际值之间的差异
    error = y_pred - y

    # 计算梯度
    # dw是权重的梯度，db是偏置的梯度
    dw = (2 / m) * X.T.dot(error)   #X.T为矩阵X的转置
    db = (2 / m) * np.sum(error)

    # 更新参数
    # 使用梯度下降公式来更新权重和偏置
    w -= alpha * dw
    b -= alpha * db

# 输出优化后的参数值
print("Optimized w:", w)
print("Optimized b:", b)
