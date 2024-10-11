import numpy as np

# 随机种子确保每次运行都能获得相同的结果
np.random.seed(0)

# 生成随机房屋数据
# 特征1：大小（平方英尺），范围从1000到2000
# 特征2：卧室数量，范围从2到5
# 特征3：年龄（年），范围从5到50
# 特征4：浴室数量，范围从1到3
size = np.random.randint(1000, 2000, 100)
bedrooms = np.random.randint(2, 6, 100)
age = np.random.randint(5, 50, 100)
bathrooms = np.random.randint(1, 4, 100)

# 合成特征数据
X = np.column_stack((size, bedrooms, age, bathrooms))  # X.shape = (100, 4)

# 生成房屋价格目标变量，这里我们使用一个简单的线性模型
# 为简化问题，我们这里设置的权重比较整齐，并假设没有噪声
weights = np.array([3, 100, -1, 50])  # 真实情况下这些权重是未知的
bias = 100  # 偏差项

# 计算目标变量
y = np.dot(X, weights) + bias   # y.shape = (100, )

print(X.shape)
# 输出前五个样本作为查看
print("前五个样本的特征:")
print(X[:5, :])
print("前五个样本的目标价格:")
print(y[:5])

w = np.zeros(4)   # w.shape = (4, )
b = 0
m = len(X[:,0])

iterations = 1000000
alpha = 0.0001

def feature_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_norm = feature_normalize(X)

def normalize(y):
    mean_y = np.mean(y)
    std_y = np.std(y)
    return (y - mean_y) / std_y, mean_y, std_y

y_norm, mean_y, std_y = normalize(y)

def gradient_descent(X, y, w, b, iterations, alpha, m):
    for i in range(iterations):
        y_pred = np.dot(X, w) + b  # p_pre.shape = (100, )

        errors = y_pred - y  # error.shape = (100, )

        dw = 2/m * np.dot(X.T, errors)   #X.T.shape = (4,100), errors.shape = (100, ), dw = (4, )
        db = 2/m * np.sum(errors)

        w = w - alpha * dw
        b = b - alpha * db

    return w, b

w, b =  gradient_descent(X_norm, y_norm, w, b, iterations, alpha, m)
print(w, b)

def cost_function(X, y, w, b, m):
    y_pred = np.dot(X, w) + b  # p_pre.shape = (100, )
    errors = y - y_pred  # error.shape = (100, )
    return 1/m * np.sum(np.dot(errors.T, errors))

print(cost_function(X_norm, y_norm, w, b, m))


