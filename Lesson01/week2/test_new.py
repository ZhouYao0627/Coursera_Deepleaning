# 单层神经网络，不含隐藏层
"""
建立神经网络的主要步骤：
1.设计模型的结构（例如有几个输入特征）
2.初始化模型的参数
3.循环
  3.1计算当前的损失(正向传播)
  3.2计算当前的梯度(反向传播)
  3.3更新参数(梯度下降)

目前的思想是：先整体，后局部，从点到面
1.首先建立整体的架构
2.然后慢慢向其中填充
"""

import numpy as np
import matplotlib.pyplot as plt
from Lesson01.week2.lr_utils import load_dataset

# 加载数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 转置为一行数据
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 转置后为(12288,209)  未转置前应是(209,12288)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T  # 转置后为(12288,50)  未转置前应是(50,12288)

# 将其标准化(采取了简易方法)
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))  # w现在是(12288, 1)
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]  # m为样本数，即图片数209

    # 前向传播
    A = sigmoid(np.dot(w.T, X) + b)  # 此时A ---> (1, 209)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)  # (12288, 1)
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


# 在optimize里计算出最优的参数
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    costs = []  # list of all the costs computed during the optimization, this will be used to plot the learning curve

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]  # 209
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)  # (12288, 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost=False):
    # 初始化W,b
    w, b = initialize_with_zeros(X_train.shape[0])  # w现在是(12288, 1), b = 0

    # 梯度下降
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从字典中找出w, b
    w = parameters["w"]
    b = parameters["b"]

    # 预测 test/train set examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))  # train accuracy: 99.04306220095694 %
    print(
        "test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))  # test accuracy: 70.0 %

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# 绘制学习曲线
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("Learning rate = " + str(d['learning_rate']))
plt.show()
