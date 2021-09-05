import numpy as np
import matplotlib.pyplot as plt
from Lesson01.week3.testCases import *
from Lesson01.week3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

"""
Reminder: The general methodology to build a Neural Network is to:

1. Define the neural network structure ( # of input units,  # of hidden units, etc).
2. Initialize the model's parameters
3. Loop:
    - Implement forward propagation
    - Compute loss
    - Implement backward propagation to get the gradients
    - Update parameters (gradient descent)

You often build helper functions to compute steps 1-3 and then merge them into one function we call nn_model().
Once you've built nn_model() and learnt the right parameters, you can make predictions on new data.
"""

"""
seed() 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随即数都相同，
如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
"""
np.random.seed(1)

X, Y = load_planar_dataset()  # X.shape:(2, 400)  Y.shape:(1, 400)


# 1.Define the neural network structure
def layer_sizes(X, Y):
    n_x = X.shape[0]  # # size of input layer     n_x = 2
    n_h = 4  # size of hidden layer
    n_y = Y.shape[0]  # size of output layer     n_y = 1

    return n_x, n_h, n_y


# 2.Initialize the model's parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01  # (4, 2)
    b1 = np.zeros(shape=(n_h, 1))  # (4, 1)
    W2 = np.random.randn(n_y, n_h) * 0.01  # (1, 4)
    b2 = np.zeros(shape=(n_y, 1))  # (1, 1)

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1  # Z1.shape:(4, 400)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2  # Z2.shape:(1, 400)
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]  # m = 400

    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))  # logprobs.shape:(1, 400)
    cost = -(1 / m) * np.sum(logprobs)

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17

    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, Y):
    m = Y.shape[1]

    A2 = cache["A2"]
    A1 = cache["A1"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y  # (1, 400)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)  # (1, 4)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (1, 1)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))  # (4, 400)
    dW1 = (1 / m) * np.dot(dZ1, X.T)  # (4, 2)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # (4, 1)

    grads = {"dZ2": dZ2,
             "dW2": dW2,
             "db2": db2,
             "dZ1": dZ1,
             "dW1": dW1,
             "db1": db1}

    return grads


def update_parameters(parameters, grads, learning_rate):
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# Integrate parts in nn_model()
def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    np.random.seed(3)  # 指定随机种子

    n_x = layer_sizes(X, Y)[0]  # n_x = 2
    n_y = layer_sizes(X, Y)[2]  # n_y = 1

    parameters = initialize_parameters(n_x, n_h, n_y)

    """
    3. Loop:
        - Implement forward propagation
        - Compute loss
        - Implement backward propagation to get the gradients
        - Update parameters (gradient descent)
    """
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)  # 与之前不同
        grads = backward_propagation(parameters, cache, Y)  # 与之前不同
        parameters = update_parameters(parameters, grads, learning_rate=0.5)

        if i % 1000 == 0 and print_cost:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)  # numpy.round_(arr, decimals = 0, out = None)：此數學函數將數組四舍五入為給定的小數位數。

    return predictions


parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# 绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

plt.figure(figsize=(16, 32))  # figsize：宽和高，单位是英尺
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
