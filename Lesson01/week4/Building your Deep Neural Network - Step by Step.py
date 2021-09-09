import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Lesson01.week4.dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from Lesson01.week4.lr_utils import load_dataset

"""
1.Initialize the parameters for a two-layer network and for an L-layer neural network.
2.Implement the forward propagation module (shown in purple in the figure below).
  2.1 Complete the LINEAR part of a layer's forward propagation step (resulting in Z^[l]).
  2.3 We give you the ACTIVATION function (relu/sigmoid).
  2.3 Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
  2.4 Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.
3.Compute the loss.
4.Implement the backward propagation module (denoted in red in the figure below).
  4.1 Complete the LINEAR part of a layer's backward propagation step.
  4.2 We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward)
  4.3 Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
  4.4 Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
5.Finally update the parameters.
"""

# load dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# reshape
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# standard
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255
train_y = train_set_y
test_y = test_set_y

m_train = train_set_x_orig[0]
num_px = train_set_x_orig[1]
m_test = test_set_x_orig[0]


# 两层神经网络：初始化参数
def initialize_parameters(layers_dims):
    n_x, n_h, n_y = layers_dims

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 实现前向传播的线性部分
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


# 实现LINEAR-> ACTIVATION 这一层的前向传播
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# 计算代价
def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = - np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / m
    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


# 为单层实现反向传播的线性部分（第L层）
def linear_backword(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dW.shape == W.shape)
    assert (dA_prev.shape == A_prev.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# 实现LINEAR-> ACTIVATION层的后向传播
def linear_activation_backword(dAL, cache, activaction):
    linear_cache, activaction_cache = cache
    if activaction == "relu":
        dZ = relu_backward(dAL, activaction_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    elif activaction == "sigmoid":
        dZ = sigmoid_backward(dAL, activaction_cache)
        dA_prev, dW, db = linear_backword(dZ, linear_cache)

    return dA_prev, dW, db


# 使用梯度下降更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(0, L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters


# 搭建两层神经网络
# 该模型可以概括为：INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT
def two_layer_model(X, Y, layers_dims, num_iterations, learning_rate, print_cost):
    np.random.seed(1)
    grads = {}
    costs = []

    # 初始化参数
    parameters = initialize_parameters(layers_dims)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(num_iterations):
        # 前向传播
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # 计算代价
        cost = compute_cost(A2, Y)

        # 反向传播
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backword(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backword(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    # 迭代完成，绘图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


"""测试与预测双层神经网络结构"""
# layers_dims = [12288, 7, 1]
# parameters = two_layer_model(train_x, train_y, layers_dims, num_iterations=2500, learning_rate=0.0075, print_cost=True)


# 多层神经网络：初始化参数
def initialize_parameters_deep(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


# 实现[LINEAR-> RELU] *（L-1）-> LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION
def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    chches = []

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        chches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    chches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, chches


# 对[LINEAR-> RELU] *（L-1） -> LINEAR -> SIGMOID组执行反向传播，就是多层网络的向后传播
def L_model_backward(AL, Y, caches):
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    L = len(caches)
    grads = {}

    current_cache = caches[-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backword(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backword(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# 搭建多层神经网络
# 实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID
def L_layer_model(X, Y, layer_dims, num_iterations, learning_rate, print_cost):
    np.random.seed(1)
    costs = []

    # 初始化参数
    parameters = initialize_parameters_deep(layer_dims)

    for i in range(0, num_iterations):
        # 前向传播
        AL, caches = L_model_forward(X, parameters)

        # 计算代价
        cost = compute_cost(AL, Y)

        # 反向传播
        grads = L_model_backward(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    # 迭代完成，绘图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


layer_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, layer_dims, num_iterations=2500, learning_rate=0.0075, print_cost=True)

