import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Lesson01.week4.dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from Lesson01.week4 import lr_utils

# 加载数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

# reshape and standardize the images before feeding them to the network
# 转置为一行数据   Reshape the training and test examples
train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # 转置后为(12288,209)  未转置前是(209,12288)
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T  # 转置后为(12288,50)  未转置前是(50,12288)

# 将其标准化(采取了简易方法)   Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255  # (12288,209)
train_y = train_set_y  # (1, 209)
test_x = test_x_flatten / 255  # (12288,50)
test_y = test_set_y  # (1, 50)

m_train = train_set_x_orig.shape[0]  # 209
num_px = train_set_x_orig.shape[1]  # 64
m_test = test_set_x_orig.shape[0]  # 50


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
    if activation == 'sigmoid':
        # 先要将Z算出来才能继续
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

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
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# 实现LINEAR-> ACTIVATION层的后向传播
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# 使用梯度下降更新参数
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(0, L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


# 搭建两层神经网络
# 该模型可以概括为：INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT
def two_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    np.random.seed(1)
    grads = {}
    costs = []

    # 初始化参数
    parameters = initialize_parameters(layers_dims)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # 开始进行迭代
    for i in range(0, num_iterations):
        # 前向传播
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # 计算代价
        cost = compute_cost(A2, Y)

        # 反向传播
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

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


"""测试与预测两层神经网络结构"""


# layers_dims = [12288, 7, 1]
# parameters = two_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True)
#
# predictions_train = predict(train_x, train_y, parameters)  # 训练集
# predictions_test = predict(test_x, test_y, parameters)  # 测试集


# 多层神经网络：初始化参数
def initialize_parameters_deep(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):  # 从1开始是因为隐藏层从1开始，第0层为输入层
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

    return parameters


# 实现[LINEAR-> RELU] *（L-1）-> LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION
def L_model_forward(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        # (上行)此时的A为下个前向传播的A_prev
        caches.append(cache)  # 将cache储存起来，留反向传播时使用

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    # (上行)这里的传入的A是经过循环的，最后一个隐藏层的A，就相当于A(L-1)
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


# 对[LINEAR-> RELU] *（L-1） -> LINEAR -> SIGMOID组执行反向传播，就是多层网络的向后传播
def L_model_backward(AL, Y, caches):
    L = len(caches)
    # Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads = {}

    ### 原版方法，只是写法较难理解--->可看链接https://blog.csdn.net/u013733326/article/details/79767169下方讨论
    # current_cache = caches[-1]
    # grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
    #                                                                                               "sigmoid")
    # # 上方两行代码将下方两行代码整合到一起了，更加简易（因为我们已经设计了linear_activation_backward，而它内部已经调用了linear_backward并完成了一系列工作）
    # # grads["dA_prev" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(
    # #     sigmoid_backward(dAL, current_cache[1]), current_cache[0])
    #
    # # 第L个已经在上方求过了，剩下的只有L-1个，然后将其反向遍历，再进行计算; 由于for l in reversed(range(L-1))这句，其实l是从L-2这里开始的
    # for l in reversed(range(L - 1)):
    #     current_cache = caches[l]
    #     dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
    #     grads["dA" + str(l + 1)] = dA_prev_temp
    #     grads["dW" + str(l + 1)] = dW_temp
    #     grads["db" + str(l + 1)] = db_temp

    ### 经过修改的方法，较易理解
    current_cache = caches[-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")
    # 第L个已经在上方求过了，剩下的只有L-1个，然后将其反向遍历，再进行计算; 由于for l in reversed(range(L-1))这句，其实l是从L-2这里开始的
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# 该函数用于预测L层神经网络的结果，当然也包含两层
def predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        p[0, i] = 1 if probas[0, i] > 0.5 else 0

    print("准确度为: " + str(float(np.sum((p == Y)) / m)))

    return p


# 搭建多层神经网络
# 实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    np.random.seed(1)
    costs = []

    # 初始化参数
    parameters = initialize_parameters_deep(layers_dims)

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


"""测试与预测多层神经网络结构"""
layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=2500, print_cost=True)

pred_train = predict(train_x, train_y, parameters)  # 训练集
pred_test = predict(test_x, test_y, parameters)  # 测试集


# 绘制预测和实际不同的图像
def print_mislabeled_images(classes, X, y, p):
    """
    X - 数据集
    y - 实际的标签
    p - 预测
    """

    a = p + y  # 由下行可以理解为，a是预测的与实际不同的图像  a.shape = (1, 50)
    mislabeled_indices = np.asarray(np.where(a == 1))
    # np.where返回的是tuple；这里的np.asarray返回的是2行，a==1个数的列；第一行都是0，第二行里的数是预测错的样本/图片所在的列
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])  # 即样本/图像的数量
    for i in range(num_images):  # 预测错的图片不止一个，故使用循环
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)  # 2行，num_images列，i+1表示从左到右从上到下的第i+1个位置。
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')  # X[:, index]其实就是第index个样本/图像
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
    plt.show()


print_mislabeled_images(classes, test_x, test_y, pred_test)

# 测试自己的图片
my_image = "E:/Pycharm/work/Coursera_Deepleaning/Lesson01/week4/pictures/my_image2.jpg"  # change this to the name of your image file
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)

# 读取图片，将其转化为三通道，并resize为64*64分辨率
image = Image.open(my_image).convert("RGB").resize((num_px, num_px))

# 将图片转化为矩阵形式并reshape以满足模型输入格式
my_image = np.array(image).reshape(num_px * num_px * 3, -1)

my_predict_image = predict(my_image, my_label_y, parameters)
plt.imshow(my_image)
# print("y = " + str(np.squeeze(my_predict_image)) + ", your L-layer model predicts a \"" + classes[
#     int(np.squeeze(my_predict_image))].decode("utf-8") + "\"picture.")
