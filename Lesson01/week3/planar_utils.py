import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

'''
①提供画图函数
②提供激活函数
③提供数据生成函数
④提供数据生成函数
'''


# 划分格子-得到每个格点的坐标-对每个格点的值进行预测(作为颜色区分)-画边界线-加上训练数据的散点图
def plot_decision_boundary(model, X, y):  # (lambda x: predict(parameters, x.T), X, Y)
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1  # x_min就是X的第0行最小值再减一，...
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    # 生成网格矩阵
    # 画格子图理解：1.xx返回每个格点的x坐标  2.yy返回的是每个格点的y坐标
    # xx和yy是两个大小相等的矩阵
    # xx和yy是提供坐标（xx,yy）
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    # model()括号里的参数是每个格点的坐标
    # #model的功能得到对应格点的预测结果 0或者1
    # numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能
    Z = model(np.c_[xx.ravel(), yy.ravel()])  # np.c_[xx.ravel(), yy.ravel()]就是model里的参数x
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 找边界线
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example   ---> (m, 2)
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue) ---> (m, 1)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))  # j=0时:range(200*0, 200*(0+1))  j=1时:range(200*1, 200*(1+1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta     ---> t.shape:(N,)
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius  ---> r.shape:(N,)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]  # ---> np.c_[...省略号...].shape:(N, 2)，最终X.shape:(m, 2)
        Y[ix] = j  # 最终Y.shape:(m, 1)

    X = X.T  # X.shape:(2, m)
    Y = Y.T  # Y.shape:(1, m)

    return X, Y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
