""""
Instructions:
  -Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.
You will learn to:
  -Build the general architecture of a learning algorithm, including:
    Initializing parameters
    Calculating the cost function and its gradient
    Using an optimization algorithm (gradient descent)
  -Gather all three functions above into a main model function, in the right order.
  
  numpy is the fundamental package for scientific computing with Python.
  h5py is a common package to interact with a dataset that is stored on an H5 file.
  matplotlib is a famous library to plot graphs in Python.
  PIL and scipy are used here to test your model with your own picture at the end.
"""""

# 1 - Packages
# In [1]:
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from Lesson01.week2.lr_utils import load_dataset

# 2 - Overview of the Problem set
# In [2]:
# Loading the data (cat/non-cat)
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()

# In [3]:
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

# In [4]:
### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_y.shape[1] # 训练集里图片的数量。
m_test = test_set_y.shape[1] # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1] # 训练、测试集里面的图片的宽度和高度（均为64x64）。
### END CODE HERE ###

print ("训练集的数量: m_train = " + str(m_train)) # m_train = 209
print ("测试集的数量 : m_test = " + str(m_test)) # m_test = 50
print ("每张图片的宽/高 : num_px = " + str(num_px)) #  num_px = 64
print ("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)") # (64, 64, 3)
print ("训练集_图片的维数 : " + str(train_set_x_orig.shape)) # (209, 64, 64, 3)
print ("训练集_标签的维数 : " + str(train_set_y.shape)) # (1, 209)
print ("测试集_图片的维数: " + str(test_set_x_orig.shape)) # (50, 64, 64, 3)
print ("测试集_标签的维数: " + str(test_set_y.shape)) # (1, 50)

# In [5]:
# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T # 未转置前应是(209,12288)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T # 未转置前应是(50,12288)
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape)) # (12288, 209)
print ("train_set_y shape: " + str(train_set_y.shape)) # (1, 209)
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape)) # (12288, 50)
print ("test_set_y shape: " + str(test_set_y.shape)) # (1, 50)
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0])) # [17 31 56 22 33]

# In [6]:
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

"""
Common steps for pre-processing a new dataset are:
  -Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
  -Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
  -"Standardize" the data
"""

# 3 - General Architecture of the learning algorithm
"""
Key steps: In this exercise, you will carry out the following steps:
- Initialize the parameters of the model
- Learn the parameters for the model by minimizing the cost  
- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude
"""

# 4 - Building the parts of our algorithm
"""
The main steps for building a Neural Network are:
  (1)Define the model structure (such as number of input features)
  (2)Initialize the model's parameters
  (3)Loop:
     Calculate current loss (forward propagation)
     Calculate current gradient (backward propagation)
     Update parameters (gradient descent)
You often build 1-3 separately and integrate them into one function we call model().
"""

# 4.1 - Helper functions
# In [7]:
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s

# In [8]:
print("====================测试sigmoid====================")
print ("sigmoid(0) = " + str(sigmoid(0))) # sigmoid(0) = 0.5
print ("sigmoid(9.2) = " + str(sigmoid(9.2))) # sigmoid(9.2) = 0.9998989708060922

# 4.2 - Initializing parameters
# -Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
#  断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况，
#  例如我们的代码只能在 Linux 系统下运行，可以先判断当前系统是否符合条件。
# -isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。

# In [9]:
# GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros(shape=(dim, 1))
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

# In [10]:
print("====================测试initialize_with_zeros====================")
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w)) # w = [[0.]
                        #     [0.]]
print ("b = " + str(b)) # b = 0

# 4.3 - Forward and Backward propagation
# In [11]:
# GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    ### END CODE HERE ###

    # 使用断言确保我的数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 创建一个字典，把dw和db保存起来。
    grads = {"dw": dw,
             "db": db}

    return grads, cost

# In [12]:
print("====================测试propagate====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"])) # dw = [[0.99993216]
                                   #     [1.99980262]]

print ("db = " + str(grads["db"])) # db = 0.49993523062470574
print ("cost = " + str(cost)) # cost = 6.000064773192205

"""
d) Optimization
  -You have initialized your parameters.
  -You are also able to compute a cost function and its gradient.
  -Now, you want to update the parameters using gradient descent.
"""

# In [13]:
# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

# In [14]:
print("====================测试optimize====================")
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"])) # w = [[0.1124579 ]
                                  #     [0.23106775]]
print ("b = " + str(params["b"])) # b = 1.5593049248448891
print ("dw = " + str(grads["dw"])) # dw = [[0.90158428]
                                   #      [1.76250842]]
print ("db = " + str(grads["db"])) # db = 0.4304620716786828

# In [15]:
# GRADED FUNCTION: predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###

    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        ### END CODE HERE ###

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X))) # predictions = [[1. 1.]]

"""
-Initialize (w,b)
-Optimize the loss iteratively to learn parameters (w,b):
    computing the cost and its gradient
    updating the parameters using gradient descent
-Use the learned (w,b) to predict the labels for a given set of examples
"""

# In [17]:
# 5 - Merge all functions into a model
"""
Implement the model function. Use the following notation:
- Y_prediction for your predictions on the test set
- Y_prediction_train for your predictions on the train set
- w, costs, grads for the outputs of optimize()
"""
# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)) # train accuracy: 99.04306220095694 %
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)) # test accuracy: 70.0 %

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

# In [18]:
print("====================测试model====================")
# Run the following cell to train your model
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# In [19]:
# Example of a picture that was wrongly classified.
index = 5
plt.imshow(test_set_x_orig[index])
plt.show()
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[np.squeeze(test_set_y[:, index])].decode("utf-8") +  "\" picture.")
      # y = 0, you predicted that it is a "non-cat" picture.

# Let's also plot the cost function and the gradients.
# In [20]:
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# 6 - Further analysis (optional/ungraded exercise)
# In [21]:
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

"""
learning rate is: 0.01
train accuracy: 99.52153110047847 %
test accuracy: 68.0 %
------------------------------------------
learning rate is: 0.001
train accuracy: 88.99521531100478 %
test accuracy: 64.0 %
-------------------------------------------
learning rate is: 0.0001
train accuracy: 68.42105263157895 %
test accuracy: 36.0 %
"""

# 1.Preprocessing the dataset is important.
# 2.You implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
# 3.Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm.



















