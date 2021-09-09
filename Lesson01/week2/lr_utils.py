import numpy as np
import h5py

"""
train_dataset是文件类型，不能查看其shape，即'File' object has no attribute 'shape'

train_set_x_orig ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
train_set_y_orig ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
test_set_x_orig ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
test_set_y_orig ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。

list(train_dataset.keys()) ---> ['list_classes', 'train_set_x', 'train_set_y']
list(test_dataset.keys()) ---> ['list_classes', 'test_set_x', 'test_set_y']
train_dataset["train_set_x"].shape ---> (209, 64, 64, 3)
train_dataset["train_set_y"].shape ---> (209,)
test_dataset["test_set_x"].shape ---> (50, 64, 64, 3)
test_dataset["test_set_y"].shape ---> (50,)
train_dataset["list_classes"].sahpe ---> (2,)
test_dataset["list_classes"].sahpe ---> (2,)
"""


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features # (209, 64, 64, 3)
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels # (209,)

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features # (50, 64, 64, 3)
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels # (50,)

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes # (2,)

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # (1, 209)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # (1, 50)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
