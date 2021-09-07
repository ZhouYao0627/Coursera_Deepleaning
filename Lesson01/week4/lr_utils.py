import numpy as np
import h5py


# 与week2中的Ir_utils一样
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
