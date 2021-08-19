import numpy as np
import scipy as sp
import sklearn as sk
from sklearn import model_selection
from sklearn import decomposition
from sklearn.model_selection import KFold
import sys
np.set_printoptions(threshold=sys.maxsize)
import cvxopt
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from svm_dual import train_svm,print_test_stats,print_training_stats,test_svm


def kernel_SVM(file_name, kernel_type, cross_val):
    gamma = np.array([10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10, 100])
    data = np.loadtxt(file_name, delimiter=',')
    data = sk.preprocessing.scale(data)
    X = np.array(data[:, :data.shape[1] - 1])
    y = np.array(data[:, data.shape[1] - 1]).reshape(-1, 1)
    y = np.where(y == 0, -1, y)
    X_split, X_test, y_split, y_test = model_selection.train_test_split(X, y, test_size=0.2, train_size=0.8)
    w_s = []
    opt_w = []
    b_s = []
    opt_b = []
    opt_gamma = 0
    opt_c = 0
    average_score = []
    c_s = []

    # Reduced Trial Data
    test_X = np.array(X_split[:800])
    test_y = np.array(y_split[:800]).reshape(-1, 1)
    print(test_y.shape)

    # REDUCED % PROBLEM 2 DATA SET
    for i in gamma:
        print("\n Gamma = ", i, "\n")
        w, b, avg_score, hyper_c = train_svm(test_X, test_y, cross_val, kernel_type, i)
        w_s = np.append(w_s, w).reshape((-1, test_X.shape[1]))
        b_s = np.append(b_s, b)
        c_s = np.append(c_s, hyper_c)
        average_score = np.append(average_score, avg_score).reshape((-1, cross_val))

    col_mean = np.mean(average_score, axis=0)
    max_index = np.argmax(col_mean)
    opt_w = w_s[max_index]
    opt_b = b_s[max_index]
    opt_c = c_s[max_index]
    opt_gamma = gamma[max_index]

    # Testing Hold Out Data
    print("AVERAGES (# CORRESPONDS TO FOLDS) FOR EACH GAMMA")
    print(average_score)
    test_accuracy = test_svm(X_test, y_test, opt_w, opt_b)
    print_test_stats(test_accuracy, opt_c, opt_gamma)

    # FULL PROBLEM 2 DATA SET
    # w,b = train_svm(X_split, y_split, 4)


def main():
    # Should I be using validation error or training error (max) to decide on model?
    # Why is my validation error so low?
    # Is there a more linear algebra way (like the linear kernel) to optimize the
    # RBF kernel calculation? Besides a double for loop?
    kernel_SVM("/content/drive/My Drive/Colab Notebooks/hw2_data_2020.csv", 'rbf', 5)


if __name__ == "__main__":
    main()