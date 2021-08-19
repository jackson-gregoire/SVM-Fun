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


def multi_svm(file_name, kernel_type, cross_val):
    gamma = np.array([10 ** (-3), 10 ** (-2), 10 ** (-1), 1.0, 10, 100])
    X = np.loadtxt(file_name)
    X = sk.preprocessing.scale(X)
    y = []
    for i in range(10):
        y_i = np.ones((200, 1)) * i
        y = np.append(y, y_i)
    X_split, X_test, y_split, y_test = model_selection.train_test_split(X, y, test_size=0.2, train_size=0.8)
    averaged_validation_scores = []
    w_s = []
    opt_w = []
    b_s = []
    opt_b = []
    opt_gamma = []
    opt_c = []
    average_score = []
    c_s = []

    # Reduced Trial Data
    test_X = np.array(X_split[:600])
    test_y = np.array(y_split[:600]).reshape(-1, 1)

    # Full Data
    # test_X = X_split
    # test_y = y_split.reshape(-1,1)

    # Training SVM via KFold for % OF PROBLEM 4 DATA
    # 0 as last arguement is the linear kernel
    for i in range(3):
        print("Class", i, "vs Rest:")
        temp_y = np.where(test_y == i, 1, -1)

        for j in gamma:
            print("\n Gamma = ", j, "\n")
            w, b, avg_score, hyper_c = train_svm(test_X, temp_y, cross_val, kernel_type, j)
            w_s = np.append(w_s, w).reshape((-1, test_X.shape[1]))
            b_s = np.append(b_s, b)
            c_s = np.append(c_s, hyper_c)
            average_score = np.append(average_score, avg_score).reshape((-1, cross_val))

        col_mean = np.mean(average_score, axis=0)
        averaged_validation_scores = np.append(averaged_validation_scores, col_mean[np.argmax(col_mean)])
        max_index = np.argmax(col_mean)
        opt_w = np.append(opt_w, w_s[max_index,:]).reshape((-1, test_X.shape[1]))
        opt_b = np.append(opt_b, b_s[max_index])
        opt_c = np.append(opt_c, c_s[max_index])
        opt_gamma = np.append(opt_gamma, gamma[max_index])

    yk_max_index = np.argmax(averaged_validation_scores)
    w = opt_w[yk_max_index,:]
    b = opt_b[yk_max_index]
    C_i = opt_c[yk_max_index]
    gam = opt_gamma[yk_max_index]

    # Testing Hold Out Data
    y_test = np.where(y_test == yk_max_index, 1, -1)
    test_accuracy = test_svm(X_test, y_test, w, b)
    print_test_stats(test_accuracy, C_i, gam)


def main():
    # LINEAR MULTICLASS SVM FOR REDUCED DATA
    multi_svm("/content/drive/My Drive/Colab Notebooks/data.txt", 'linear', 5)

    # RBF MULTICLASS SVM FOR REDUCED DATA
    # multi_svm("/content/drive/My Drive/Colab Notebooks/data.txt", 'rbf',5)


if __name__ == "__main__":
    main()