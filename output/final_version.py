import os
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier


def load_data():
    '''
    load the data, return as the data and label
    '''
    file_name1 = "clean_data_with_target.csv"
    file_name2 = "clean_data_with_target_after_pca.csv"
    pwd = os.getcwd()
    data = pd.read_csv(os.path.join(pwd, "data", file_name1)).values
    data_pca = pd.read_csv(os.path.join(pwd, "data", file_name2)).values
    Y = data[:, -1]
    X = np.delete(data, -1, axis=1)
    Y_pca = data_pca[:, -1]
    X_pca = np.delete(data_pca, -1, axis=1)
    return X, Y, X_pca, Y_pca


def knn_classification(X, Y, k_range, cv):
    '''
    knn classification
    return the figure that value which n is the better
    '''
    train_errors = list()
    test_errors = list()
    for i in k_range:
        print("start train N = ", i)
        train_error = list()
        test_error = list()
        clf_knn = KNeighborsClassifier(n_neighbors=i)
        for train, test in cv.split(X, Y):
            print(X[train].shape)
            print(Y[train].shape)
            clf_knn.fit(X[train], Y[train])
            train_error.append(clf_knn.score(X[train], Y[train]))
            test_error.append(clf_knn.score(X[test], Y[test]))
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))

    plt.figure()
    plt.plot(k_range, train_errors, label='Train')
    plt.plot(k_range, test_errors, label='Test')
    plt.legend(loc='lower left')
    plt.ylim([0, 1.1])
    plt.xlim([0, 21])
    plt.title('Evaluation of KNN model')
    plt.xlabel('The number of N')
    plt.ylabel('Performance')
    plt.show()


def dt_classification(X, Y, cv):
    '''
    decision tree model

    '''
    train_errors = list()
    test_errors = list()
    depths = range(5, 30)
    #depths = range(X.shape[1]-10, X.shape[1])
    for depth in depths:
        train_error = list()
        test_error = list()
        clf_dt = tree.DecisionTreeClassifier(max_depth=depth)
        for train, test in cv.split(X, Y):
            clf_dt.fit(X[train], Y[train])
            train_error.append(clf_dt.score(X[train], Y[train]))
            test_error.append(clf_dt.score(X[test], Y[test]))
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))

    plt.figure()
    plt.plot(depths, train_errors, label='Train')
    plt.plot(depths, test_errors, label='Test')
    plt.legend(loc='lower left')
    plt.ylim([0, 1.1])
    plt.xlim([depths[0]-2, depths[-1]+2])
    plt.title('Evaluation of DT model')
    plt.xlabel('The depth')
    plt.ylabel('Performance')
    plt.show()


def rf_classification(X, Y, cv):
    train_errors = list()
    test_errors = list()
    tree_nums = range(5, 100, 5)
    for tree_num in tree_nums:
        train_error = list()
        test_error = list()
        clf_rf = RandomForestClassifier(n_estimators=tree_num)
        for train, test in cv.split(X, Y):
            clf_rf.fit(X[train], Y[train])
            train_error.append(clf_rf.score(X[train], Y[train]))
            test_error.append(clf_rf.score(X[test], Y[test]))
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))
    plt.figure()
    plt.plot(tree_nums, train_errors, label='Train')
    plt.plot(tree_nums, test_errors, label='Test')
    plt.legend(loc='lower left')
    plt.ylim([0, 1.1])
    plt.xlim([0, 102])
    plt.title('Evaluation of random forest model')
    plt.xlabel('The number of trees')
    plt.ylabel('Performance')
    plt.show()


def svm_classification(X, Y, cv):
    kernels = [ 'linear','poly', 'rbf', 'sigmoid']
    x = [1, 2, 3, 4]
    train_errors = list()
    test_errors = list()
    for kernel_unit in kernels:
        train_error = list()
        test_error = list()
        clf_svm = svm.SVC(kernel=kernel_unit, probability=True, gamma='scale')
        for train, test in cv.split(X, Y):
            clf_svm.fit(X[train], Y[train])
            train_error.append(clf_svm.score(X[train], Y[train]))
            test_error.append(clf_svm.score(X[test], Y[test]))
        print("the kernel is ", kernel_unit)
        print("the accuracy of train is ", train_error)
        print("the accuracy of test is ", test_error)
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))
    plt.figure()
    plt.scatter(x, train_errors, label='Train')
    plt.scatter(x, test_errors, label='Test')
    plt.xticks(x, kernels, rotation=45)
    plt.legend(loc='lower left')
    plt.ylim([0, 1.1])
    plt.xlim([0, 5])
    plt.title('Evaluation of SVM model')
    plt.xlabel('The Kernel')
    plt.ylabel('Performance')


if __name__ == "__main__":

    X, Y, X_pca, Y_pca = load_data()
    print(X.shape)
    print(Y.shape)
    cv = StratifiedKFold(n_splits=3)
    k_range = range(1, 20, 2)
    #knn_classification(X_pca, Y_pca, k_range, cv)
    #dt_classification(X_pca, Y_pca, cv)
    #rf_classification(X_pca, Y_pca, cv)
    svm_classification(X, Y, cv)