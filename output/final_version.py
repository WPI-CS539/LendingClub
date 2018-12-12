import os
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from output.problem2 import *

def load_data():
    '''
    load the data, return as the data and label
    '''
    file_name1 = "clean_data_with_target.csv"
    file_name2 = "clean_data_with_target_after_pca.csv"
    pwd = os.getcwd()
    data = pd.read_csv(os.path.join(pwd, "data", file_name1)).values
    data_pca = pd.read_csv(os.path.join(pwd, "data", file_name2)).values
    np.random.shuffle(data)
    np.random.shuffle(data_pca)
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
    kernels = ['poly', 'rbf', 'sigmoid']
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
    plt.show()

def ANN_classification(X, Y, cv):
    # train_errors = list()
    test_accuracies = list()
    specificities = list()
    x_axis= []
    for i in range(1, 11):
        x_axis.append(0.0001 * (i * 2))
        print("start train N = ", i)
        train_error = list()
        test_accuracy = list()
        specificity = list()
        for train, test in cv.split(X, Y):
            X_train = np.asmatrix(X[train])
            X_test = np.asmatrix(X[test])
            Y_train = Y[train]
            Y_test = Y[test]
            W1, b1, W2, b2 = ann_train(X_train, Y_train, alpha=0.0001*(i * 2), n_epoch=10)
            Y_predict, P = ann_predict(X_test, W1, b1, W2, b2)
            cm = confusion_matrix(Y_test, Y_predict)
            test_accuracy.append((cm[1, 1] + cm[0, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 1] + cm[1, 0]))
            specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
        # train_errors.append(sum(train_error) / len(train_error))
        test_accuracies.append(sum(test_accuracy) / len(test_accuracy))
        specificities.append(sum(specificity) / len(specificity))
    plt.figure()
    plt.plot(x_axis, test_accuracies, label='Test')
    plt.plot(x_axis, specificities, label='specificity')
    plt.legend(loc='lower left')
    plt.title('Evaluation of ANN model')
    plt.xlabel('The number of alpha')
    plt.ylabel('Performance')
    plt.show()



# def NN_classification(X, Y, cv):
#     clf_NN = MLPClassifier(hidden_layer_sizes=(9, 9, 9,),
#                   activation='tanh', batch_size='auto',
#                   learning_rate ='constant', learning_rate_init = 0.001,
#                   max_iter = 200,
#                   shuffle = True,
#                   verbose = True,
#                   warm_start = False,
#                   early_stopping = True, n_iter_no_change = 30, validation_fraction = 0.1,
#                   beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
#
#     X_test = X[20000:25000,:]
#     Y_test = Y[20000:25000]
#     X = X[0:20000,:]
#     Y = Y[0:20000]
#     print(X.shape)
#     print(Y.shape)
#     clf_NN.fit(X, Y)
#     predictions = clf_NN.predict(X_test)
#     print(confusion_matrix(Y_test, predictions))
#     print(classification_report(Y_test, predictions))
#     print(clf_NN.score(X_test, Y_test))


if __name__ == "__main__":

    X, Y, X_pca, Y_pca = load_data()
    cv = StratifiedKFold(n_splits=2)
    ## after pca
    k_range = range(1, 20, 2)
    knn_classification(X_pca, Y_pca, k_range, cv)
    dt_classification(X_pca, Y_pca, cv)
    rf_classification(X_pca, Y_pca, cv)
    #svm_classification(X_pca, Y_pca, cv)
    #ANN_classification(X_pca, Y_pca, cv)

    ## before pca
    # k_range = range(1, 20, 2)
    # knn_classification(X, Y, k_range, cv)
    # dt_classification(X, Y, cv)
    # rf_classification(X_pca, Y_pca, cv)
    # svm_classification(X, Y, cv)
    # ANN_classification(X, Y, cv)