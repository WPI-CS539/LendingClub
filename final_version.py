import os
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

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
    specificities = list()
    for i in k_range:
        print("start train N = ", i)
        train_error = list()
        test_error = list()
        specificity = list()
        clf_knn = KNeighborsClassifier(n_neighbors=i)
        for train, test in cv.split(X, Y):
            clf_knn.fit(X[train], Y[train])
            train_error.append(clf_knn.score(X[train], Y[train]))
            test_error.append(clf_knn.score(X[test], Y[test]))
            predictions = clf_knn.predict(X[test])
            cm = confusion_matrix(Y[test], predictions)
            specificity.append(cm[1,1]/(cm[1,1] + cm[1,0]))
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))
        specificities.append(sum(specificity)/len(specificity))

    print("test accuracy:", test_errors)
    print("test specificity:", specificities)
    plt.figure()
    plt.plot(k_range, train_errors, label='Train')
    plt.plot(k_range, test_errors, label='Test')
    plt.plot(k_range, specificities, label='specificity(true negative rate)')
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
    specificities = list()
    depths = range(1, 30, 2)
    for depth in depths:
        train_error = list()
        test_error = list()
        specificity = list()
        clf_dt = tree.DecisionTreeClassifier(max_depth=depth)
        for train, test in cv.split(X, Y):
            clf_dt.fit(X[train], Y[train])
            train_error.append(clf_dt.score(X[train], Y[train]))
            test_error.append(clf_dt.score(X[test], Y[test]))
            predictions = clf_dt.predict(X[test])
            cm = confusion_matrix(Y[test], predictions)
            specificity.append(cm[1,1]/(cm[1,1] + cm[1,0]))
        specificities.append(sum(specificity)/len(specificity))
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))

    print("test accuracy:", test_errors)
    print("test specificity:", specificities)
    plt.figure()
    plt.plot(depths, train_errors, label='Train')
    plt.plot(depths, test_errors, label='Test')
    plt.plot(depths, specificities, label='specificity')
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
    specificities = list()
    tree_nums = range(100, 2000, 500)
    for tree_num in tree_nums:
        train_error = list()
        test_error = list()
        specificity = list()
        clf_rf = RandomForestClassifier(n_estimators=tree_num)
        for train, test in cv.split(X, Y):
            clf_rf.fit(X[train], Y[train])
            train_error.append(clf_rf.score(X[train], Y[train]))
            test_error.append(clf_rf.score(X[test], Y[test]))
            predictions = clf_rf.predict(X[test])
            cm = confusion_matrix(Y[test], predictions)
            specificity.append(cm[1,1]/(cm[1,1] + cm[1,0]))
        specificities.append(sum(specificity)/len(specificity))
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))

    print("test accuracy:", test_errors)
    print("test specificity:", specificities)
    plt.figure()
    plt.plot(tree_nums, specificities, label='specificity')
    plt.plot(tree_nums, train_errors, label='Train')
    plt.plot(tree_nums, test_errors, label='Test')
    plt.legend(loc='lower left')
    plt.ylim([0, 1.1])
    plt.xlim([90, 2100])
    plt.title('Evaluation of random forest model')
    plt.xlabel('The number of trees')
    plt.ylabel('Performance')
    plt.show()


def svm_classification(X, Y, cv):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    x = [1, 2, 3, 4]
    train_errors = list()
    test_errors = list()
    specificities = list()
    for kernel_unit in kernels:
        train_error = list()
        test_error = list()
        specificity = list()
        clf_svm = svm.SVC(kernel=kernel_unit, probability=True, gamma='scale')
        for train, test in cv.split(X, Y):
            clf_svm.fit(X[train], Y[train])
            train_error.append(clf_svm.score(X[train], Y[train]))
            test_error.append(clf_svm.score(X[test], Y[test]))
            predictions = clf_svm.predict(X[test])
            cm = confusion_matrix(Y[test], predictions)
            specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
        specificities.append(sum(specificity) / len(specificity))
        print("the kernel is ", kernel_unit)
        print("the accuracy of train is ", train_error)
        print("the accuracy of test is ", test_error)
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))

    print("test accuracy:", test_errors)
    print("test specificity:", specificities)
    plt.figure()
    plt.scatter(x, specificities, label='specificity')
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




def NN_classification(X, Y, cv):
    clf_NN = MLPClassifier(hidden_layer_sizes=(9, 9, 9,),
                  activation='tanh', batch_size='auto',
                  learning_rate ='constant', learning_rate_init = 0.001,
                  max_iter = 200,
                  shuffle = True,
                  verbose = True,
                  warm_start = False,
                  early_stopping = True, n_iter_no_change = 30, validation_fraction = 0.1,
                  beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
    print(X.shape)
    print(Y.shape)
    clf_NN.fit(X, Y)
    predictions = clf_NN.predict(X_test)
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
    print(clf_NN.score(X_test, Y_test))


def Ada_classification(X, Y, cv):
    clf_dt = tree.DecisionTreeClassifier()
    n_ests = range(100, 500, 50)
    train_errors = list()
    test_errors = list()
    specificities = list()
    for n_est in n_ests:
        bdt = AdaBoostClassifier(clf_dt, n_estimators=n_est)
        train_error = list()
        test_error = list()
        specificity = list()
        for train, test in cv.split(X, Y):
            bdt.fit(X[train], Y[train])
            train_error.append(bdt.score(X[train], Y[train]))
            test_error.append(bdt.score(X[test], Y[test]))
            predictions = bdt.predict(X[test])
            cm = confusion_matrix(Y[test], predictions)
            specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
        specificities.append(sum(specificity) / len(specificity))
        train_errors.append(sum(train_error)/len(train_error))
        test_errors.append(sum(test_error)/len(test_error))
    print("test accuracy:", test_errors)
    print("test specificity:", specificities)

    print("test accuracy:", test_errors)
    print("test specificity:", specificities)
    plt.figure()
    plt.plot(n_ests, specificities, label='specificity')
    plt.plot(n_ests, train_errors, label='Train')
    plt.plot(n_ests, test_errors, label='Test')

    plt.legend(loc='lower left')
    plt.ylim([0.5, 1.1])
    plt.xlim([90, 510])
    plt.title('Evaluation of Adaboost with DT model')
    plt.xlabel('The estimators')
    plt.ylabel('Performance')
    plt.show()


if __name__ == "__main__":

    X, Y, X_pca, Y_pca = load_data()

    cv = StratifiedKFold(n_splits=2)
    # k_range = range(1, 21, 2)
    # knn_classification(X, Y, k_range, cv)

    # dt_classification(X_pca, Y_pca, cv)

    # rf_classification(X, Y, cv)

    # svm_classification(X_pca, Y_pca, cv)
    # NN_classification(X_pca, Y_pca, cv)
    Ada_classification(X, Y, cv)