import os
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
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
    #plt.ylim([0, 1.1])
    #plt.xlim([0, 21])
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
    #plt.ylim([0, 1.1])
    #plt.xlim([depths[0]-2, depths[-1]+2])
    plt.title('Evaluation of DT model')
    plt.xlabel('The depth')
    plt.ylabel('Performance')
    plt.show()


def rf_classification(X, Y, cv):
    train_errors = list()
    test_errors = list()
    specificities = list()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num=11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=2, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, Y)
    print(rf_random.cv_results_)
    print(rf_random.best_estimator_)
    print(rf_random.best_score_)
    print(rf_random.cv_results_["mean_test_score"])
    print(rf_random.cv_results_["mean_train_score"])

    test_score = rf_random.cv_results_["mean_test_score"]
    train_score = rf_random.cv_results_["mean_train_score"]
    clf_rf = rf_random.best_estimator_
    train_error = list()
    test_error = list()
    specificity = list()
    cv = StratifiedKFold(n_splits=10)
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
    plt.plot(range(len(test_score)), test_score, label='Test Score')
    plt.plot(range(len(train_score)), train_score, label='Train Score')
    plt.legend(loc='lower left')
    plt.title('Evaluation of random forest model')
    plt.xlabel('Different Parameters')
    plt.ylabel('Performance')
    plt.show()

    # tree_nums = range(100, 2200, 1000)
    #
    # for tree_num in tree_nums:
    #     train_error = list()
    #     test_error = list()
    #     specificity = list()
    #     clf_rf = RandomForestClassifier(n_estimators=tree_num, max_features=int(math.sqrt(X.shape[1])), max_depth=10)
    #     for train, test in cv.split(X, Y):
    #         clf_rf.fit(X[train], Y[train])
    #         train_error.append(clf_rf.score(X[train], Y[train]))
    #         test_error.append(clf_rf.score(X[test], Y[test]))
    #         predictions = clf_rf.predict(X[test])
    #         cm = confusion_matrix(Y[test], predictions)
    #         specificity.append(cm[1,1]/(cm[1,1] + cm[1,0]))
    #     specificities.append(sum(specificity)/len(specificity))
    #     train_errors.append(sum(train_error)/len(train_error))
    #     test_errors.append(sum(test_error)/len(test_error))
    #
    # print("test accuracy:", test_errors)
    # print("test specificity:", specificities)
    # plt.figure()
    # plt.plot(tree_nums, specificities, label='specificity')
    # plt.plot(tree_nums, train_errors, label='Train')
    # plt.plot(tree_nums, test_errors, label='Test')
    # plt.legend(loc='lower left')
    # #plt.ylim([0, 1.1])
    # #plt.xlim([0, 102])
    # plt.title('Evaluation of random forest model')
    # plt.xlabel('The number of trees')
    # plt.ylabel('Performance')
    # plt.show()


def svm_classification(X, Y, cv):
    ## rbf
    train_errors = list()
    test_errors = list()
    specificities = list()
    # Create the random grid
    random_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6), 'kernel': ['rbf'], 'probability':[True]}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    svm_ = svm.SVC()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    svm_random = RandomizedSearchCV(estimator=svm_, param_distributions=random_grid, n_iter=5, cv=2, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    svm_random.fit(X, Y)
    print(svm_random.cv_results_)
    print(svm_random.best_estimator_)
    print(svm_random.best_score_)
    print(svm_random.cv_results_["mean_test_score"])
    print(svm_random.cv_results_["mean_train_score"])

    test_score = svm_random.cv_results_["mean_test_score"]
    train_score = svm_random.cv_results_["mean_train_score"]
    clf_svm = svm_random.best_estimator_
    train_error = list()
    test_error = list()
    specificity = list()
    # cv = StratifiedKFold(n_splits=10)
    for train, test in cv.split(X, Y):
        clf_svm.fit(X[train], Y[train])
        train_error.append(clf_svm.score(X[train], Y[train]))
        test_error.append(clf_svm.score(X[test], Y[test]))
        predictions = clf_svm.predict(X[test])
        cm = confusion_matrix(Y[test], predictions)
        specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    specificities.append(sum(specificity) / len(specificity))
    train_errors.append(sum(train_error) / len(train_error))
    test_errors.append(sum(test_error) / len(test_error))
    print("test accuracy:", test_errors)
    print("test specificity:", specificities)

    plt.figure()
    plt.plot(range(len(test_score)), test_score, label='Test Score')
    plt.plot(range(len(train_score)), train_score, label='Train Score')
    plt.legend(loc='lower left')
    plt.title('Evaluation of SVM model with "rbf"')
    plt.xlabel('Different Parameters')
    plt.ylabel('Performance')
    plt.show()

    ## poly

    # Create the random grid
    random_grid = {'C': np.logspace(-3, 2, 6), 'degree': [int(x) for x in np.linspace(start=2, stop=5, num=4)],
                   'gamma': np.logspace(-3, 2, 6), 'kernel': ['poly'], 'probability': [True]}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    svm_ = svm.SVC()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    svm_random = RandomizedSearchCV(estimator=svm_, param_distributions=random_grid, n_iter=5, cv=2, verbose=2,
                                    random_state=42, n_jobs=-1)
    # Fit the random search model
    svm_random.fit(X, Y)
    print(svm_random.cv_results_)
    print(svm_random.best_estimator_)
    print(svm_random.best_score_)
    print(svm_random.cv_results_["mean_test_score"])
    print(svm_random.cv_results_["mean_train_score"])

    test_score = svm_random.cv_results_["mean_test_score"]
    train_score = svm_random.cv_results_["mean_train_score"]
    clf_svm = svm_random.best_estimator_
    train_error = list()
    test_error = list()
    specificity = list()
    # cv = StratifiedKFold(n_splits=10)
    for train, test in cv.split(X, Y):
        clf_svm.fit(X[train], Y[train])
        train_error.append(clf_svm.score(X[train], Y[train]))
        test_error.append(clf_svm.score(X[test], Y[test]))
        predictions = clf_svm.predict(X[test])
        cm = confusion_matrix(Y[test], predictions)
        specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    specificities.append(sum(specificity) / len(specificity))
    train_errors.append(sum(train_error) / len(train_error))
    test_errors.append(sum(test_error) / len(test_error))
    print("test accuracy:", test_errors)
    print("test specificity:", specificities)

    plt.figure()
    plt.plot(range(len(test_score)), test_score, label='Test Score')
    plt.plot(range(len(train_score)), train_score, label='Train Score')
    plt.legend(loc='lower left')
    plt.title('Evaluation of SVM model with "poly"')
    plt.xlabel('Different Parameters')
    plt.ylabel('Performance')
    plt.show()

    ## sigmoid

    # Create the random grid
    random_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6), 'kernel': ['sigmoid'], 'probability': [True]}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    svm_ = svm.SVC()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    svm_random = RandomizedSearchCV(estimator=svm_, param_distributions=random_grid, n_iter=5, cv=2, verbose=2,
                                    random_state=42, n_jobs=-1)
    # Fit the random search model
    svm_random.fit(X, Y)
    print(svm_random.cv_results_)
    print(svm_random.best_estimator_)
    print(svm_random.best_score_)
    print(svm_random.cv_results_["mean_test_score"])
    print(svm_random.cv_results_["mean_train_score"])

    test_score = svm_random.cv_results_["mean_test_score"]
    train_score = svm_random.cv_results_["mean_train_score"]
    clf_svm = svm_random.best_estimator_
    train_error = list()
    test_error = list()
    specificity = list()
    # cv = StratifiedKFold(n_splits=10)
    for train, test in cv.split(X, Y):
        clf_svm.fit(X[train], Y[train])
        train_error.append(clf_svm.score(X[train], Y[train]))
        test_error.append(clf_svm.score(X[test], Y[test]))
        predictions = clf_svm.predict(X[test])
        cm = confusion_matrix(Y[test], predictions)
        specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    specificities.append(sum(specificity) / len(specificity))
    train_errors.append(sum(train_error) / len(train_error))
    test_errors.append(sum(test_error) / len(test_error))
    print("test accuracy:", test_errors)
    print("test specificity:", specificities)

    plt.figure()
    plt.plot(range(len(test_score)), test_score, label='Test Score')
    plt.plot(range(len(train_score)), train_score, label='Train Score')
    plt.legend(loc='lower left')
    plt.title('Evaluation of SVM model with "sigmoid"')
    plt.xlabel('Different Parameters')
    plt.ylabel('Performance')
    plt.show()

    plt.figure()
    kernels = ['rbf', 'poly', 'sigmoid']
    x = [1,2,3]
    plt.scatter(x, specificities, label='specificity')
    plt.scatter(x, train_errors, label='Train')
    plt.scatter(x, test_errors, label='Test')
    plt.xticks(x, kernels, rotation=45)
    plt.legend(loc='lower left')
    #plt.ylim([0, 1.1])
    #plt.xlim([0, 5])
    plt.title('Evaluation of SVM model')
    plt.xlabel('The Kernel')
    plt.ylabel('Performance')
    plt.show()

    # kernels = ['poly', 'rbf', 'sigmoid']
    # x = [1, 2, 3]
    # train_errors = list()
    # test_errors = list()
    # specificities = list()
    # for kernel_unit in kernels:
    #     train_error = list()
    #     test_error = list()
    #     specificity = list()
    #     clf_svm = svm.SVC(kernel=kernel_unit, probability=True, gamma='scale')
    #     for train, test in cv.split(X, Y):
    #         print("Here")
    #         clf_svm.fit(X[train], Y[train])
    #         train_error.append(clf_svm.score(X[train], Y[train]))
    #         test_error.append(clf_svm.score(X[test], Y[test]))
    #         predictions = clf_svm.predict(X[test])
    #         cm = confusion_matrix(Y[test], predictions)
    #         specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    #     specificities.append(sum(specificity) / len(specificity))
    #     print("the kernel is ", kernel_unit)
    #     print("the accuracy of train is ", train_error)
    #     print("the accuracy of test is ", test_error)
    #     train_errors.append(sum(train_error)/len(train_error))
    #     test_errors.append(sum(test_error)/len(test_error))
    #
    # print("test accuracy:", test_errors)
    # print("test specificity:", specificities)
    # plt.figure()
    # plt.scatter(x, specificities, label='specificity')
    # plt.scatter(x, train_errors, label='Train')
    # plt.scatter(x, test_errors, label='Test')
    # plt.xticks(x, kernels, rotation=45)
    # plt.legend(loc='lower left')
    # #plt.ylim([0, 1.1])
    # #plt.xlim([0, 5])
    # plt.title('Evaluation of SVM model')
    # plt.xlabel('The Kernel')
    # plt.ylabel('Performance')
    # plt.show()



def ANN_classification(X, Y, cv):
    train_accuracies = list()
    test_accuracies = list()
    specificities = list()
    x_axis= []
    for i in range(1, 51):
        x_axis.append(0.0001 * (i * 2))
        print("start train N = ", i)
        train_accuracy = list()
        test_accuracy = list()
        specificity = list()
        for train, test in cv.split(X, Y):
            X_train = np.asmatrix(X[train])
            X_test = np.asmatrix(X[test])
            Y_train = Y[train]
            Y_test = Y[test]
            W1, b1, W2, b2 = ann_train(X_train, Y_train, alpha=0.0001*(i * 2), n_epoch=1)
            Y_predict_train, P = ann_predict(X_train, W1, b1, W2, b2)
            Y_predict, P = ann_predict(X_test, W1, b1, W2, b2)
            cm_train = confusion_matrix(Y_train, Y_predict_train)
            cm = confusion_matrix(Y_test, Y_predict)
            train_accuracy.append((cm_train[1, 1] + cm_train[0, 0]) / (cm_train[0, 0] + cm_train[0, 1] + cm_train[1, 1] + cm_train[1, 0]))
            test_accuracy.append((cm[1, 1] + cm[0, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 1] + cm[1, 0]))
            specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
        train_accuracies.append(sum(train_accuracy) / len(train_accuracy))
        test_accuracies.append(sum(test_accuracy) / len(test_accuracy))
        specificities.append(sum(specificity) / len(specificity))
    plt.figure()
    plt.plot(x_axis, train_accuracies, label='Train')
    plt.plot(x_axis, test_accuracies, label='Test')
    plt.plot(x_axis, specificities, label='specificity')
    plt.legend(loc='lower left')
    plt.title('Evaluation of ANN model')
    plt.xlabel('Alpha')
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

    X_test = X[20000:25000,:]
    Y_test = Y[20000:25000]
    X = X[0:20000,:]
    Y = Y[0:20000]
    print(X.shape)
    print(Y.shape)
    clf_NN.fit(X, Y)
    predictions = clf_NN.predict(X_test)
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))
    print(clf_NN.score(X_test, Y_test))


def Ada_classification(X, Y, cv):
    train_errors = list()
    test_errors = list()
    specificities = list()
    # Number of trees in random forest
    learning_rate = [x for x in np.linspace(start=0.1, stop=2, num=20)]

    n_estimators = [int(x) for x in np.linspace(start=50, stop=2000, num=100)]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    ab = AdaBoostClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    ab_random = RandomizedSearchCV(estimator=ab, param_distributions=random_grid, n_iter=5, cv=2, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    ab_random.fit(X, Y)
    print(ab_random.cv_results_)
    print(ab_random.best_estimator_)
    print(ab_random.best_score_)
    print(ab_random.cv_results_["mean_test_score"])
    print(ab_random.cv_results_["mean_train_score"])

    test_score = ab_random.cv_results_["mean_test_score"]
    train_score = ab_random.cv_results_["mean_train_score"]
    clf_ab = ab_random.best_estimator_
    train_error = list()
    test_error = list()
    specificity = list()
    cv = StratifiedKFold(n_splits=10)
    for train, test in cv.split(X, Y):
        clf_ab.fit(X[train], Y[train])
        train_error.append(clf_ab.score(X[train], Y[train]))
        test_error.append(clf_ab.score(X[test], Y[test]))
        predictions = clf_ab.predict(X[test])
        cm = confusion_matrix(Y[test], predictions)
        specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    specificities.append(sum(specificity) / len(specificity))
    train_errors.append(sum(train_error) / len(train_error))
    test_errors.append(sum(test_error) / len(test_error))
    print("test accuracy:", test_errors)
    print("test specificity:", specificities)

    plt.figure()
    plt.plot(range(len(test_score)), test_score, label='Test Score')
    plt.plot(range(len(train_score)), train_score, label='Train Score')
    plt.legend(loc='lower left')
    plt.title('Evaluation of adaBoost model')
    plt.xlabel('Different Parameters')
    plt.ylabel('Performance')
    plt.show()




    # clf_dt = tree.DecisionTreeClassifier()
    # n_ests = range(100, 500, 50)
    # train_errors = list()
    # test_errors = list()
    # specificities = list()
    # for n_est in n_ests:
    #     bdt = AdaBoostClassifier(clf_dt, n_estimators=n_est)
    #     train_error = list()
    #     test_error = list()
    #     specificity = list()
    #     for train, test in cv.split(X, Y):
    #         bdt.fit(X[train], Y[train])
    #         train_error.append(bdt.score(X[train], Y[train]))
    #         test_error.append(bdt.score(X[test], Y[test]))
    #         predictions = bdt.predict(X[test])
    #         cm = confusion_matrix(Y[test], predictions)
    #         specificity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    #     specificities.append(sum(specificity) / len(specificity))
    #     train_errors.append(sum(train_error)/len(train_error))
    #     test_errors.append(sum(test_error)/len(test_error))
    # print("test accuracy:", test_errors)
    # print("test specificity:", specificities)
    #
    # print("test accuracy:", test_errors)
    # print("test specificity:", specificities)
    # plt.figure()
    # plt.plot(n_ests, specificities, label='specificity')
    # plt.plot(n_ests, train_errors, label='Train')
    # plt.plot(n_ests, test_errors, label='Test')
    #
    # plt.legend(loc='lower left')
    # #plt.ylim([0.5, 1.1])
    # #plt.xlim([90, 510])
    # plt.title('Evaluation of Adaboost with DT model')
    # plt.xlabel('The estimators')
    # plt.ylabel('Performance')
    # plt.show()


if __name__ == "__main__":

    X, Y, X_pca, Y_pca = load_data()
    cv = StratifiedKFold(n_splits=2)

    ## after pca
    # k_range = range(1, 35, 2)
    # knn_classification(X_pca, Y_pca, k_range, cv)
    # dt_classification(X_pca, Y_pca, cv)
    # rf_classification(X_pca, Y_pca, cv)
    #svm_classification(X_pca, Y_pca, cv)
    ANN_classification(X_pca, Y_pca, cv)
    #Ada_classification(X_pca, Y_pca, cv)

    ## before pca
    # k_range = range(1, 35, 2)
    # knn_classification(X, Y, k_range, cv)
    # dt_classification(X, Y, cv)
    #rf_classification(X, Y, cv)
    # svm_classification(X, Y, cv)
    # ANN_classification(X, Y, cv)
    #Ada_classification(X, Y, cv)