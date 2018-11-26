import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate

target_name = "clean_target03.csv"
data_name = "pca_result.csv"

pwd = os.getcwd()
target = pd.read_csv(os.path.join(pwd, "data", target_name)).values.ravel()
data = pd.read_csv(os.path.join(pwd, "data", data_name)).values

print(np.count_nonzero(target))
print(data.shape)
print(target.shape)


# train_x = data[:int(4*data.shape[0]/5), :]
# train_y = target[:int(4*target.shape[0]/5)]
#
# test_x = data[int(4*data.shape[0]/5):, :]
# test_y_label = target[int(4*target.shape[0]/5):]

from sklearn.model_selection import cross_val_score

# DT
from sklearn import tree
clf_dt = tree.DecisionTreeClassifier()
scores = cross_val_score(clf_dt, data, target, cv=5)
print("Accuracy of DT: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# random froset
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(clf_rf, data, target, cv=5)
print("Accuracy of random forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# SVM
from sklearn import svm
clf_svm = svm.SVC(gamma='scale')
scores = cross_val_score(clf_rf, data, target, cv=5)
print("Accuracy of SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#scoring = ['precision_macro', 'recall_macro']




#
# print(scores['test_recall_macro'])
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#
# files_csv = []
# path = "data/"
# for root, dirs, files in os.walk(path):
#     for name in files:
#         if ".csv" in name:
#             files_csv.append(name)
#
#
#
# pwd = os.getcwd()
# for file in files_csv:
#     trainFile = os.path.join(pwd, "data", file)
#     trainData = pd.read_csv(trainFile)
#
# train_matrix= trainData.values
# train_matrix = np.delete(train_matrix, 0, axis=1)
#
# train_matrix = train_matrix.astype(int)
#
# y = train_matrix[:,10]
# x = np.delete(train_matrix, 10, axis=1)
#
# train_x = x[:int(4*x.shape[0]/5), :]
# train_y = y[:int(4*y.shape[0]/5)]
#
# test_x = x[int(4*x.shape[0]/5):, :]
# test_y_label = y[int(4*y.shape[0]/5):]
# print(test_x.shape)
# print(train_x.shape)
#
# DT
# from sklearn import tree
# clf_dt = tree.DecisionTreeClassifier()
# clf_dt = clf_dt.fit(train_x, train_y)
#
# test_y_dt = clf_dt.predict(test_x)
#
# diff = 0
# for i in range(test_y_dt.shape[0]):
#     if test_y_dt[i] != test_y_label[i]:
#         diff += 1
# err_rate = diff/test_y_dt.shape[0]
# print(err_rate)
#
#
# # random froset
# from sklearn.ensemble import RandomForestClassifier
# clf_rf = RandomForestClassifier(n_estimators=10)
# clf_rf = clf_rf.fit(train_x, train_y)
#
# test_y_rf = clf_rf.predict(test_x)
#
# diff = 0
# for i in range(test_y_rf.shape[0]):
#     if test_y_rf[i] != test_y_label[i]:
#         diff += 1
# err_rate = diff/test_y_rf.shape[0]
# print(err_rate)
#
# # SVM
# from sklearn import svm
# clf_svm = svm.SVC(gamma='scale')
# clf_svm.fit(train_x, train_y)
#
# test_y_svm = clf_svm.predict(test_x)
#
# diff = 0
# for i in range(test_y_svm.shape[0]):
#     if test_y_svm[i] != test_y_label[i]:
#         diff += 1
# err_rate = diff/test_y_svm.shape[0]
# print(err_rate)
