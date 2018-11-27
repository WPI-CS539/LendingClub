import os
import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# # using PCA data
# target_name = "clean_target03.csv"
# data_name = "pca_result.csv"
#
# pwd = os.getcwd()
# target = pd.read_csv(os.path.join(pwd, "data", target_name)).values.ravel()
# data = pd.read_csv(os.path.join(pwd, "data", data_name)).values
#
# print(np.count_nonzero(target))
# print(data.shape)
# print(target.shape)
#
#
# from sklearn.model_selection import cross_val_score
#
# # DT
# from sklearn import tree
# clf_dt = tree.DecisionTreeClassifier()
# scores = cross_val_score(clf_dt, data, target, cv=5)
# print("Accuracy of DT: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# # random froset
# from sklearn.ensemble import RandomForestClassifier
# clf_rf = RandomForestClassifier(n_estimators=100)
# scores = cross_val_score(clf_rf, data, target, cv=5)
# print("Accuracy of random forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# # SVM
# from sklearn import svm
# clf_svm = svm.SVC(gamma='scale')
# scores = cross_val_score(clf_svm, data, target, cv=5)
# print("Accuracy of SVM: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# using RAW data
file = "down-sample.csv"
pwd = os.getcwd()
trainFile = os.path.join(pwd, "data", file)
trainData = pd.read_csv(trainFile)
train_matrix= trainData.values
#train_matrix = np.delete(train_matrix, 0, axis=1)

y = train_matrix[:,10]
x = np.delete(train_matrix, 10, axis=1)


from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=6)

# Knn
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=25)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

plt.subplot(221)
for train, test in cv.split(x, y):
    probas_ = clf_knn.fit(x[train], y[train]).predict_proba(x[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of KNN')
plt.legend(loc="lower right")

# plt.show()
#scores0 = cross_val_score(clf_knn, x, y, cv=5)
#print("Accuracy of KNN: %0.2f (+/- %0.2f)" % (scores0.mean(), scores0.std() * 2))

# DT
from sklearn import tree
clf_dt = tree.DecisionTreeClassifier()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

plt.subplot(222)
for train, test in cv.split(x, y):
    probas_ = clf_dt.fit(x[train], y[train]).predict_proba(x[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of DT')
plt.legend(loc="lower right")

#plt.show()
# scores1 = cross_val_score(clf_dt, x, y, cv=5)
# print("Accuracy of DT: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

# random froset
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=100)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

plt.subplot(223)
for train, test in cv.split(x, y):
    probas_ = clf_rf.fit(x[train], y[train]).predict_proba(x[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of random forest')
plt.legend(loc="lower right")

#plt.show()
# scores2 = cross_val_score(clf_rf, x, y, cv=5)
# print("Accuracy of random forest: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

# SVM
from sklearn import svm
clf_svm = svm.SVC(gamma='scale',  probability=True)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

plt.subplot(224)
for train, test in cv.split(x, y):
    probas_ = clf_svm.fit(x[train], y[train]).predict_proba(x[test])

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of SVM')
plt.legend(loc="lower right")

plt.show()
# scores3 = cross_val_score(clf_svm, x, y, cv=5)
# print("Accuracy of SVM: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))



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
