import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("clean_result02.csv", header=0)
data_df = pd.DataFrame(data)
print(data_df.loan_status.value_counts())

# Separating out the target
y = data_df['loan_status']
# Separating out the features
x = data_df.drop(['loan_status'], axis=1)
x = pd.DataFrame(StandardScaler().fit_transform(x))
x = x.join(y)
x.to_csv("clean_data_with_target.csv", index=False)
print(x.loan_status.value_counts())


x = data_df.drop(['loan_status'], axis=1)
pca = PCA(n_components=9)
pca.fit(x)
print(pca.explained_variance_ratio_)
X = pd.DataFrame(pca.fit_transform(x))
X = X.join(y)
print(X.loan_status.value_counts())
X.to_csv("clean_data_with_target_after_pca.csv", index=False)

print(X[X.columns[0]].tolist(), X[X.columns[1]])
plt.axis([int(min(X[X.columns[0]].tolist()))-1, int(max(X[X.columns[0]].tolist()))+1
         , int(min(X[X.columns[1]].tolist()))-1, int(max(X[X.columns[1]].tolist()))+1])

df_majority = X[X.loan_status == 1]
df_minority = X[X.loan_status == 0]

plt.plot(df_majority[df_majority.columns[0]].tolist(), df_majority[df_majority.columns[1]], 'r.')
plt.plot(df_minority[df_minority.columns[0]].tolist(), df_minority[df_minority.columns[1]], 'b.')

plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

