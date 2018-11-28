import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("clean_result02.csv", header=0)
data_df = pd.DataFrame(data)

# Separating out the target
y = data_df['loan_status']
# Separating out the features
x = data_df.drop(['loan_status'], axis=1)
x = pd.DataFrame(StandardScaler().fit_transform(x))
x = x.join(y)
x.to_csv("clean_data_with_target.csv", index=False)


x = data_df.drop(['loan_status'], axis=1)
pca = PCA(n_components=9)
pca.fit(x)
print(pca.explained_variance_ratio_)
X = pd.DataFrame(pca.fit_transform(x))
X = X.join(y)
X.to_csv("clean_data_with_target_after_pca.csv", index=False)