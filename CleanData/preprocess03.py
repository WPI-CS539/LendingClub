import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("clean_result02.csv", header=0)
data_df = pd.DataFrame(data)

# Separating out the features
x = data_df.loc[:, data_df.columns != 'loan_status'].values
# Separating out the target
y = data_df.loc[:, ['loan_status']].values
pca = PCA(n_components=9)
pca.fit(x)
print(pca.explained_variance_ratio_)
X = pca.fit_transform(x)
# X = pca.fit_transform(data_df.values)

# X = pca.transform(data_df)
x_df = pd.DataFrame(data=x)
y_df = pd.DataFrame(data=y)
X_Df = pd.DataFrame(data=X)

x_df.to_csv("clean_result03.csv", index=False)
y_df.to_csv("clean_target03.csv", index=False)
X_Df.to_csv("pca_result.csv", index=False)