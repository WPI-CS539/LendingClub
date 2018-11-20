import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("clean_result02.csv", header=0)
data_df = pd.DataFrame(data)

pca = PCA(n_components=45)
X = pca.fit_transform(data_df.values)
X = pca.transform(data_df)
X_Df = pd.DataFrame(data=X)
X_Df.to_csv("clean_result03.csv")