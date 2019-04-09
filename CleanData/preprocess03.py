import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

data = pd.read_csv("clean_result02.csv", header=0)
data_df = pd.DataFrame(data)
print(data_df.loan_status.value_counts())

# Remove high correlated predictors
# Separating out the target
y = data_df['loan_status']
# Separating out the features
x = data_df.drop(['loan_status'], axis=1)
correlation(x, 0.95)
x = x.join(y)
data_df = x
print(data_df.loan_status.value_counts())
print(data_df.shape)

#
# Downsample to 5W

df_majority = data_df[data_df.loan_status == 1.0]
df_minority = data_df[data_df.loan_status == 0.0]
print(df_majority.shape)
print(df_minority.shape)
# DownSample minority class
df_majority_downsampled = resample(df_majority,
                                 replace=False,  # sample with replacement
                                 n_samples=25000,  # to match majority class
                                 random_state=123)  # reproducible results

# DownSample mi nority class
df_minority_downsampled = resample(df_minority,
                                 replace=False,  # sample with replacement
                                 n_samples=25000,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_downsampled = pd.concat([df_minority_downsampled, df_majority_downsampled])

# Display new class counts
print(df_downsampled.loan_status.value_counts())
df_downsampled.to_csv("downsampled.csv", index=False)

data_df = pd.read_csv("downsampled.csv", header=0)
# Separating out the target
y = data_df['loan_status']
print(y.shape)
# Separating out the features
x = data_df.drop(['loan_status'], axis=1)
x = pd.DataFrame(StandardScaler().fit_transform(x))
x = x.join(y)
x.to_csv("clean_data_with_target.csv", index=False)
print("------")
print(x.loan_status.value_counts())


## Plot
data = pd.read_csv("clean_data_with_target.csv", header=0)
data_df = pd.DataFrame(data)
y = data_df['loan_status']
# # Separating out the features
x = data_df.drop(['loan_status'], axis=1)
pca = PCA(n_components=67)
pca.fit(x)
projected = pca.fit_transform(x)
print(x.shape)
print(projected.shape)
marker_size=1
plt.scatter(projected[:, 0], projected[:, 1], marker_size,
            c=y ,cmap=plt.cm.get_cmap('Greys_r', 2) )
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

plt.scatter(projected[:, 0], projected[:, 1], marker_size,
            c=y ,cmap=plt.cm.get_cmap('Greys', 2) )
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

pca = PCA().fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

plt.show()
print(np.cumsum(pca.explained_variance_ratio_))



#Set cutoff
x = data_df.drop(['loan_status'], axis=1)
pca = PCA(n_components=44)
pca.fit(x)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))
X = pd.DataFrame(pca.fit_transform(x))
X = X.join(y)
print(X.loan_status.value_counts())
X.to_csv("clean_data_with_target_after_pca.csv", index=False)
