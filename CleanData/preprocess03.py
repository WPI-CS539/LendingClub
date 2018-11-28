import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

data = pd.read_csv("clean_result02.csv", header=0)
data_df = pd.DataFrame(data)

# Separating out the target
y = data_df['loan_status']
# Separating out the features
x = data_df.drop(['loan_status'], axis=1)
x = pd.DataFrame(StandardScaler().fit_transform(x))
x = x.join(y)
x.to_csv("clean_data_with_target.csv", index=False)


##Separate majority and minority classes
df_majority = x[x['loan_status'] == 1]
df_minority = x[x['loan_status'] == 0]
print(df_minority.loan_status.value_counts())

# Upsample minority class
df_majority_downsampled = resample(df_majority,
                                 replace=False,  # sample with replacement
                                 n_samples=8520,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_downsampled = pd.concat([df_minority, df_majority_downsampled])

# Display new class counts
print(df_downsampled.loan_status.value_counts())
df_downsampled.to_csv("clean_data_with_target_down_sample.csv", index=False)


x = data_df.drop(['loan_status'], axis=1)
pca = PCA(n_components=9)
pca.fit(x)
print(pca.explained_variance_ratio_)
X = pd.DataFrame(pca.fit_transform(x))
X = X.join(y)
X.to_csv("clean_data_with_target_after_pca.csv", index=False)


##Separate majority and minority classes
df_X_majority = X[X.loan_status == 1]
df_X_minority = X[X.loan_status == 0]
print(df_minority.loan_status.value_counts())

# Upsample minority class
df_majority_downsampled = resample(df_X_majority,
                                 replace=False,  # sample with replacement
                                 n_samples=8520,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_downsampled = pd.concat([df_X_minority, df_majority_downsampled])

# Display new class counts
print(df_downsampled.loan_status.value_counts())
df_downsampled.to_csv("clean_data_with_target_down_sample_after_pca.csv", index=False)