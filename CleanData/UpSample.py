import pandas as pd
from sklearn.utils import resample

data = pd.read_csv("clean_result02.csv", header=0)
df = pd.DataFrame(data)

##Separate majority and minority classes
df_majority = df[df.loan_status == 1]
df_minority = df[df.loan_status == 0]
print(df_majority.loan_status.value_counts())

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=49289,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
print(df_upsampled.loan_status.value_counts())
df_upsampled.to_csv("up-sample.csv", index=False)