import pandas as pd
from sklearn.utils import resample

data = pd.read_csv("clean_result02.csv", header=0)
df = pd.DataFrame(data)

##Separate majority and minority classes
df_majority = df[df.loan_status == 1]
df_minority = df[df.loan_status == 0]
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
df_downsampled.to_csv("down-sample.csv", index=False)