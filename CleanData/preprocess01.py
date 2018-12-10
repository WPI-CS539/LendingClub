import pandas as pd

data = pd.read_csv("clean_result00.csv", header=0)
data_df = pd.DataFrame(data)

## Modify "term": Delete string " month", convert string to integer.
def cleanterm(x):
    x = x.replace(" months", "").replace(" ", "")
    return int(x)
data_df['term'] = data_df['term'].apply(cleanterm)

## Modify verification_status, convert "Verified" and "Source Verified" to 1, and "Not Verified" to 0.
def cleanverificationstatus(x):
    x = x.replace("Not Verified", "0").replace("Source Verified", "1").replace("Verified", "1")
    return int(x)

data_df['verification_status'] = data_df['verification_status'].apply(cleanverificationstatus)

data_df.to_csv("clean_result01.csv", index=False)