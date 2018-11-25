import pandas as pd

data = pd.read_csv("clean_result01.csv", header=0)
data_df = pd.DataFrame(data)

###### Modify issue_d to integer ######
def cleanissued(x):
    x = x.replace("Dec-2013", "43").replace("Nov-2013", "44").replace("Oct-2013", "45")\
        .replace("Sep-2013", "46").replace("Aug-2013", "47")
    return int(x)
data_df['issue_d'] = data_df['issue_d'].apply(cleanissued)

###### Modify earliest_cr_line to integer ######
def cleanyearmonth(x):
    month = x[:3]
    monthN = 1
    if month is "Dec":
        monthN = 1
    if month is "Nov":
        monthN = 2
    if month is "Oct":
        monthN = 3
    if month is "Sep":
        monthN = 4
    if month is "Aug":
        monthN = 5
    if month is "Jul":
        monthN = 6
    if month is "Jun":
        monthN = 7
    if month is "May":
        monthN = 8
    if month is "Apr":
        monthN = 9
    if month is "Mar":
        monthN = 10
    if month is "Feb":
        monthN = 11
    if month is "Jan":
        monthN = 12
    year = int(x[-4:])
    x = (2017 - year - 1) * 12 + 6 + monthN
    if x < 0:
        x = 0
    return int(x)
data_df['earliest_cr_line'] = data_df['earliest_cr_line'].apply(cleanyearmonth)

###### Modify last_payment_date to integer ######
data_df = data_df[data_df['last_pymnt_d'].notnull()]
data_df['last_pymnt_d'] = data_df['last_pymnt_d'].apply(cleanyearmonth)

###### Modify last_credit_pull_d to integer ######
data_df = data_df[data_df['last_credit_pull_d'].notnull()]
data_df['last_credit_pull_d'] = data_df['last_credit_pull_d'].apply(cleanyearmonth)

## Convert initial_list_status to 0 or 1
def cleaninitialliststatus(x):
    x = x.replace("w", "1").replace("f", "0")
    return int(x)
data_df['initial_list_status'] = data_df['initial_list_status'].apply(cleaninitialliststatus)

## Convert application_type to 2 column- individual_pay/direct_pay

## Convert hardship_flag to 0 or 1
def cleanhardshipflag(x):
    x = x.replace("Y", "1").replace("N", "0")
    return int(x)
data_df['hardship_flag'] = data_df['hardship_flag'].apply(cleanhardshipflag)

## Convert emp_length NA's to 0
## Delete 21 empty rows
## Convert all NA's with columns' mean value separately

data_df = data_df.apply(lambda x: x.fillna(x.mean()), axis=0)
data_df.to_csv("clean_result02.csv", index=False)