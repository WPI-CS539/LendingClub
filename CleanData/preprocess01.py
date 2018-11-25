import pandas as pd
names_no = ["id", "member_id", "emp_title", "pymnt_plan", "desc", "home_ownership", "title", "mths_since_last_delinq",
         "mths_since_last_record", "next_pymnt_d", "mths_since_last_major_derog", "annual_inc_joint", "dti_joint",
         "verification_status_joint", "open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
         "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl",
         "inq_last_12m", "mths_since_recent_bc_dlq", "revol_bal_joint",
         "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
         "sec_app_revol_util", "sec_app_open_il_6m", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
         "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog", "hardship_type",
         "hardship_reason", "hardship_status", "deferral_term", "hardship_amount", "hardship_start_date",
         "hardship_end_date", "payment_plan_start_date", "hardship_length", "hardship_dpd", "hardship_loan_status",
         "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount", "hardship_last_payment_amount",
         "grade", "sub_grade", "zip_code", "addr_state", "purpose", "policy_code", "for_compare", "application_type",
         "mths_since_recent_revol_delinq"]

names_all = ["id", "member_id", "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate", "installment",
             "grade", "sub_grade", "emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status",
             "issue_d", "loan_status", "pymnt_plan", "desc", "purpose", "title", "zip_code", "addr_state", "dti",
             "delinq_2yrs", "earliest_cr_line", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
             "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "initial_list_status", "out_prncp",
             "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
             "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
             "next_pymnt_d", "last_credit_pull_d", "collections_12_mths_ex_med", "mths_since_last_major_derog",
             "policy_code", "application_type", "annual_inc_joint", "dti_joint", "verification_status_joint",
             "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_il_6m", "open_il_12m",
             "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m",
             "max_bal_bc", "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m",
             "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths",
             "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
             "mort_acc", "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
             "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
             "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
             "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
             "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
             "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit", "revol_bal_joint",
             "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
             "sec_app_revol_util", "sec_app_open_il_6m", "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
             "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog", "hardship_flag",
             "hardship_type", "hardship_reason", "hardship_status", "deferral_term", "hardship_amount",
             "hardship_start_date", "hardship_end_date", "payment_plan_start_date", "hardship_length",
             "hardship_dpd", "hardship_loan_status", "orig_projected_additional_accrued_interest",
             "hardship_payoff_balance_amount", "hardship_last_payment_amount"]

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

names = diff(names_all, names_no)

data = pd.read_csv("accepted_2007_to_2017.csv", header=0, usecols=names)
data_df = pd.DataFrame(data)

## Modify "term": Delete string " month", convert string to integer.
def cleanterm(x):
    x = x.replace(" months", "").replace(" ", "")
    return int(x)
data_df['term'] = data_df['term'].apply(cleanterm)

## Modify "emp_length": Delete " years", " year", "+", convert string to integer.
def cleanemplength(x):
    x = x.replace(" years", "").replace(" year", "").replace("+", "").replace("< ", "-").replace("n/a", "-1")
    return int(x)

data_df['emp_length'] = data_df['emp_length'].apply(cleanemplength)
data_df = data_df[data_df['emp_length'] > 0]

## Modify verification_status, convert "Verified" and "Source Verified" to 1, and "Not Verified" to 0.
def cleanverificationstatus(x):
    x = x.replace("Not Verified", "0").replace("Source Verified", "1").replace("Verified", "1")
    return int(x)

data_df['verification_status'] = data_df['verification_status'].apply(cleanverificationstatus)

## Modify loan_status, convert
def cleanloanstatus(x):
    x = x.replace("Current", "1").replace("Does not meet the credit policy. Status:Fully Paid", "1")\
        .replace("Fully Paid", "1").replace("Does not meet the credit policy. Status:Charged Off", "0")\
        .replace("Charged Off", "0").replace("Issued", "0").replace("In Grace Period", "0")\
        .replace("Late (16-30 days)", "0").replace("Late (31-120 days)", "0").replace("Default", "0")
    return int(x)
data_df['loan_status'] = data_df['loan_status'].apply(cleanloanstatus)

data_df.to_csv("clean_result01.csv", index=False)