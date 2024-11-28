import os
import time
import itertools
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import threading
import warnings
warnings.filterwarnings('ignore')

# Set up file paths
current_dir = os.getcwd() 
file_path = os.path.join(current_dir, "data/Credit_score_data.csv")
df = pd.read_csv(file_path) 

# Extracting and transforming credit history age
df.groupby('Customer_ID')['Month'].nunique().value_counts()
temp_df = df['Credit_History_Age'].str.extract(r'(?P<Years>\d+) Years and (?P<Months>\d+) Months').astype(float)
temp_df['Customer_ID'] = df['Customer_ID'] 
temp_df['Total_months'] = temp_df['Years'] * 12 + temp_df['Months']
temp_df.groupby('Customer_ID')['Total_months'].transform(pd.Series.diff).unique()
df.groupby('Customer_ID')['Credit_Score'].nunique().value_counts()

# Helper functions for data analysis and transformation
def describe_numerical_column(series, col_name):
    description = series.describe()
    num_null_values = series.isnull().sum()

    q1, q3 = series.quantile([0.25, 0.75])
    IQR = q3 - q1
    
    return {
        'Min. value': series.min(),
        'Outlier lower range': q1 - 1.5 * IQR,
        'Outlier upper range': q3 + 1.5 * IQR,
        'Max. value': series.max()
    }

def summarize_numerical_column_with_deviation(data, num_col, group_col='Customer_ID', absolute_summary=True, median_standardization_summary=False):
    Summary_dict = {}
    
    if absolute_summary:
        Summary_dict[num_col] = describe_numerical_column(data[num_col], num_col)
    
    if median_standardization_summary:
        default_MAD = return_max_MAD(data, num_col, group_col)
        num_col_standardization = data.groupby(group_col)[num_col].apply(
            median_standardization, default_value=default_MAD
        )
        Summary_dict[f'Median standardization of {num_col}'] = describe_numerical_column(
            num_col_standardization, f'Median standardization of {num_col}'
        )
        Summary_dict['Max. MAD'] = default_MAD
    
    return Summary_dict

def return_max_MAD(data, num_col, group_col = 'Customer_ID'):
    return (data.groupby(group_col)[num_col].agg(lambda x: (x - x.median()).abs().median())).max()
    
def validate_age(x):
    diff = x.diff()
    if (diff == 0).sum() == 7:
        return True
    elif ((diff.isin([0, 1])).sum() == 7) and ((diff == 1).sum() == 1):
        return True
    else:
        return False
        
def median_standardization(x, default_value):
    med = x.median() 
    abs = (x - med).abs()
    MAD = abs.median()
    if MAD == 0:
        if ((abs == 0).sum() == abs.notnull().sum()):
            return x * 0
        else:
            return (x - med)/default_value
    else:
        return (x - med)/MAD

def return_num_of_modes(x):
    return len(x.mode())

def return_mode(x):
    modes = x.mode()
    if len(modes) == 0:
        return np.nan
    return modes.min()

def forward_backward_fill(x):
    return x.fillna(method='ffill').fillna(method='bfill')

def return_mode_median_filled_int(x):
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(int(modes.median()))

def return_mode_average_filled(x):
    modes = x.mode()
    if len(modes) == 1:
        return x.fillna(modes[0])
    else:
        return x.fillna(modes.mean())

def fill_month_history(x):
    first_non_null_idx = x.argmin()
    first_non_null_value = x.iloc[first_non_null_idx]
    return pd.Series(first_non_null_value + np.array(range(-first_non_null_idx, 8-first_non_null_idx)), index = x.index)

def loading_animation():
    # Loading animation to indicate progress
    for frame in itertools.cycle(['|', '/', '-', '\\\\']):
        if not threading.current_thread().do_run:
            break
        sys.stdout.write(f'\rData cleaning... {frame}')
        sys.stdout.flush()
        time.sleep(0.1)
        
def clean_data(df):
    print("Process (Clean) data:")
    print("*********************")
    
    try:
        import threading
        loading_thread = threading.Thread(target=loading_animation, daemon=True)
        loading_thread.do_run = True
        loading_thread.start()
        
        start_time = time.time()

        # Data cleaning and feature transformation
        df['Customer_ID'].unique()
        df['Customer_ID'].nunique()
        df['Customer_ID'].str.contains('CUS_0x').value_counts()
        df.drop(columns = ['Name'], inplace = True)

        # Cleaning Age column
        df['Age'][~df['Age'].str.isnumeric()].unique()
        df['Age'] = df['Age'].str.replace('_', '')
        df['Age'][~df['Age'].str.isnumeric()].unique()
        df['Age'] = df['Age'].astype(int)

        # Dropping unnecessary columns
        df.drop(columns = ['SSN'], inplace = True)
        df['Occupation'].unique()
        df['Occupation'][df['Occupation'] == '_______'] = np.nan
        df['Occupation'].unique()

        # Cleaning Annual Income column
        df['Annual_Income'][~df['Annual_Income'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
        df['Annual_Income'] = df['Annual_Income'].str.replace('_', '')
        df['Annual_Income'][~df['Annual_Income'].str.fullmatch('([0-9]*[.])?[0-9]+')]
        df['Annual_Income'] = df['Annual_Income'].astype(float)

        # Cleaning Number of Loans column
        df['Num_of_Loan'][~df['Num_of_Loan'].str.isnumeric()].unique()
        df['Num_of_Loan'] = df['Num_of_Loan'].str.replace('_', '').astype(int)

        # Cleaning Number of Delayed Payments column
        temp_series = df['Num_of_Delayed_Payment'][df['Num_of_Delayed_Payment'].notnull()]
        temp_series[~temp_series.str.isnumeric()].unique()
        df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].str.replace('_', '').astype(float)
        summary_num_delayed_payments = summarize_numerical_column_with_deviation(df, 'Num_of_Delayed_Payment', median_standardization_summary = True)
        df['Num_of_Delayed_Payment'][(df['Num_of_Delayed_Payment'] > summary_num_delayed_payments['Num_of_Delayed_Payment']['Outlier upper range']) | (df['Num_of_Delayed_Payment'] < 0)] = np.nan
        df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(pd.Series.diff).value_counts(normalize = True)
        temp = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(lambda x: (x == x.mode()[0]).sum()/x.notnull().sum()).value_counts(normalize = True)
        temp[temp.index > 0.5].sum() # Idenitfying how many times the mode occurs in more than 50% of non-null data per customer
        df.groupby('Customer_ID')['Num_of_Delayed_Payment'].agg(lambda x: len(x.mode())).value_counts()
        df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(return_mode_median_filled_int).astype(int)
        
        # Cleaning Changed Credit Limit column
        df['Changed_Credit_Limit'][~df['Changed_Credit_Limit'].str.fullmatch('[+-]?([0-9]*[.])?[0-9]+')].unique()
        df['Changed_Credit_Limit'][df['Changed_Credit_Limit'] == '_'] = np.nan 
        df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].astype(float)

        # Cleaning Credit Mix column
        df['Credit_Mix'].unique()
        df['Credit_Mix'][df['Credit_Mix'] == '_'] = np.nan

        # Cleaning Outstanding Debt column
        df['Outstanding_Debt'][~df['Outstanding_Debt'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
        df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace('_', '')
        df['Outstanding_Debt'][~df['Outstanding_Debt'].str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
        df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(float)

        # Cleaning Amount Invested Monthly column
        temp_series = df['Amount_invested_monthly'][df['Amount_invested_monthly'].notnull()]
        temp_series[~temp_series.str.fullmatch('([0-9]*[.])?[0-9]+')].unique()
        df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace('_', '').astype(float)
        df['Amount_invested_monthly'][df['Amount_invested_monthly'] > 8000] = np.nan
        summary_amount_invested_monthly = summarize_numerical_column_with_deviation(df, 'Amount_invested_monthly', median_standardization_summary = True)
        df.groupby('Customer_ID')['Amount_invested_monthly'].transform(return_num_of_modes).value_counts()
        df['Amount_invested_monthly'] = df.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.fillna(x.median()))
        

        # Cleaning Payment Behaviour column
        df['Payment_Behaviour'].unique()
        df['Payment_Behaviour'][df['Payment_Behaviour'] == '!@9#%8'] = np.nan
        df['Payment_Behaviour'] = df.groupby('Customer_ID')['Payment_Behaviour'].transform(lambda x: return_mode(x) if len(x.mode()) == 1 else forward_backward_fill(x))
        
        #Monthly Balance column
        temp_series = df['Monthly_Balance'][df['Monthly_Balance'].notnull()]
        temp_series[temp_series.str.fullmatch('[+-]*([0-9]*[.])?[0-9]+') == False].unique()
        df['Monthly_Balance'][df['Monthly_Balance'] == '__-333333333333333333333333333__'] = np.nan
        df['Monthly_Balance'] = df['Monthly_Balance'].astype(float)
        summary_monthly_balance = summarize_numerical_column_with_deviation(df, 'Monthly_Balance', median_standardization_summary = True)
        df.groupby('Customer_ID')['Monthly_Balance'].nunique().value_counts()
        df['Monthly_Balance'] = df.groupby('Customer_ID')['Monthly_Balance'].transform(lambda x: x.fillna(x.median()))

        # Encoding Month column
        df['Month'] = df['Month'].map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8})

        # Sorting and copying dataframe
        df.sort_values(by = ['Customer_ID', 'Month'], ignore_index = True, inplace = True)
        df.drop(columns = 'ID', inplace = True)
        df_copy = df.copy()

        # Handling outliers and missing values in Age column
        df['Age'][(df['Age'] > 100) | (df['Age'] <= 0)] = np.nan 
        summary_age = summarize_numerical_column_with_deviation(df, 'Age', median_standardization_summary = True)
        df['Age'][df.groupby('Customer_ID')['Age'].transform(median_standardization, default_value = return_max_MAD(df, 'Age')) > 80] = np.nan
        df['Age'] =  df.groupby('Customer_ID')['Age'].transform(forward_backward_fill).astype(int)

        # Handling Occupation column
        df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(forward_backward_fill)
        df['Occupation'].isnull().sum()

        # Handling Annual Income and Monthly Inhand Salary columns
        summary_annual_income = summarize_numerical_column_with_deviation(df, 'Annual_Income', 'Customer_ID', True, False)
        summary_monthly_inhand_salary = summarize_numerical_column_with_deviation(df, 'Monthly_Inhand_Salary', 'Customer_ID', True, True)
        df['Annual_Income'][df['Monthly_Inhand_Salary'].notnull()] = df[df['Monthly_Inhand_Salary'].notnull()].groupby(['Customer_ID', 'Monthly_Inhand_Salary'], group_keys = False)['Annual_Income'].transform(return_mode)
        df['Monthly_Inhand_Salary'] = df.groupby(['Customer_ID', 'Annual_Income'], group_keys = False)['Monthly_Inhand_Salary'].transform(forward_backward_fill)
        df['Monthly_Inhand_Salary'].isnull().sum()
        df['Annual_Income'][df['Monthly_Inhand_Salary'].isnull()] = np.nan
        df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].transform(forward_backward_fill)
        df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].transform(forward_backward_fill)

        # Handling Number of Bank Accounts column
        summary_num_bank_accounts = summarize_numerical_column_with_deviation(df, 'Num_Bank_Accounts', median_standardization_summary = True)
        df['Num_Bank_Accounts'][df['Num_Bank_Accounts'] < 0] = np.nan
        df['Num_Bank_Accounts'][df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Bank_Accounts')).abs() > 2] = np.nan
        df['Num_Bank_Accounts'] = df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(forward_backward_fill).astype(int)

        # Handling Number of Credit Cards column
        summary_num_credit_cards = summarize_numerical_column_with_deviation(df, 'Num_Credit_Card', median_standardization_summary = True)
        df['Num_Credit_Card'][df.groupby('Customer_ID')['Num_Credit_Card'].transform(median_standardization, default_value = return_max_MAD(df, 'Num_Credit_Card')).abs() > 2] = np.nan
        df['Num_Credit_Card'] = df.groupby('Customer_ID')['Num_Credit_Card'].transform(forward_backward_fill).astype(int)

        # Handling Interest Rate column
        summary_interest_rate = summarize_numerical_column_with_deviation(df, 'Interest_Rate', median_standardization_summary = True)
        df['Interest_Rate'] = df.groupby('Customer_ID')['Interest_Rate'].transform(lambda x: x.median())

        # Handling Number of Loans column
        num_of_loans = df['Type_of_Loan'].str.split(', ').str.len()
        df['Num_of_Loan'][num_of_loans.notnull()] = num_of_loans[num_of_loans.notnull()]
        df['Num_of_Loan'][num_of_loans.isnull()] = 0
        df['Num_of_Loan'] = df.groupby('Customer_ID')['Num_of_Loan'].transform(forward_backward_fill).astype(int)
        
        df['Type_of_Loan'].fillna('No Loan', inplace = True)
        temp_series = df['Type_of_Loan']
        temp_lengths = temp_series.str.split(', ').str.len().astype(int) # Number of loans
        temp_lengths_max = temp_lengths.max()
        for index, val in temp_lengths.items():
            temp_series[index] = (temp_lengths_max - val) * 'No Loan, ' + temp_series[index]
        temp = temp_series.str.split(pat = ', ', expand = True)
        unique_loans = set()
        for col in temp.columns:
            temp[col] = temp[col].str.lstrip('and ')
            unique_loans.update(temp[col].unique())
            
        temp.columns = [f'Last_Loan_{i}' for i in range(int(df['Num_of_Loan'].max()), 0, -1)]
        df = pd.merge(df, temp, left_index = True, right_index = True)
        df.drop(columns = 'Type_of_Loan', inplace = True)
        # Handling Number of Delayed Payments column
        df['Num_of_Delayed_Payment'] = df.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(return_mode_median_filled_int).astype(int)

        # Handling Changed Credit Limit column
        df['Changed_Credit_Limit'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(return_mode_average_filled)

        # Handling Number of Credit Inquiries column
        df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(forward_backward_fill).astype(int)

        # Handling Credit Mix column
        df['Credit_Mix'] = df.groupby('Customer_ID')['Credit_Mix'].transform(forward_backward_fill)

        # Handling Credit History Age column
        df[['Years', 'Months']] = df['Credit_History_Age'].str.extract(r'(?P<Years>\d+) Years and (?P<Months>\d+) Months').astype(float) 
        df['Credit_History_Age'] = df['Years'] * 12 + df['Months']
        df.drop(columns = ['Years', 'Months'], inplace = True)
        df['Credit_History_Age'] = df.groupby('Customer_ID')['Credit_History_Age'].transform(fill_month_history).astype(int)

        # Handling Payment of Minimum Amount column
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes': 1, 'No': 0, 'NM': np.nan})
        df['Payment_of_Min_Amount'] = df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(lambda x: x.fillna(x.mode()[0]))
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({1: 'Yes', 0: 'No'})

        df['Total_EMI_per_month'] = df['Total_EMI_per_month'].astype(int)
    
    except Exception as e:
        # Handling exceptions
        sys.stdout.write(f'An error occurred: {str(e)}')
    
    finally:
    
        loading_thread.do_run = False
        loading_thread.join()
        # Displaying loading completion message
        end_time = time.time()
        sys.stdout.write(f'\rData cleaning {end_time - start_time:.2f} seconds.\n')
        sys.stdout.flush()
    
    return df

