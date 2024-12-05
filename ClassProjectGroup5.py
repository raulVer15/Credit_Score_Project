# Course: Programming Languages 3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC DATA ANALYSIS
# date: 09/10/09
# Student 1: Serafin Arboleda
# Student 2: Darren Benavides
# Student 3: Raul Verduzco
# description: Implementation Basic Data Analysis Routines

import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import time

# cleaning data
import itertools
import sys
import threading
import warnings

# NN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

warnings.filterwarnings('ignore')

# load the data
def load_data():
    try:
        # finds all csv
        file = glob.glob('**/credit_score_data*.csv',
                        recursive= True,
                        include_hidden=True)
        df = pd.read_csv(file[0])
        return df
    except Exception as e:
        print_time("An error occurred during loading the data: {}", str(e))
        return None
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
        sys.stdout.write(f'\rcleaning...{frame}')
        sys.stdout.flush()
        time.sleep(0.1)
        
def clean_data(df):
    print_time("Performing Data Clean Up")
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
        print("An error has occurred, load data before cleaning")
    
    finally:
    
        loading_thread.do_run = False
        loading_thread.join()
        # Displaying loading completion message
        end_time = time.time()
        #sys.stdout.write(f'\rData cleaning {end_time - start_time:.2f} seconds.\n')
        sys.stdout.write("\n")
        sys.stdout.flush()
    
    return df

def train_NN(df):
    try:
        # Define features and target
        continuous_features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 
                            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                            'Delay_from_due_date', 'Num_of_Delayed_Payment',
                            'Changed_Credit_Limit', 'Outstanding_Debt',
                            'Credit_Utilization_Ratio', 'Credit_History_Age',
                            'Total_EMI_per_month', 'Amount_invested_monthly',
                            'Monthly_Balance', 'Num_Bank_Accounts', 'Num_Credit_Inquiries']

        categorical_features = ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 
                                'Payment_of_Min_Amount', 'Last_Loan_1', 'Last_Loan_2', 
                                'Last_Loan_3', 'Last_Loan_4', 'Last_Loan_5']

        target = ['Credit_Score']
        # Validate columns
        missing_columns = [col for col in continuous_features + categorical_features + target if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing in the dataset: {missing_columns}") 
        
        encoder = OneHotEncoder(handle_unknown="ignore")
        
        # Encode categorical features
        encoded_features = encoder.fit_transform(df[categorical_features])
        
        #Calculating column len
        num_continuous_features = len(continuous_features)
        num_encoded_columns = encoded_features.shape[1]
        total_columns = num_continuous_features + num_encoded_columns
        
        # Load data
        scaler = StandardScaler()
        scaled_continuous = scaler.fit_transform(df[continuous_features])
        
        
        encoded_target = encoder.fit_transform(df[target])
        encoded_target_df = pd.DataFrame(encoded_target.toarray(), columns=encoder.get_feature_names_out(target))
        df = pd.concat([df, encoded_target_df], axis=1)


        X = np.hstack([scaled_continuous, encoded_features.toarray()])
        y = encoded_target.toarray()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Get input dimensions
        total_columns = X_train.shape[1]

        # Build model
        model = Sequential()
        model.add(Dense(total_columns, input_dim=total_columns, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(total_columns * 2, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(total_columns * 4, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(total_columns * 2, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))

        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        print("Training model")
        # Train the model
        model.fit(X_train, y_train, #verbose=0,
                validation_split=0.2,
                epochs=40,
                batch_size=128,
                callbacks=[early_stopping, lr_scheduler])

        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Calculate metrics
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

        # Get current timestamp
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print metrics
        print("********************************")
        print("\n")
        print_time(f"Model Accuracy: {test_acc:.2f}")
        print_time(f"Model Precision: {precision:.2f}")
        print_time(f"Model Recall: {recall:.2f}")
        print_time(f"Model f1_score: {f1:.2f}")
        print_time(f"Model Confusion Matrix: \n")
        # Print confusion matrix using tabulate
        headers = ["Good", "Poor", "Standard"]
        table = tabulate(conf_matrix, headers=headers, tablefmt="grid")
        print(table)
        print("\n")
        # Visualize the confusion matrix
        #sns.heatmap(conf_matrix, annot=True, fmt='d')
        #plt.xlabel('Predicted')
        #plt.ylabel('Actual')
        #plt.show()
        # Return metrics in an array
        return [X_test, y_test, encoder, model]

    except Exception:
         # Handle any errors that occur
         return f"An error occurred: Load data before training"

    # finally:
    #     # Optional: Cleanup code or final statement
    #     print("Model training and evaluation process completed.")
def predictions(x_test, y_test, encoder, model):
    # Make Predictions
    predictions = model.predict(x_test)
    # getting y_test values
    y_tested = encoder.inverse_transform(y_test)
    # getting the value of the predictions
    y_predicted = encoder.inverse_transform(predictions)

    # printing the first 15 values of the test and predicted values 
    data = []
    for i in range(15):
        data.append([y_tested[i], y_predicted[i]])

    headers = ["True Value", "Predicted Value"]
    print(tabulate(data, headers=headers, tablefmt="grid"))

    print("\n")

def print_time(format_string, *args, **kwargs):
    """Formatted print function with timestamps."""
    current_time = datetime.now().strftime("%H:%M:%S")
    message = format_string.format(*args, **kwargs)
    print(f"[{current_time}] {message}")

def verfiy_option(option):
    try:
        int(option)
        return True 
        
    except ValueError:
        return False
def print_menu():
    title = "\n***** MENU *****"
    option_1 = "   (1) Load Data"
    option_2 = "   (2) Process Data"
    option_3 = "   (3) Train Neural Network"
    option_4 = "   (4) Generate Predictions"
    option_5 = "   (5) Quit"
    options = {title: 0, option_1:1, option_2:2, option_3:3, option_4:4, option_5:5}
    # MENU print statements
    for i in options:
        print(i)
    return options
    
def update_menu(options, pass_op):
    for key, value in list(options.items()):
        if value == int(pass_op):
            del options[key]
    for i in options:
        print(i)


# main
def main():
    # main loop
    options = print_menu()

    quit = True
    while (quit):
        # Request Option
        option = input(" \n Please select option: ")
        try:
            if verfiy_option(option):
                if (int(option) ==1):
                    print("\n")
                    print_time("Loading and cleaning input data set:")
                    print("************************************************")
                    load_start_time = time.time()
                    print_time("Starting Script")
                    df = load_data()
                    print_time("Loading training data set")
                    rows = df.shape[0]
                    cols = df.shape[1]
                    print_time(f"Total Columns Read: {cols}")
                    print_time(f"Total Rows Read: {rows}")
                    load_end_time = time.time()
                    load_duration = round(load_end_time - load_start_time, 2)
                    print("\n")
                    print_time("Time to load is: {} seconds", load_duration)

                    update_menu(options, option)

                elif(int(option) == 2):
                    print("\n")
                    print_time("  Process (Clean) data:")
                    print("************************************************")
                    clean_start_time = time.time()
                    df = clean_data(df)
                    cleaned_rows = df.shape[0]
                    print_time(f"Total Rows after cleaning is: {cleaned_rows}")
                    clean_end_time = time.time()
                    cleaning_duration = round(clean_end_time - clean_start_time, 2)
                    print("\n")
                    print_time("Time to clean is: {} seconds", cleaning_duration)
                    update_menu(options, option)

                elif(int(option) == 3):
                    print("\n")
                    print_time("Train Neural Network")
                    print("************************************************")
                    NN_start_time = time.time()
                    x_test, y_test, encoder, model = train_NN(df)
                    NN_end_time = time.time()
                    NN_duration = round(NN_end_time - NN_start_time, 2)
                    print("\n")
                    print_time("Time to train is: {} seconds", NN_duration)
                    update_menu(options, option)

                elif(int(option) == 4):
                    print("\n")
                    print_time("Generate Prediction")
                    print("************************************************")
                    predictions(x_test, y_test, encoder, model)
                    print_time(f"Generating prediction using selected Neural Network")
                    print_time(f"Size of training set")
                    print_time(f"Size of testing set")
                    print_time(f"Predictions generated (predictions.csv have been generated)....")
                    print_time(f"Size of testing set")
                    update_menu(options, option)

                elif(int(option) == 5):
                    print("\n")
                    print_time("Goodbye ...")
                    print("\n")
                    quit = False
                else:
                    print("Invalid input, select available options")
                    update_menu(options, 6)

            else:
                print(" Please enter a number options, try again")
        
        except Exception as e:
            print_time("An error occurred: {}", str(e))
            
main()
