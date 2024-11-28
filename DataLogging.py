from datetime import datetime
import pandas as pd
import os
import time
from NeuralNetwork import train_NN
from DataCleaning import clean_data

def printf(format_string, *args, **kwargs):
    """Formatted print function with timestamps."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = format_string.format(*args, **kwargs)
    print(f"[{current_time}] {message}")

def load_data(csv_file):
    try:
        # Log the start of the script
        printf("Loading and cleaning input data set:")
        printf("************************************")
        printf("Starting Script")

        # Check if file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"File not found: {csv_file}")

        # Log loading training data set
        load_start_time = time.time()
        printf("Loading training data set")

        # Load the dataset
        df = pd.read_csv(csv_file)

        # Log total rows and columns
        total_columns = len(df.columns)
        total_rows = len(df)
        printf("Total Columns Read: {}", total_columns)
        printf("Total Rows Read: {}", total_rows)

        load_end_time = time.time()
        load_duration = round(load_end_time - load_start_time, 2)
        printf("Time to load is: {} seconds", load_duration)

        return df

    except Exception as e:
        printf("An error occurred during data loading and cleaning: {}", str(e))
        return None

# Example usage
csv_file = r"C:\Github\Credit_Score_Project\data\credit_score_data.csv"
df = load_data(csv_file)
if df is not None:
    clean_df = clean_data(df)
    print(clean_df.info())
    train_NN(clean_df)

