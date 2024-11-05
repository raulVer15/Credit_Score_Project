# Course: Programming Languages 3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC DATA ANALYSIS
# date: 09/10/09
# Student 1: Serafin Arboleda
# Student 2: Darren Benavides
# Student 3: Daniel Velasquez
# Student 4: Raul Verduzco
# description: Implementation Basic Data Analysis Routines

import pandas as pd


def read_CSV():
    df = pd.read_csv('credit_score_data.csv')                   # read CSV to dateframe

    return df

def clean_df(df):
    
    df.drop(columns = 'Unnamed: 0', inplace = True)             # drop Unnamed column
    
    df.drop_duplicates()                                        # drops duplicates 
                                                                # drops if ID repeats
    df.drop_duplicates(subset=["ID"], keep="first", inplace=True)
    
    # drop repeating inputs of same user in the same month using CID
    df.drop_duplicates(subset=["Customer_ID", "Month"],keep="first", inplace=True)
    return df

def main():
    
    df = read_CSV()
    cleanedDf = clean_df(df)
    print(cleanedDf)

main()
