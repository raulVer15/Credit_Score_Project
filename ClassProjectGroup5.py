# Course: Programming Languages 3500
# CLASS Project
# PYTHON IMPLEMENTATION: BASIC DATA ANALYSIS
# date: 09/10/09
# Student 1: Serafin Arboleda
# Student 2: Darren Benavides
# Student 3: Daniel Velasquez
# Student 4: Raul Verduzco
# description: Implementation Basic Data Analysis Routines

import os
import numpy as pd
import pandas as pd


def read_CSV():
    df = pd.read_csv('credit_score_data.csv')                   # read CSV to dateframe

    return df

def clean_df(df):

    # Step 1 - Dropping unnecessary columns
    df.drop(columns = 'Unnamed: 0', inplace = True)             # drop Unnamed column
    df.drop(columns = 'Name', inplace = True)                    # name col dropped

    # Step 2- correcting data types
    df['Age']= df['Age'].str.replace('_','')
    df['Age'] = df['Age'].astype(int)



    
    #df.drop_duplicates()                                        # drops duplicate    
    #df.drop_duplicates(subset=["ID"], keep="first", inplace=True)
    
    # drop repeating inputs of same user in the same month using CID
    #df.drop_duplicates(subset=["Customer_ID", "Month"],keep="first", inplace=True)
    return df

def main():
    
    df = read_CSV()
    cleanedDf = clean_df(df)
    print(cleanedDf.info())

main()
