#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
import re
import pickle

def import_pickle(path):
    df = pd.read_pickle(path)
    print(f'shape of data: {df.shape}\ncolumns: {df.columns}')
    return df

# read data, comment out when not used
def import_data(path):
    start = time()
    df = pd.read_csv(path, index_col = 0)
    print(f'time used for importing: {time() - start}s')
    print(f'shape of data: {df.shape}\ncolumns: {df.columns}')
    return df

def drop_columns(df, columns:list):
    return df.drop(columns, axis = 1)

def fill_empty(df_col):
    column = []
    for i in df_col:
        if i == " ":
            column.append(np.nan)
        else:
            column.append(i)
    return column

def drop_nan(df, columns:list):
    for i in columns:
        df = df[df[i].notna()]
    return df

def split_folder(string):
    """
    Get the lowest level folder name
    """
    return string.split('\\')[-1]

def count_nan(df):
    print('Percentage of valid value\n')
    print(df.count()/len(df)*100, '\n')
    print('Number of missing values\n')
    print(len(df) - df.count())
    return None

def split_file(string):
    return string.split('/')[0]

def emailAddress_type(string):
    return string[string.find('@')+1:]


if __name__ == "__main__":
    df = import_data("..\data\emails_processed.csv")
    df = drop_columns(df, ['message','meta_data','Mime-Version','Content-Type',\
        'Content-Transfer-Encoding'])

    for i in df.columns:   # fill empty cell " "
        df[i] = fill_empty(df[i])  
    df = drop_nan(df, ['To','Subject','X-To','X-FileName'])      # drop na in some columns
    df['X-Folder_Category'] = df['X-Folder'].map(split_folder)    # remove name in X-folder
    df['file'] = df['file'].map(split_file)  # get employee name
    df['address_type'] = df['From'].map(emailAddress_type)

    df.reset_index(inplace= True, drop=True)
    count_nan(df)
    previewdata = df.head(100)

    df.to_csv('..\data\emails_processed_cleaning_bySender_v2.csv')
    df.to_pickle('..\data\emails_processed_cleaning_bySender_v2.pkl')


