#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
import re

def import_rawdata(path):
    raw = pd.read_csv(path)
    print(f"shape of data: {raw.shape}")
    print("preview data:\n", raw.head())
    return raw

def initialize_data(raw):
    """
    Split the raw data into metadata and body message
    """

    def split_body(string):
        first_idx = string.index('X-FileName')
        second_idx = string[first_idx:].index('\n')
        body = string[first_idx+second_idx:] 
        return body

    def split_meta(string):
        first_idx = string.index('X-FileName')
        second_idx = string[first_idx:].index('\n')
        body = string[:first_idx+second_idx]
        return body

    df = pd.DataFrame()
    df['body_msg'] = raw.message.map(split_body)
    df['meta_data'] = raw.message.map(split_meta)
    return df

def get_header(df_1):
    header = []
    for i in df_1.split('\n'):
        header.append(i[:i.index(':')])
    return header       

def metadata_split(string):
    output = []
    for i in range(len(header)-1): #need to handle last split
        output.append(\
            string[string.index(header[i])+len(header[i])+1:string.index(header[i+1])-1]) #+1 to surpass ":"
    output.append(string[string.index(header[-1])+len(header[-1])+1:])
    return output 

def metadata_by_col(df_metasplit, header):
    return pd.DataFrame(df_metasplit.to_list(), columns = header)


if __name__ == "__main__":

    if 'raw' not in globals():
        raw = import_rawdata("..\data\emails.csv")

    df = initialize_data(raw)

    header = get_header(df.iloc[0,1])
    df_metasplit = df['meta_data'].map(metadata_split)
    df_metaByCol = metadata_by_col(df_metasplit, header)

    df_final = pd.concat((raw,df, df_metaByCol), axis=1)
    df_final.to_csv('..\data\emails_processed.csv')
    

