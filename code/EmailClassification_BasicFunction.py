#%%
import numpy as np
import pandas as pd
import numpy as np
from time import time
import os, re
from nltk.tokenize import RegexpTokenizer

def import_pickle(path:str):
    start = time()
    print('Start reading pickle')
    df = pd.read_pickle(path)
    print(f"shape: {df.shape}\ncolumns: {df.columns}")
    print(f"Data imported successfully!\n import time: {time() - start}s")
    return df

def extract_bodymsg(df):
    return pd.DataFrame(df['Subject'] + " " + df["body_msg"], columns = ['msg'], index=df.index)

def regex_tokenizer(text, stopwords, limit=False):
    output = []
    Tokenizer = RegexpTokenizer(r"n't|([a-z]+[-_]?[a-z]+)|[a-z]+[a-z0-9]*", gaps=False)
    for i in Tokenizer.tokenize(text.lower()):
        if i not in stopwords:
            output += [i]
        if type(limit) != bool and len(output) >= limit:
            break
    return output