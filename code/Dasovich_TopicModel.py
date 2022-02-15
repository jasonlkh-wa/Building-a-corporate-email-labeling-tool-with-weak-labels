#%% 
from urllib.request import DataHandler
from joblib import dump, load
from nltk import pos_tag
from nltk.tokenize.regexp import regexp_tokenize
from numpy.lib.function_base import vectorize
import pandas as pd
import numpy as np
import pickle
from IPython.display import display
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
import re, os
from EmailClassification_BasicFunction import import_pickle, extract_bodymsg, regex_tokenizer
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stopwords = stopwords.words('english')
stopwords.extend(['re','cc','fwd','fw'])
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, silhouette_score, calinski_harabasz_score
from sklearn.base import TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.models import Word2Vec, KeyedVectors, LdaModel, CoherenceModel, LsiModel#, wrappers
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
import spacy
import warnings
warnings.filterwarnings("ignore") # ignore the warnings
pd.options.display.max_rows = 500
np.random.seed(1) # set seed for sklearn

wnl = WordNetLemmatizer()
def lemmatization(sentence:list):
    """
    Detect the part of speech of the word and perform word lemmatization.
    """
    output = []
    for word, tag in pos_tag(sentence):
        tag = tag[0].lower() if tag[0].lower() in ['a', 'r', 'n', 'v'] else 'n'
        output.append(wnl.lemmatize(word, pos=tag))
    return output

if __name__ == '__main__':
    external_raw = load(r'..\data\dasovich_external_spam_result_v2.3')  # import data
    external_raw = external_raw.loc[external_raw['pred_label']==0]  # drop spam emails
    internal_raw = load(r'..\data\dasovich_internal_emails')
    raw = pd.concat([external_raw, internal_raw], axis=0)
    raw['msg'] = extract_bodymsg(raw)
    raw['msg_token'] = raw['msg'].apply(regex_tokenizer, stopwords=stopwords, limit=False)
    raw = raw.sort_values(by='Date', axis=0)
    
    x_train, x_test = train_test_split(raw['msg_token'], test_size=0.3, shuffle=False)

    # Lemmatization
    print('---- preprocess: lemmatization ----')
    x_train = x_train.apply(lemmatization)
    x_test = x_test.apply(lemmatization)

    # Building Bigram
    print('---- preprocess: bigram ----')
    bigram = Phrases(x_train, min_count=5, threshold=100)
    bigram_mod = Phraser(bigram)
    x_train = x_train.apply(lambda x: bigram_mod[x])
    x_test = x_test.apply(lambda x: bigram_mod[x])

    # Corpora
    id2word = corpora.Dictionary(x_train)
    train_corpus = [id2word.doc2bow(i) for i in x_train]
    test_corpus = [id2word.doc2bow(i) for i in x_test]
    print('---- fitting model ----')

    # Lda model
    n_topic = 4
    lda = LdaModel(corpus=train_corpus, id2word=id2word, num_topics=n_topic, update_every=1,
                chunksize=100, passes=10, alpha='asymmetric', per_word_topics=True, random_state=1)

    topic_dict = {}
    for i in lda.print_topics(num_words=20):
        print(f'Topic: {i[0]}')
        topic_word_df = pd.DataFrame([], columns=['weight'])
        word_list = i[1].split('+')
        for j in word_list:
            weight, word = j.split('*')
            topic_word_df = topic_word_df.append(pd.DataFrame(weight, index=[word], columns=['weight']))
        display(topic_word_df)
        topic_dict[i[0]] = topic_word_df

    
    # Measure performance
    lda_coherence = CoherenceModel(model=lda, texts=x_train, dictionary=id2word, coherence='c_v')
    coherence_score = lda_coherence.get_coherence()
    print(f'\nCoherence Score of {n_topic}-topics: ', coherence_score)

    # Measure test data performance
    test_lda_coherence = CoherenceModel(model=lda, texts=x_test, dictionary=id2word, coherence='c_v')
    test_coherence_score = test_lda_coherence.get_coherence()
    print(f'\nCoherence Score of test data: ', test_coherence_score)

    # Show distribution of spam emails in topics
    topic_df = pd.DataFrame(columns=list(range(n_topic)))
    for i in lda.get_document_topics(train_corpus):
        temp_df = pd.DataFrame(data=[[j[1] for j in i]], columns=[j[0] for j in i])
        topic_df = pd.concat([topic_df, temp_df], axis=0, join='outer')
    topic_df = topic_df.fillna(0)
    topic_df.index = x_train.index
    topic_df = pd.concat([raw, topic_df], axis=1, join='inner')
    print(f'topic distribution:\n{topic_df.loc[:, list(range(n_topic))].idxmax(axis=1).value_counts()}')
    
    topic_analysis = topic_df.loc[:, ['Subject', 'body_msg']+list(range(n_topic))]
    preview_topic_top3 = topic_analysis.sort_values(by=1, ascending=False).sample(3, random_state=1)  # randomly select 3 emails from topics to check if the topic identified makes sense
    print(preview_topic_top3.loc[65452, 'body_msg'])

    # # Lda Mallet  # requires user to download the Mallet model before running the code
    # mallet_path = r'..\model\mallet-2.0.8\bin\mallet'
    # for n_topic in range(21,41):
    #     ldamallet = wrappers.LdaMallet(mallet_path, corpus=train_corpus, num_topics=n_topic, id2word=id2word)
    #     ldamallet_coherence = CoherenceModel(model=ldamallet, texts=x_train, dictionary=id2word, coherence='c_v')
    #     coherence_score = ldamallet_coherence.get_coherence()
    #     print(f'\nCoherence Score of {n_topic}-topics: ', coherence_score)

    # # Measure performance of LSA model
    # lsa = LsiModel(corpus=train_corpus, id2word=id2word, num_topics=n_topic,chunksize=100)
    # lsa_coherence = CoherenceModel(model=lsa, texts=x_train, dictionary=id2word, coherence='c_v')
    # coherence_score = lsa_coherence.get_coherence()
    # print(f'\nCoherence Score of {n_topic}-topics: ', coherence_score)

    print('\n-----End of Program-----')
