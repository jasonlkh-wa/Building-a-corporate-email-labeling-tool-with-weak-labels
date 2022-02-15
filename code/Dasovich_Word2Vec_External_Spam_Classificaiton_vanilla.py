#%%
import pandas as pd
import numpy as np
from time import time 
import re, os
from IPython.display import display
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, WordNetCorpusReader, wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost
from EmailClassification_BasicFunction import regex_tokenizer, extract_bodymsg
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.base import TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, set_config
from gensim.models import Word2Vec, KeyedVectors, word2vec
from gensim.models.phrases import Phraser, Phrases
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore") # ignore the warnings
set_config(verbosity=0) # mute xgboost
pd.options.display.max_rows = 500
stopwords = stopwords.words('english')
stopwords.extend(['re','cc','fwd','fw'])
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


def word2vec_average(token_list:list, w2v_model):
    """
    Averaging the word2vec score of the input.
    """
    embedding_sum = np.zeros(w2v_model[0].shape)
    count = 0
    for i in token_list:
        try:
            temp = w2v_model[i]
            embedding_sum += w2v_model[i]
            count += 1
        except KeyError:
            pass      
    return embedding_sum/count


class Classifier():
    def __init__(self, **kwargs):
        self.logisticRegression = LogisticRegression(**kwargs.get('logisticRegression', {}))
        self.decisionTree = DecisionTreeClassifier(**kwargs.get('decisionTree', {}))
        self.randomForest = RandomForestClassifier(**kwargs.get('randomForest', {}))
        self.svc = SVC(**kwargs.get('svc', {}))
        self.xgboost = XGBClassifier(**kwargs.get('xgboost', {}))
        #self.gaussianNB = GaussianNB(**kwargs.get('gaussianNB', {}))  # users can add any models here

    def fitall(self, x_train, y_train):
        """
        Fit all the models within the instance.
        """
        models = [i for i in dir(self) if not i.startswith("__") and not callable(getattr(self, i))]
        for model in models:
            getattr(self, model).fit(x_train, y_train)
        print('-----All models are fit-----')

    def cross_val_score(self, x_train, y_train, cv=3):
        """
        Perform cross validation among the models.
        """
        models = [i for i in dir(self) if not i.startswith("__") and not callable(getattr(self, i))]
        for model in models:
            cross_val = cross_val_score(getattr(self, model), x_train, y_train, cv=cv, verbose=True)
            print(f"model: {model}, avg_score: {cross_val.mean()}\nscores: {cross_val}")
    
    def overfit_test(self, x_train, y_train, x_test, y_test, model=False):
        """
        Calculate the accuracy, precision and recall of both train set and test set.
        """
        print('Overfit Test')
        if model == False:
            models = [i for i in dir(self) if not i.startswith("__") and not callable(getattr(self, i))]
        else:
            models = [model]
        for mdl in models:
            getattr(self, mdl).fit(x_train, y_train)
            train_prediction, test_prediction = getattr(self, mdl).predict(x_train), getattr(self, mdl).predict(x_test)
            print(f'\n{mdl} train accuracy, precision, recall:\n{getattr(self, mdl).score(x_train, y_train)}, {precision_score(y_train, train_prediction)}, {recall_score(y_train, train_prediction)}')
            print(f'{mdl} test accuracy, precision, recall:\n{getattr(self, mdl).score(x_test, y_test)}, {precision_score(y_test, test_prediction)}, {recall_score(y_test, test_prediction)}')
        print('Overfit Test Ended')

    def gridsearchcv(self, model, x_train, y_train, param_grid):
        """
        Perform Grid Search on one of the models and return the grid.
        """
        grid = GridSearchCV(getattr(self, model), param_grid=param_grid, cv=3, verbose=True, return_train_score=True, scoring=['precision', 'recall', 'f1'], refit='precision')
        grid.fit(x_train, y_train)
        print(f"best params: {grid.best_params_}\nbest score: {grid.best_score_}")
        return grid

    def predict(self, x_train, model=False)->dict:
        """
        Predicts the result and saved all the result in a dictionary. The result can retrieved by calling the dictionary with the model name defined in Classifier, e.g. dict['logisticRegression']
        """
        print('Predicting the result')
        output = {}
        if model == False:
            models = [i for i in dir(self) if not i.startswith("__") and not callable(getattr(self, i))]
        else:
            models = [model]
        for mdl in models:
            output[mdl] = getattr(self, mdl).predict(x_train)
        print('All predictions are done!')
        return output

    def majority_predict(self, x_train, model=None):
        """
        Predict the result based on majority vote approach.
        """
        output = self.predict(x_train)
        pred = np.zeros(len(x_train))
        model_count = 0
        for key, i in output.items():
            if model == None or key in model:
                model_count += 1
                pred += i
        pred = (pred>model_count//2).astype(np.float)
        return pred

def ensemble_result(x, clf, models=['logisticRegression', 'svc', 'xgboost']):
    return pd.DataFrame([getattr(clf, model).predict(x) for model in models], index=models, columns=x.index).T

def print_prediction_metrics(prediction, y, set_name=''):
    """
    Print all the model evaluation metrics.
    """
    print(f"{set_name} set result:")
    print(confusion_matrix(y, prediction))
    print('precision:', precision_score(y, prediction))
    print('recall:', recall_score(y, prediction))
    print('accuracy:', sklearn.metrics.accuracy_score(y, prediction))
    print('f1_score:', f1_score(y, prediction))


if __name__ == '__main__':
    labeled_data, unlabeled_data, validation_set = load(r'..\data\dasovich_labeled_data_v3.joblib')
    snorkel_hand_label_set = pd.read_csv(r'..\data\dasovich_snorkel400_ground_label.csv', index_col=0)

    labeled_data['msg'] = extract_bodymsg(labeled_data)
    labeled_data['msg_token'] = labeled_data['msg'].apply(regex_tokenizer, stopwords=stopwords, limit=False)
    
    x_train, x_test, y_train, y_test = train_test_split(labeled_data['msg_token'], labeled_data['label'], test_size=0.3, shuffle=True, random_state=1)

    w2v_model_trained = word2vec.Word2Vec(x_train, vector_size=300, window=10, min_count=2, workers=1, epochs=16, alpha=0.03, min_alpha=0.0007, compute_loss=True, seed=1)
    print(w2v_model_trained.get_latest_training_loss())
    w2v_model = w2v_model_trained.wv
    
    # Word Embedding with w2v model
    x_train = x_train.apply(word2vec_average, w2v_model=w2v_model)
    x_test = x_test.apply(word2vec_average, w2v_model=w2v_model)
    x_train = pd.DataFrame(x_train.tolist(), index=x_train.index).add_prefix('dim_')
    x_test = pd.DataFrame(x_test.tolist(), index=x_test.index).add_prefix('dim_')

    # Drop nan data
    print(f'x_train dropped: {len(x_train) - len(x_train.dropna())}')
    print(f'x_test dropped: {len(x_test) - len(x_test.dropna())}')
    x_train, x_test = x_train.dropna(), x_test.dropna() 
    y_train, y_test = y_train.loc[x_train.index], y_test.loc[x_test.index]  # remove corresponding y label 

    # Building the individual models
    clf = Classifier(
                    logisticRegression={'C':0.2, 'class_weight':{0:2, 1:1}},
                    svc={'C':0.3, 'kernel':'linear', 'class_weight':{0:2, 1:1}},
                    decisionTree={'max_depth':2, 'random_state':1, 'class_weight':{0:2, 1:1}},
                    randomForest={'max_depth':3, 'n_estimators':50, 'random_state':1, 'class_weight':{0:2, 1:1}},
                    xgboost={'eta':0.2, 'n_estimators':25, 'max_depth':2, 'random_state':1, 'class_weight':{0:2, 1:1}})
    
    clf.fitall(x_train, y_train)
    clf.cross_val_score(x_train, y_train)
    clf.overfit_test(x_train, y_train, x_test, y_test)
    
    # Building the Ensemble Model
    ensemble = LogisticRegression(C=0.85)
    ensemble_x_train = ensemble_result(x_train, clf, models=['logisticRegression', 'randomForest', 'svc', 'xgboost']) 
    ensemble.fit(ensemble_x_train, y_train)
    train_ensemble_prediction = ensemble.predict(ensemble_x_train)
    print_prediction_metrics(train_ensemble_prediction, y_train, "train")

    # Compare XGBoost vs Ensemble model
    test_maj_prediction = clf.majority_predict(x_test, model=['logisticRegression', 'svc', 'xgboost'])  # xgboost alone gives best precision and recall
    print_prediction_metrics(test_maj_prediction, y_test)
    ensemble_test = ensemble_result(x_test, clf, models=['logisticRegression', 'randomForest', 'svc', 'xgboost']) 
    test_ensemble_prediction = ensemble.predict(ensemble_test)
    print_prediction_metrics(test_ensemble_prediction, y_test, "test")

    # Test the mode with validation set    
    validation_set['msg'] = extract_bodymsg(validation_set)
    validation_set['msg_token'] = validation_set['msg'].apply(regex_tokenizer, stopwords=stopwords, limit=False)
    x_validate = validation_set['msg_token'].apply(word2vec_average, w2v_model=w2v_model)
    x_validate = pd.DataFrame(x_validate.tolist(), index=x_validate.index).add_prefix('dim_')
    y_validate = pd.read_csv(r'..\data\dasovich_400_ground_label.csv')['label']

    validate_maj_prediction = clf.majority_predict(x_validate, model=['logisticRegression', 'svc', 'xgboost'])
    print_prediction_metrics(validate_maj_prediction, y_validate)
    ensemble_validate = ensemble_result(x_validate, clf, models=['logisticRegression', 'randomForest', 'svc', 'xgboost'])
    validate_ensemble_prediction = ensemble.predict(ensemble_validate)
    print_prediction_metrics(validate_ensemble_prediction, y_validate, "validation")

    # Check the performance on 400 label in snorkel set
    x_snorkel = labeled_data.loc[snorkel_hand_label_set.index]
    x_snorkel = x_snorkel['msg_token'].apply(word2vec_average, w2v_model=w2v_model)
    x_snorkel = pd.DataFrame(x_snorkel.tolist(), index=x_snorkel.index).add_prefix('dim_')
    y_snorkel = snorkel_hand_label_set['label']

    snorkel_maj_prediction = clf.majority_predict(x_snorkel, model=['logisticRegression', 'svc', 'xgboost'])
    print_prediction_metrics(snorkel_maj_prediction, y_snorkel)
    ensemble_snorkel = ensemble_result(x_snorkel, clf, models=['logisticRegression', 'randomForest', 'svc', 'xgboost'])
    snorkel_ensemble_prediction = ensemble.predict(ensemble_snorkel)
    print_prediction_metrics(snorkel_ensemble_prediction, y_snorkel, "snorkel")
    
    ### output the data to topic model ###    
    # Export test set predictions
    ensemble_test = ensemble_result(x_test, clf, models=['logisticRegression', 'randomForest', 'svc', 'xgboost']) 
    test_ensemble_prediction = ensemble.predict(ensemble_test)
    final_output_test = pd.concat([labeled_data.loc[x_test.index], pd.DataFrame(test_ensemble_prediction, columns=['pred_label'], index=x_test.index)], axis=1)
 
    # Predict Train set for output dataset
    ensemble_train = ensemble_result(x_train, clf, models=['logisticRegression', 'randomForest', 'svc', 'xgboost']) 
    train_ensemble_prediction = ensemble.predict(ensemble_train)
    final_output_train = pd.concat([labeled_data.loc[x_train.index], pd.DataFrame(train_ensemble_prediction, columns=['pred_label'], index=x_train.index)], axis=1)

    # Validate set for output dataset
    final_output_validation = validation_set
    final_output_validation['label'] = np.array(y_validate)
    
    # Unlabeled set for output dataset
    unlabeled_data['msg'] = extract_bodymsg(unlabeled_data).applymap(str.lower)
    unlabeled_data['msg_token'] = unlabeled_data['msg'].apply(regex_tokenizer, stopwords=stopwords, limit=False)
    
    x_unlabeled = unlabeled_data['msg_token'].apply(word2vec_average, w2v_model=w2v_model)
    x_unlabeled = pd.DataFrame(x_unlabeled.tolist(), index=x_unlabeled.index).add_prefix('dim_')
    ensemble_unlabeled = ensemble_result(x_unlabeled, clf, models=['logisticRegression', 'randomForest', 'svc', 'xgboost'])
    unlabeled_ensemble_prediction = ensemble.predict(ensemble_unlabeled)
    final_output_unlabeled = pd.concat([unlabeled_data, pd.DataFrame(unlabeled_ensemble_prediction, columns=['pred_label'], index=unlabeled_data.index)], axis=1)
  
    final_output = pd.concat([final_output_train, final_output_test, final_output_unlabeled, final_output_validation], axis=0)
    final_output = final_output.drop(['count_to_enron', 'msg', 'label'], axis=1)

    # add the snorkel hand label to increase quality of the output data
    final_output.loc[snorkel_hand_label_set.index, 'pred_label'] = snorkel_hand_label_set.loc[:, 'label']
    dump(final_output, r'..\data\dasovich_external_spam_result_v2.3') 

    print('\nEnd of Program')

#%% for grid search only 
if __name__ == '__main__':
    # grid = GridSearchCV(ensemble, param_grid={'C':[i*0.05 for i in range(1, 21)]}, cv=3, verbose=True, return_train_score=True, scoring=['precision', 'recall', 'f1'], refit='precision').fit(ensemble_x_train, y_train)
    # grid = clf.gridsearchcv('xgboost', x_train, y_train, param_grid={'eta':[0.03], 'n_estimators':[25,50], 'max_depth':[2,3,4], 'random_state':[1]})
    # grid_result = grid.cv_results_
    # temp_df = pd.DataFrame([grid_result['mean_train_precision'], grid_result['mean_test_precision'], grid_result['mean_train_recall'], grid_result['mean_test_recall'], grid_result['mean_train_f1'],\
    #     grid_result['mean_test_f1']]).T
    pass
