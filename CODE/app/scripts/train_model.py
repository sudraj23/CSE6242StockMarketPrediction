import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib as plt
import datetime
import time
import numpy as np
import gc

#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)

# Preprocess News Data

#Using 2010 to 2016 data
def process_news_data(news_data):
    print("Shape : {}".format(news_data.shape))
    news_data = news_data.drop(news_data.columns[0], axis=1)
    news_data['time'] = pd.to_datetime(news_data['time'])
    news_data['sourceTimestamp'] = pd.to_datetime(news_data['sourceTimestamp'])
    news_data['firstCreated'] = pd.to_datetime(news_data['firstCreated'])
    news_data['provider'] = news_data['provider'].astype('category')
    news_data['subjects'] = news_data['subjects'].astype('category')
    news_data['audiences'] = news_data['audiences'].astype('category')
    news_data['assetCodes'] = news_data['assetCodes'].astype('category')
    news_data['assetName'] = news_data['assetName'].astype('category')
    news_data = news_data[news_data['time'].dt.year > 2009]
    
    news_data['rel_firstMention'] = 1.0*news_data['firstMentionSentence']/news_data['sentenceCount']
    news_data['rel_SentCount'] = 1.0*news_data['sentimentWordCount']/news_data['wordCount']
    
    return news_data

df1 = pd.read_csv("../data/news_train1.csv")
df1 = process_news_data(df1)
df2 = pd.read_csv("../data/news_train2.csv")
df2 = process_news_data(df2)
df3 = pd.read_csv("../data/news_train3.csv")
df3 = process_news_data(df3)
df4 = pd.read_csv("../data/news_train4.csv")
df4 = process_news_data(df4)
df5 = pd.read_csv("../data/news_train5.csv")
df5 = process_news_data(df5)
df6 = pd.read_csv("../data/news_train6.csv")
df6 = process_news_data(df6)
df7 = pd.read_csv("../data/news_train7.csv")
df7 = process_news_data(df7)
df8 = pd.read_csv("../data/news_train8.csv")
df8 = process_news_data(df8)
df9 = pd.read_csv("../data/news_train9.csv")
df9 = process_news_data(df9)
df10 = pd.read_csv("../data/news_train10.csv")
df10 = process_news_data(df10)
news_data = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10])
del df1,df2,df3,df4,df5,df6,df7,df8,df9,df10
# news_data = pd.concat([df1,df2,df3,df4])

news_data.shape

news_data = news_data[pd.notnull(news_data['headline'])]
news_data.shape

## Removing late report of news
news_data['news_delay'] = news_data['time'] - news_data['sourceTimestamp']
news_data = news_data[news_data.news_delay < datetime.timedelta(days=1)]
news_data['time'] = news_data['time'].dt.date
news_data.shape

# Analysis

news_columns = ['time','urgency', 'assetCodes', 'assetName', 'relevance', 'sentimentClass','rel_firstMention', 
                'firstMentionSentence', 'sentenceCount','rel_SentCount', 'sentimentWordCount', 'wordCount']
news_data = news_data[news_columns]
gc.collect()

tmp = news_data.groupby(['time', 'assetName'])['sentimentClass'].nunique()
print("Combination of days and assets : {}".format(tmp.count()))
print("Total sentiment counts for these combinations : {}".format(tmp.sum()))

# There are total 2480771 sentiment counts, spanning 1602799 combination of days and assets.
# Taking average will change the actual sentiment of the aggregated news.
# Need to find a way of weighted aggregating the news data.

# Preparing news data

print(news_data.columns)
news_data.groupby('urgency').describe()

news_data = news_data[(news_data['urgency'] != 2)]
news_data.groupby('urgency').describe()

range_sent = news_data['sentimentClass'].unique()
range_urg = news_data['urgency'].unique()
print("Unique Sentiment Values : {}".format(range_sent))
print("Unique Urgency Values:  {}".format(range_urg))

columns = ['rel_firstMention', 'rel_SentCount', 'relevance', 'firstMentionSentence', 
           'sentenceCount', 'sentimentWordCount', 'wordCount']
for i in range_sent:
    for j in range_urg:
        for col in columns:
            new_col = col + "_" + str(j) + '_' + str(i)
            news_data[new_col] = 0.0
            news_data.loc[((news_data['sentimentClass'] == i)  & (news_data['urgency'] == j)),new_col] = \
                news_data.loc[((news_data['sentimentClass'] == i)  & (news_data['urgency'] == j)),col]
news_data.drop(labels=columns + ['urgency','sentimentClass'], axis=1, inplace=True)
gc.collect()
print("News data Shape : {}".format(news_data.shape))

# Removed 'columns'
# 

### Preprocess Market Data 

df1 = pd.read_csv("../data/market_train1.csv")
df2 = pd.read_csv("../data/market_train2.csv")
df3 = pd.read_csv("../data/market_train3.csv")
df4 = pd.read_csv("../data/market_train4.csv")
df5 = pd.read_csv("../data/market_train5.csv")
market_data = pd.concat([df1,df2,df3,df4,df5])
del df1,df2,df3,df4,df5

gc.collect()

market_data = market_data.drop(market_data.columns[0], axis=1)
market_data['time'] = pd.to_datetime(market_data['time'])
market_data['assetName'] = market_data['assetName'].astype('category')
market_data.sort_values(['assetCode', 'time'], inplace=True)
market_data['returnsOpenNextMktres1'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(9)
market_data['returnsOpenNextMktres2'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(8)
market_data['returnsOpenNextMktres3'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(7)
market_data['returnsOpenNextMktres4'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(6)
market_data['returnsOpenNextMktres5'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(5)
market_data['returnsOpenNextMktres6'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(4)
market_data['returnsOpenNextMktres7'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(3)
market_data['returnsOpenNextMktres8'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(2)
market_data['returnsOpenNextMktres9'] = market_data.groupby('assetCode')['returnsOpenNextMktres10'].shift(1)

print("Min timestamp : {}, Max timestamp : {}, Market data shape : {}".format(market_data['time'].min(), 
                                                                              market_data['time'].max(),
                                                                              market_data.shape))
market_data = market_data[market_data['time'].dt.year > 2009]
print("Shape of market data after time filtering : {}".format(market_data.shape))

## Merging Dataframes

def data_prep(market_df,news_df):
    market_df['time'] = market_df.time.dt.date
    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])

    kcol = ['time', 'assetCodes']
    news_df = news_df.groupby(kcol, as_index=False).mean()

    market_df = pd.merge(market_df, news_df, how='left', left_on=['time', 'assetCode'], 
                            right_on=['time', 'assetCodes'])

    lbl = {k: v for v, k in enumerate(market_df['assetCode'].unique())}
    market_df['assetCodeT'] = market_df['assetCode'].map(lbl)
    
    market_df = market_df.dropna(axis=0)
    
    return market_df


market_news = data_prep(market_data, news_data)

market_news['datetime'] = pd.to_datetime(market_news['time'])
market_news['avgprice'] = (market_news['close'] + market_news['open'])/2.0
market_news['returnsOpenPrevMktres1_dir'] = market_news['returnsOpenPrevMktres1'].apply(lambda x: 0 if x<0 else 1)
market_news['returnsOpenPrevMktres10_dir'] = market_news['returnsOpenPrevMktres10'].apply(lambda x: 0 if x<0 else 1)
market_news.shape

gc.collect()
market_news.head()

feature_columns = [ 'returnsOpenPrevMktres1_dir', 'returnsOpenPrevMktres10_dir', 
                   'relevance_1_1', 'firstMentionSentence_1_1', 
        'sentimentWordCount_1_1', 
       'relevance_3_1', 'firstMentionSentence_3_1', 'sentimentWordCount_3_1',
       'relevance_1_0', 'firstMentionSentence_1_0', 
       'sentimentWordCount_1_0', 'relevance_3_0',
       'firstMentionSentence_3_0', 'sentimentWordCount_3_0', 
       'relevance_1_-1', 'firstMentionSentence_1_-1',
       'sentimentWordCount_1_-1',
       'relevance_3_-1', 
       'firstMentionSentence_3_-1', 
       'sentimentWordCount_3_-1', 
        'rel_SentCount_1_1', 'rel_SentCount_3_1', 'rel_firstMention_1_1', 'rel_firstMention_3_1', 
        'rel_firstMention_1_0', 'rel_SentCount_1_0', 'rel_firstMention_3_0', 'rel_firstMention_1_-1', 
        'rel_SentCount_3_0', 'rel_SentCount_1_-1', 'rel_firstMention_3_-1', 'rel_SentCount_3_-1']
                   
target_columns =['returnsOpenNextMktres10', 'returnsOpenNextMktres1','returnsOpenNextMktres2', 'returnsOpenNextMktres3',
                 'returnsOpenNextMktres4', 'returnsOpenNextMktres5', 'returnsOpenNextMktres6', 'returnsOpenNextMktres7',
                'returnsOpenNextMktres8', 'returnsOpenNextMktres9']


market_train = market_news[market_news['datetime'].dt.year < 2016]
market_test = market_news[market_news['datetime'].dt.year >= 2016]

market_train = market_train[feature_columns + target_columns] .fillna(0)
market_test = market_test[feature_columns + target_columns] .fillna(0)

X_train = market_train[feature_columns].values
X_test = market_test[feature_columns].values
up_train = market_train['returnsOpenNextMktres10'].map(lambda x: 0 if x<0 else 1).values
up_test = market_test['returnsOpenNextMktres10'].map(lambda x: 0 if x<0 else 1).values
Y_train = up_train
Y_test = up_test


### XG Boost 

import random
random.seed(1)
import time

import xgboost as xgb
# import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import get_scorer
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# # Set up decay learning rate
# def learning_rate_power(current_round):
#     base_learning_rate = 0.1
#     min_learning_rate = 0.01
#     lr = base_learning_rate * np.power(0.995,current_round)
#     return max(lr, min_learning_rate)

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

tune_params = {'n_estimators': [500,1000],
              'max_depth': [5,10],
              'colsample_bytree':[0.6,0.8],
              'min_child_weight': [5, 10],
              'gamma': [0.5,1.5],
              'reg_lambda':[1e-3,1e-1],
               'reg_alpha':[1e-3, 1e-1],
              'learning_rate':[0.1,0.01]}

fit_params = {'early_stopping_rounds':100,
              'eval_metric': 'auc',
              'eval_set': [(X_train, Y_train), (X_test, Y_test)],
              'verbose': 200
              }

# xgb_clf = xgb.XGBClassifier(objective='binary:logistic',silent=True, nthread=1)

# folds = 5
# param_comb = 40

# #skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# rs = RandomizedSearchCV(xgb_clf, param_distributions=tune_params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv = tscv.split(X_train), verbose=3, random_state=1001 )


#rs.fit(X_train, Y_train, **fit_params)

# rs.best_estimator_

### Building Model on best estimated parameters

xgb_clf = xgb.XGBClassifier(n_jobs=4,
                             objective='binary:logistic',
                            random_state=300)
opt_params = {'n_estimators': 5000,
              'max_depth': 8,
              'subsample':0.7,
              'colsample_bytree':0.8,
              'min_child_weight': 10,
              'gamma': 2,
              'reg_lambda':1,
               'reg_alpha':2,
              'learning_rate':0.01}
xgb_clf.set_params(**opt_params)
xgb_clf.fit(X_train, Y_train,**fit_params)

### Accuracies on Training and Validation set

print('Training accuracy: ', accuracy_score(Y_train, xgb_clf.predict(X_train)))
print('Validation accuracy: ', accuracy_score(Y_test, xgb_clf.predict(X_test)))

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    return(TP, FP, TN, FN)

(TP,FP,TN,FN) = perf_measure(Y_train, xgb_clf.predict(X_train))
#confusion_matrix(Y_train, lgb_clf.predict(X_train))
TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
print(TPR*100)
print(TNR*100)

(TP,FP,TN,FN)

# save the model to disk
import pickle
filename = '../static/finalized_model.sav'
pickle.dump(xgb_clf, open(filename, 'wb'))

## Visualizing Result

# features_imp = pd.DataFrame()
# features_imp['features'] = list(feature_columns)[:]
# features_imp['importance'] = xgb_clf.feature_importances_
# features_imp = features_imp.sort_values(by='importance', ascending=False).reset_index()

# y = -np.arange(16)
# plt.figure(figsize=(10,6))
# plt.barh(y, features_imp.loc[:15,'importance'].values)
# plt.yticks(y,(features_imp.loc[:15,'features']))
# plt.xlabel('Feature importance')
# plt.title('Features importance')
# plt.tight_layout()
# plt.savefig('features_importance.png')

# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()

# # columns_corr = ['takeSequence', 'companyCount','marketCommentary','sentenceCount',\
# #            'firstMentionSentence','relevance','sentimentClass','sentimentWordCount','noveltyCount24H',\
# #            'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D','returnsOpenNextMktres10']
# columns_corr = list(features_imp.loc[:15,'features'])
# colormap = plt.cm.RdBu
# plt.figure(figsize=(25,10))
# sns.heatmap(market_news[columns_corr].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
# plt.title('Pair-wise correlation')
# plt.savefig('correlation_matrix.png')

# ### Saving the Prediction Accuracy for 1st to 10th day

# accuracy = {}
# X_train = market_train[feature_columns].values
# X_test = market_test[feature_columns].values


# for i in range(1, 11):    
#     response = 'returnsOpenNextMktres' + str(i)
    
#     up_train = market_train[response].map(lambda x: 0 if x<0 else 1).values
#     up_test = market_test[response].map(lambda x: 0 if x<0 else 1).values
#     Y_train = up_train
#     Y_test = up_test
    
#     xgb_clf = xgb.XGBClassifier(n_jobs=4,
#                              objective='binary:logistic',
#                             random_state=300)
#     opt_params = {'n_estimators': 5000,
#                   'max_depth': 8,
#                   'subsample':0.7,
#                   'colsample_bytree':0.8,
#                   'min_child_weight': 10,
#                   'gamma': 2,
#                   'reg_lambda':1,
#                    'reg_alpha':2,
#                   'learning_rate':0.01}
#     fit_params = {'early_stopping_rounds':100,
#                   'eval_metric': 'auc',
#                   'eval_set': [(X_train, Y_train), (X_test, Y_test)],
#                   'verbose': 200
#                   }
#     xgb_clf.set_params(**opt_params)
#     xgb_clf.fit(X_train, Y_train,**fit_params)
    
#     train_acc = accuracy_score(Y_train, xgb_clf.predict(X_train))
#     test_acc = accuracy_score(Y_test, xgb_clf.predict(X_test))
    
#     print('Training accuracy: ', train_acc)
#     print('Validation accuracy: ', test_acc)
    
#     (TP,FP,TN,FN) = perf_measure(Y_train, xgb_clf.predict(X_train))
#     #confusion_matrix(Y_train, lgb_clf.predict(X_train))
#     TNR = TN/(TN+FP)
#     TPR = TP/(TP+FN)
#     print(TPR*100)
#     print(TNR*100)
    
#     accuracy[i] = [train_acc, test_acc, TPR, TNR]

# # Plot Train and Test accuracy
# train_acc = [val[0] for key, val in accuracy.items()]
# test_acc = [val[1] for key, val in accuracy.items()]
# x_axis = [key for key, val in accuracy.items()]
# plt.plot( x_axis, train_acc)
# plt.plot( x_axis, test_acc)
# plt.ylabel('XGB Train/Test Accuracy')
# plt.xlabel('$n^{th}$ day prediction')
# plt.legend(["train_accuracy", "test_accuracy"])
# plt.tight_layout()
# plt.savefig('train_test_accuracy.png')

# # Plot TPR and TNR accuracy
# tpr_acc = [val[2] for key, val in accuracy.items()]
# tnr_acc = [val[3] for key, val in accuracy.items()]
# x_axis = [key for key, val in accuracy.items()]
# plt.plot( x_axis, tpr_acc)
# plt.plot( x_axis, tnr_acc)
# plt.ylabel('XGB TPR/TNR')
# plt.xlabel('$n^{th}$ day prediction')
# plt.legend(["TPR", "TNR"])
# plt.tight_layout()
# plt.savefig('tpr_tnr.png')

# # Plot Test Accuracy
# plotdata = []
# plotdata.append([key for key, val in accuracy.items()])
# plotdata.append([val[0] for key, val in accuracy.items()])
# plt.plot(plotdata[0], plotdata[1])
# plt.axis([min(plotdata[0]), max(plotdata[0]), min(plotdata[1]), max(plotdata[1])])
# plt.ylabel('XGB Test Accuracy')
# plt.xlabel('$n^{th}$ day prediction')
# plt.show()