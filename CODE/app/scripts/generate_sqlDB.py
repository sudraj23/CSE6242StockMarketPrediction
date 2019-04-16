import pandas as pd
import datetime
import time
import numpy as np
import gc
import sqlite3
from pandas.io import sql
import re
import json
import os

### Market Data ####

df1 = pd.read_csv("../data/market_train1.csv")
df2 = pd.read_csv("../data/market_train2.csv")
df3 = pd.read_csv("../data/market_train3.csv")
df4 = pd.read_csv("../data/market_train4.csv")
df5 = pd.read_csv("../data/market_train5.csv")
market_data = pd.concat([df1,df2,df3,df4,df5])
del df1,df2,df3,df4,df5

print("Read market data!")

market_data = market_data.drop(market_data.columns[0], axis=1)
market_data['time'] = pd.to_datetime(market_data['time'])
market_data['assetName'] = market_data['assetName'].astype('category')
market_data.sort_values(['assetCode', 'time'], inplace=True)
market_data = market_data[market_data['time'].dt.year >= 2009]
print("Shape of market data after filtering : {}".format(market_data.shape))

def generate_asset_list(market_data):
    columns = ['time', 'assetName', 'volume', 'close', 'open', 
           'returnsOpenPrevMktres1', 'returnsOpenPrevMktres10', 'returnsOpenNextMktres10']
    market_data = market_data[columns]
    agg_data = market_data.groupby(['assetName']).count()['time']\
                    .reset_index(name='count').sort_values(['count'], ascending=False)
    asset_list  = agg_data['assetName'][1:1001].tolist()
    asset_list.sort()
    f = open("../static/asset.json", "w")
    f.write("""{"label":"label","items":[\n""")
    for i, name in enumerate(asset_list):
        if(i>=999):
            f.write("""{"label":[\"""" + name + """\"]}]}\n""")
        else:
            f.write("""{"label":[\"""" + name + """\"]},\n""")
    f.close()

generate_asset_list(market_data)
print("Asset List Generated")


def generate_market_db(market_data):
    with open("../static/asset.json", "r") as f:
        content = f.read()
    asset_list = json.loads(content)
    
    if not os.path.exists("../data/SQLdatabase"):
        os.makedirs("../data/SQLdatabase")
    conn = sqlite3.connect("../data/SQLdatabase/MarketData.db")

    for i in asset_list['items']:
        if i['label'][0] != 'Unknown':
            table_name = re.sub(r'\W+', '', i['label'][0])
            table_name="Market_" + table_name
            ind_stock = market_data[market_data['assetName'] == i['label'][0]]
            ind_stock = ind_stock.sort_values( 'time')
            ind_stock.to_sql(table_name, conn)
    conn.close()
    
generate_market_db(market_data)
print("Market Database created!")

del market_data

def process_news_data(news_data):

    news_data = news_data.drop(news_data.columns[0], axis=1)
    news_data['time'] = pd.to_datetime(news_data['time'])
    news_data['sourceTimestamp'] = pd.to_datetime(news_data['sourceTimestamp'])
    news_data['firstCreated'] = pd.to_datetime(news_data['firstCreated'])
    news_data['provider'] = news_data['provider'].astype('category')
    news_data['subjects'] = news_data['subjects'].astype('category')
    news_data['audiences'] = news_data['audiences'].astype('category')
    news_data['assetCodes'] = news_data['assetCodes'].astype('category')
    news_data['assetName'] = news_data['assetName'].astype('category')
    news_data = news_data[news_data['time'].dt.year >= 2009]
    
    news_data['rel_firstMention'] = 1.0*news_data['firstMentionSentence']/news_data['sentenceCount']
    news_data['rel_SentCount'] = 1.0*news_data['sentimentWordCount']/news_data['wordCount']
    
    news_data = news_data[pd.notnull(news_data['headline'])]
    news_data['news_delay'] = news_data['time'] - news_data['sourceTimestamp']
    news_data = news_data[news_data.news_delay < datetime.timedelta(days=1)]
    news_data['time'] = news_data['time'].dt.date
    news_data1 = news_data[pd.notnull(news_data['assetName'])]
    
    return news_data

print("Start Reading News Data")
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
print("Read 50% of News Data")
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
print("Read 100% of News Data")
news_data = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10])
del df1,df2,df3,df4,df5,df6,df7,df8,df9,df10

news_col = ['time', 'headline', 'urgency', 'sentenceCount', 'wordCount', 'assetName', 'firstMentionSentence', 
           'relevance', 'sentimentClass','sentimentWordCount','rel_firstMention', 'rel_SentCount']
news_data = news_data[news_col]
print("Shape of News data after filtering : {}".format(news_data.shape))

print("Gennerating news data db!")
def generate_newsDB(news_data):
    with open("../static/asset.json", "r") as f:
        content = f.read()
    asset_list = json.loads(content)
    
    if not os.path.exists("../data/SQLdatabase"):
        os.makedirs("../data/SQLdatabase")
        
    conn = sqlite3.connect("../data/SQLdatabase/NewsData.db")
    for i in asset_list['items']:
        if i['label'][0] != 'Unknown':
            table_name = re.sub(r'\W+', '', i['label'][0])
            table_name= "News_" + table_name
            ind_stock = news_data[news_data['assetName'] == i['label'][0]]
            ind_stock = ind_stock.sort_values( 'time')
            ind_stock.to_sql(table_name, conn)
    conn.close()  

generate_newsDB(news_data)
print("News Database created!")