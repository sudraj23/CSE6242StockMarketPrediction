#import logging
#from logging.handlers import RotatingFileHandler
from flask import render_template
from flask import url_for
from app import app
import sqlite3
from flask import Flask, request, g
import csv
import json
import sys
import re
import pandas as pd
import matplotlib as plt
import datetime
import time
import numpy as np
import gc
import sqlite3
import re
import pickle

# DATABASE = 'static/SQLdatabase/market_data.db'
DATABASE = 'data/SQLdatabase/MarketData.db'

# DATABASEn = 'static/SQLdatabase/news_data1.db'

DATABASEn = 'data/SQLdatabase/NewsData.db'

MLmodel='static/finalized_model.sav'

loaded_model = pickle.load(open(MLmodel, 'rb'))

app.config.from_object(__name__)

def connect_to_database():
    return sqlite3.connect(DATABASE)

def connect_to_databasen():
    return sqlite3.connect(DATABASEn)


def get_db():
    db = getattr(g, 'db', None)
    if db is None:
        db = g.db = connect_to_database()
    return db

def get_dbn():
    dbn = getattr(g, 'dbn', None)
    if dbn is None:
        dbn = g.dbn = connect_to_databasen()
    return dbn


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()

def close_connectionn(exception):
    dbn = getattr(g, 'dbn', None)
    if dbn is not None:
        dbn.close()



def execute_query(query, args=()):
    cur = get_db().execute(query, args)
    rows = cur.fetchall()
    cur.close()
    return rows

def execute_queryn(query, args=()):
    curn = get_dbn().execute(query, args)
    rowsn = curn.fetchall()
    curn.close()
    return rowsn


def make_prediction(assetName, date):
    news_con = sqlite3.connect(DATABASEn)
    table_name = re.sub(r'\W+', '', assetName)
    table_name = "News_" + table_name
    print(assetName)
    print(table_name)
    query = """SELECT * FROM """ + table_name  + """ WHERE "time" = ?"""
    news_df = pd.read_sql(sql=query, con=news_con, params=[date])
    news_con.close()
    market_con = sqlite3.connect(DATABASE)

    table_name = re.sub(r'\W+', '', assetName)
    table_name = "Market_" + table_name
    query = """SELECT * FROM """ + table_name  + """ WHERE "time" = ?"""
    date_time = date + " 22:00:00"
    market_df = pd.read_sql(sql=query, con=market_con, params=[date_time])
    market_con.close()

    range_sent = [ 1, 0, -1]
    range_urg = [1, 3]
    # print("Unique Sentiment Values : {}".format(range_sent))
    # print("Unique Urgency Values:  {}".format(range_urg))

    cols_filtered = ['rel_firstMention', 'rel_SentCount', 'relevance', 'firstMentionSentence',
               'sentenceCount', 'sentimentWordCount', 'wordCount']
    for i in range_sent:
        for j in range_urg:
            for col in cols_filtered:
                new_col = col + "_" + str(j) + '_' + str(i)
                news_df[new_col] = 0.0
                news_df.loc[((news_df['sentimentClass'] == i)  & (news_df['urgency'] == j)),new_col] =                     news_df.loc[((news_df['sentimentClass'] == i)  & (news_df['urgency'] == j)),col]
    news_df.drop(labels=cols_filtered + ['urgency','sentimentClass'], axis=1, inplace=True)
    gc.collect()

    news_df['returnsOpenPrevMktres1']  = float(market_df['returnsOpenPrevMktres1'])
    news_df['returnsOpenPrevMktres10'] = float(market_df['returnsOpenPrevMktres10'])
    news_df['returnsOpenPrevMktres1_dir'] = news_df['returnsOpenPrevMktres1'].apply(lambda x: 0 if x<0 else 1)
    news_df['returnsOpenPrevMktres10_dir'] = news_df['returnsOpenPrevMktres10'].apply(lambda x: 0 if x<0 else 1)

    req_feature_columns = ['returnsOpenPrevMktres1_dir', 'returnsOpenPrevMktres10_dir', 'relevance_1_1',
                           'firstMentionSentence_1_1', 'sentimentWordCount_1_1', 'relevance_3_1',
                           'firstMentionSentence_3_1', 'sentimentWordCount_3_1', 'relevance_1_0',
                           'firstMentionSentence_1_0', 'sentimentWordCount_1_0', 'relevance_3_0',
                           'firstMentionSentence_3_0', 'sentimentWordCount_3_0', 'relevance_1_-1',
                           'firstMentionSentence_1_-1','sentimentWordCount_1_-1','relevance_3_-1',
                           'firstMentionSentence_3_-1', 'sentimentWordCount_3_-1', 'rel_SentCount_1_1',
                           'rel_SentCount_3_1', 'rel_firstMention_1_1', 'rel_firstMention_3_1',
                           'rel_firstMention_1_0', 'rel_SentCount_1_0', 'rel_firstMention_3_0', 'rel_firstMention_1_-1',
                           'rel_SentCount_3_0', 'rel_SentCount_1_-1', 'rel_firstMention_3_-1', 'rel_SentCount_3_-1']
    X_data = news_df[req_feature_columns].values
    #loaded_model = pickle.load(open(MLmodel, 'rb'))
    Y_predict = loaded_model.predict(X_data)
    X_mean = news_df.groupby(['time','assetName'], as_index=False).mean()[req_feature_columns].values
    Y_predict_mean = loaded_model.predict(X_mean)
    del news_df, market_df
    return Y_predict, Y_predict_mean


def predict_stock(assetName, start_date, end_date):

    market_con = sqlite3.connect(DATABASE)
    table_name = re.sub(r'\W+', '', assetName)

    table_name = "Market_" + table_name
    print(assetName)
    print(table_name)
    query = """SELECT * FROM """ + table_name  + """ WHERE "time" >= ? AND "time" <= ? """
    start_date_time = start_date + " 22:00:00"
    end_date_time = end_date + " 22:00:00"
    market_df = pd.read_sql(sql=query, con=market_con, params=[start_date_time, end_date_time])
    market_con.close()

    news_con = sqlite3.connect(DATABASEn)
    table_name = re.sub(r'\W+', '', assetName)
    table_name = "News_" + table_name
    print(assetName)
    print(table_name)
    query = """SELECT * FROM """ + table_name  + """ WHERE "time" >= ? AND "time" <= ? """
    news_df = pd.read_sql(sql=query, con=news_con, params=[start_date, end_date])
    news_con.close()

    range_sent = [ 1, 0, -1]
    range_urg = [1, 3]
    cols_filtered = ['rel_firstMention', 'rel_SentCount', 'relevance', 'firstMentionSentence',
               'sentenceCount', 'sentimentWordCount', 'wordCount']
    for i in range_sent:
        for j in range_urg:
            for col in cols_filtered:
                new_col = col + "_" + str(j) + '_' + str(i)
                news_df[new_col] = 0.0
                news_df.loc[((news_df['sentimentClass'] == i)  & (news_df['urgency'] == j)),new_col] = \
                    news_df.loc[((news_df['sentimentClass'] == i)  & (news_df['urgency'] == j)),col]
    news_df.drop(labels=cols_filtered + ['urgency','sentimentClass'], axis=1, inplace=True)
    gc.collect()

    market_df['time'] = pd.to_datetime(market_df['time']).dt.date
    news_df['time'] = pd.to_datetime(news_df['time']).dt.date

    def data_prep(market_df,news_df):
        kcol = ['time']
        news_df = news_df.groupby(kcol, as_index=False).mean()
        market_df = pd.merge(market_df, news_df, how='left', left_on=['time'], right_on=['time'])
        null_df = market_df[market_df.isna().any(axis=1)]
        market_df = market_df.dropna(axis=0)
        return null_df, market_df

    null_df, market_news = data_prep(market_df, news_df)

    del news_df, market_df

    market_news['returnsOpenPrevMktres1_dir'] = market_news['returnsOpenPrevMktres1'].apply(lambda x: 0 if x<0 else 1)
    market_news['returnsOpenPrevMktres10_dir'] = market_news['returnsOpenPrevMktres10'].apply(lambda x: 0 if x<0 else 1)

    req_feature_columns = ['returnsOpenPrevMktres1_dir', 'returnsOpenPrevMktres10_dir', 'relevance_1_1',
                               'firstMentionSentence_1_1', 'sentimentWordCount_1_1', 'relevance_3_1',
                               'firstMentionSentence_3_1', 'sentimentWordCount_3_1', 'relevance_1_0',
                               'firstMentionSentence_1_0', 'sentimentWordCount_1_0', 'relevance_3_0',
                               'firstMentionSentence_3_0', 'sentimentWordCount_3_0', 'relevance_1_-1',
                               'firstMentionSentence_1_-1','sentimentWordCount_1_-1','relevance_3_-1',
                               'firstMentionSentence_3_-1', 'sentimentWordCount_3_-1', 'rel_SentCount_1_1',
                               'rel_SentCount_3_1', 'rel_firstMention_1_1', 'rel_firstMention_3_1',
                               'rel_firstMention_1_0', 'rel_SentCount_1_0', 'rel_firstMention_3_0', 'rel_firstMention_1_-1',
                               'rel_SentCount_3_0', 'rel_SentCount_1_-1', 'rel_firstMention_3_-1', 'rel_SentCount_3_-1']

    X_data = market_news[req_feature_columns].values
    Y_predict = loaded_model.predict(X_data)
    market_news['predicted'] = Y_predict
    market_news['expected'] = market_news['returnsOpenNextMktres10'].map(lambda x: 0 if x<0 else 1)
    return_col = ['time', 'volume', 'close', 'open','predicted','expected']
    market_news = market_news[return_col]

    null_df['expected'] = null_df['returnsOpenNextMktres10'].map(lambda x: 0 if x<0 else 1)
    null_df['predicted'] = null_df['expected']
    null_df = null_df[return_col]

    final_df = pd.concat([market_news, null_df]).sort_values(['time'])
    final_df['average'] = (final_df['open'] + final_df['close'])/2.0
    final_df['time'] = pd.to_datetime( final_df['time']).dt.strftime('%d/%m/%Y')

    del null_df, market_news
    gc.collect()
    return final_df




@app.route('/index')
def index():
	return render_template('index.html')
@app.route('/')
def main():
    return render_template('main.html')

@app.route("/viewdb")
def viewdb():
    rows = execute_query("""SELECT DISTINCT assetName FROM market_data ORDER BY assetName""")
    json_row=[]
    count=0
    for row in rows:
    	count=count+1
    	json_rower={'label':row}
    	json_row.append(json_rower)
    #app.logger.info('%d rows',count)
    print(count, file=sys.stdout)
    return json.dumps(json_row)
    # return '<br>'.join(str.strip(str(row),"(),") for row in rows)

@app.route('/asset')
def print_data():
    """Respond to a query of the format:
    myapp/?assetName=Apple Inc&start_time=2007-02-01&end_time=2012-02-01
    with assetName and start and end time specified in the query"""
    #start_time = start_time.time()
    cur = get_db().cursor()
    asset = request.args.get('assetName')

    #print minute_of_day
    asset_table = re.sub(r'\W+', '', asset)#New line
    asset_table = "Market_" + asset_table#New line
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')

    # print(asset)
    #query = """SELECT strftime('%d/%m/%Y',time), close, volume, (close+open)/2 AS Average
    #        FROM """ + asset_table + """ WHERE time>= """ + start_time + """ AND time<= """ + end_time
    # query = """SELECT strftime('%d/%m/%Y',time), close, volume, (close+open)/2 AS Average
    #         FROM """ + asset_table
    # # print(query)
    # result = execute_query(query)
    #         # """SELECT strftime('%d/%m/%Y',time), close, volume, (close+open)/2 AS Average
    #         # FROM ?
    #         #  WHERE time>=?
    #         #          AND time<=?""",
    #         query,
    #     (start_time, end_time)
    # )

    result = predict_stock(asset, start_time, end_time).values.tolist()

    # print("##################################################################################")
    # print(result)
    str_rows = [','.join(map(str, row)) for row in result]
    #query_time = time.time() - start_time
    #logging.info("executed query in %s" % query_time)
    cur.close()
    header = 'Date,Volume,Close,Open,Predicted,Expected,Average,\n'
    return header + '\n'.join(str_rows)


@app.route('/news')
def print_news_data():
    """Respond to a query of the format:
    myapp/?assetName=Apple Inc&start_time=2007-02-01&end_time=2012-02-01
    with assetName and start and end time specified in the query"""
    #start_time = start_time.time()
    curn = get_dbn().cursor()
    asset = request.args.get('assetName')
    asset_table = re.sub(r'\W+', '', asset)#New line
    asset_table = "News_" + asset_table#New line
    #print minute_of_day
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')

    #print(colour)
    result = execute_queryn(
            """SELECT strftime('%d/%m/%Y',time) AS date_time, headline, urgency, sentimentClass, relevance
            FROM  """ +  asset_table+ """
             WHERE time>=?
                     AND time<=?""",
            (start_time, end_time)
    )
    if result!=[]:
        colour,meancol=make_prediction(asset, start_time)
    else:
        colour=[]
        meancol=''
    #str_rows = [','.join(map(str, row)) for row in result]
    #query_time = time.time() - start_time
    #logging.info("executed query in %s" % query_time)

    json_row=[]
    count=0
    if result!=[]:
        json_rower={'date':'Date','label':'Headline','colour':'Color','predicted':'Prediction', 'urgency':'Urgency',
                    'sentimentClass':'SentimentClass', 'relevance':'Relevance'}
        json_row.append(json_rower)

    for row in result:
        json_rower={'date':row[0],'label':row[1],'colour':str(colour[count]),'predicted':str(meancol), 'urgency':str(row[2]),
                    'sentimentClass':str(row[3]), 'relevance':str(row[4])}
        json_row.append(json_rower)
        count=count+1

    #app.logger.info('%d rows',count)
    print(count, file=sys.stdout)
    curn.close()
    return json.dumps(json_row)
    # header = 'Date,Headline,\n'
    # return header + '\n'.join(str_rows)





















#,title='Home',user=user)
#The routes are the different URLs that the application implements.
 #In Flask, handlers for the application routes are written
 #as Python functions, called view functions.
 # View functions are mapped to one or more route URLs so that Flask
 #  knows what logic to execute when a client requests a given URL.
 #This view function is actually pretty simple, it just returns a
 #greeting as a string.
#The two strange @app.route lines above the function are
#decorators, a unique feature of the Python language.
#A common pattern with decorators is to use them to
#register functions as callbacks for certain events.
#In this case, the @app.route decorator creates an
#association between the URL given as an argument and
#the function.
#In this example there are two decorators, which
#associate the URLs / and /index to this function.
 # This means that when a web browser requests either
 # of these
 # two URLs, Flask is going to invoke this
 # function and pass the return value of it back to the browser as a response

