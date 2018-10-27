from flask import Flask
from flask import request

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
import tensorflow as tf

app = Flask(__name__)
#-------------------------------------------
# POST Request to /queryModel:
#   {
#       'storeName':<STORE_NAME>,
#       'storeOpen':<STORE_OPEN>,
#       'storePromo':<STORE_PROMO>,
#       'date':<DATE>,
#   }
#-------------------------------------------
@app.route('/querymodel', methods=['POST'])
def parseRequest():
    inputs = request.json
    storeName = inputs['storeName']
    storeOpen = inputs['storeOpen']
    promo = inputs['storePromo']
    date = inputs['date']
    salesPrediction = model(storeName, storeOpen, promo, date)
    return salesPrediction

#--------------------------------------------
# Justin: Model function goes here! Import
# all necessary functions from model.py if 
# needed :)
#--------------------------------------------
def model(storeName, storeOpen, promo, date):
    #~~~~~~~~VARIABLES~~~~~~~~
    NPREDICTORS = 28#len(features)
    NOUTPUTS = 1#len(target)
    with tf.Session() as sess:
        feature_data = tf.placeholder("float32", [None, NPREDICTORS])
        target_data = tf.placeholder("float32", [None, NOUTPUTS])
        w1 = tf.Variable(tf.truncated_normal([NPREDICTORS, NOUTPUTS], stddev= 0.0))
        b1 = tf.Variable(tf.ones([NOUTPUTS]))
        logits = tf.matmul(feature_data, w1) + b1
        # cost = tf.losses.mean_squared_error(
        #     labels = target_data,
        #     predictions = logits           
        # )
        #change this line to chose a different model
        filename = "model_candidates/t2883(2393)/trained_model.ckpt-270000"
        saver = tf.train.Saver({'weights' : w1, 'bias' : b1})
        saver.restore(sess, filename)
        print("Model restored.")
        # Check the values of the variables
        print("w1 : %s" % w1.eval())
        print("b1 : %s" % b1.eval())
        features = ['Store', 'Open', 'Promo', 
        'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekofYear',
        'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 
        'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 
        'AssortA', 'AssortB', 'AssortC', 
        'DayMon', 'DayTue', 'DayWed', 'DayThu', 'DayFri', 'DaySat', 'DaySun']
        input = np.zeros([28]) 
        input[0] = storeName
        input[1] = storeOpen
        input[2] = promo
        # parse dates
        df = pd.DataFrame
        df['Date'] = pd.date_range(start=date, end=date)
        cal = calender()
        holidays = cal.holidays(start=date, end=date)
        # SchoolHoliday
        input[3] = 1 if (df['Date'].isin(holidays)) else 0
        # Year
        # Month
        # Day
        # WeekOfYear
        # CompetitionDistance
        # CompetitionOpenSinceMonth
        # CompetitionOpenSinceYear
        # Promo2
        # Promo2SinceWeek
        # Promo2SinceYear
        # onehot encoding of Store Type
        # onehot encoding of AssortmentType
        # onehot encoding of Day Type
        predicted = sess.run(logits, feed_dict = {
            feature_data : input.values
        })
        # print("test error: {}".format(
        #     np.sqrt(cost.eval(feed_dict= {
        #         feature_data : test_x,
        #         target_data : test_y.reshape(len(test_x), NOUTPUTS)
        #     }))
        # ))
    return predicted
    