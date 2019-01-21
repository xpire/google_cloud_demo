#credits: https://www.kaggle.com/elenapetrova/time-series-analysis-and-forecasts-with-prophet

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import tensorflow as tf
from tensorflow import keras
import shutil, os

import warnings
warnings.filterwarnings("ignore")

#~~~~~~~~VARIABLES~~~~~~~~
NPREDICTORS = 27#len(features)
NOUTPUTS = 1#len(target)
NHIDDEN = 20
NUMITERATIONS = 500000
BATCHSIZE = 1000
HYPERPARAMETERTUNING = 10
                    
#~~~~~~~~Pre-processing~~~~~~~~
def preprocess():
    train_file = 'input/train.csv'
    test_file = 'input/test.csv'
    store_file = 'input/store.csv'
    output_file = 'output/new_rossmann_prediction.csv'

    train = pd.read_csv(train_file,
                        parse_dates= True,
                        low_memory= False,
                        index_col= 'Date')
    test = pd.read_csv(test_file)
    store = pd.read_csv(store_file,
                        low_memory= False)
    print("#~~~~~~~~Pre-processing~~~~~~~~")
    print("General shape:{}\n{}".format(train.shape, train.describe()))

    train['Year'] = train.index.year
    train['Month'] = train.index.month
    train['Day'] = train.index.day
    train['WeekofYear'] = train.index.weekofyear

    #adding new variable
    # train['SalesPerCustomer'] = train['Sales']/train['Customers']
    # print("Sales per Customer:\n",train['SalesPerCustomer'].describe())

    print("Date parsed correctly?\n", train.index)

    #Missing values, removes zero sales stores and closed stores
    train = train[(train["Open"] != 0) & (train["Sales"] != 0)]
    print("Train shape after removing zeroes:\n",train.shape)

    #Missing values in store
    print("Store before:\n{}".format(store.head()))
    print("CompetitionDistance NaNs:\n",store[pd.isnull(store.CompetitionDistance)].count())
    print("Promo2SinceWeek NaNs:\n",store[pd.isnull(store.Promo2SinceWeek)].count())
    print("Promo2SinceYear NaNs:\n",store[pd.isnull(store.Promo2SinceYear)].count())

    store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
    store.fillna(0,inplace = True)
    print("Store after filling in missing values:\n{}".format(store.head()))

    #Train set merging with additional store information
    #how = inner makes sure that only observations in the intersection of train and store sets are merged together
    train_store = pd.merge(train, store, how = 'inner', on = 'Store')
    print("Total:", train_store.shape)
    # print(train_store.head())

    #store types
    # print(train_store.groupby('StoreType')['Sales'].describe())
    # print(train_store.groupby('StoreType')['Customers','Sales','SalesPerCustomer'].sum())
    # onehot encoding of stores
    train_store['StoreTypeA'] = (train_store['StoreType'] == 'a')
    train_store['StoreTypeB'] =  (train_store['StoreType'] == 'b') 
    train_store['StoreTypeC'] =  (train_store['StoreType'] == 'c') 
    train_store['StoreTypeD'] =  (train_store['StoreType'] == 'd') 
    train_store = train_store.drop(columns = ['StoreType'])

    # onehot encoding of stores
    train_store['AssortA'] =  (train_store['Assortment'] == 'a') 
    train_store['AssortB'] =  (train_store['Assortment'] == 'b') 
    train_store['AssortC'] =  (train_store['Assortment'] == 'c') 
    train_store = train_store.drop(columns = ['Assortment'])

    #onehot encoding of holidays
    # train_store['StateHolidayA'] =  (train_store['StateHoliday'] == 'a') 
    # train_store['StateHolidayB'] =  (train_store['StateHoliday'] == 'b') 
    # train_store['StateHolidayC'] =  (train_store['StateHoliday'] == 'c') 
    train_store = train_store.drop(columns = ['StateHoliday'])
    #from testing, have no effect

    #onehot encoding of promoInterval 
    # train_store['promoJan'] = ("Jan" in train_store['PromoInterval']) 
    # train_store['promoFeb'] = ("Feb" in train_store['PromoInterval']) 
    # train_store['promoMar'] = ("Mar" in train_store['PromoInterval']) 
    # train_store['promoApr'] = ("Apr" in train_store['PromoInterval']) 
    # train_store['promoMay'] = ("May" in train_store['PromoInterval']) 
    # train_store['promoJun'] = ("Jun" in train_store['PromoInterval']) 
    # train_store['promoJul'] = ("Jul" in train_store['PromoInterval']) 
    # train_store['promoAug'] = ("Aug" in train_store['PromoInterval']) 
    # train_store['promoSep'] = ("Sep" in train_store['PromoInterval']) 
    # train_store['promoOct'] = ("Oct" in train_store['PromoInterval']) 
    # train_store['promoNov'] = ("Nov" in train_store['PromoInterval']) 
    # train_store['promoDec'] = ("Dec" in train_store['PromoInterval']) 
    train_store = train_store.drop(columns = ['PromoInterval'])
    #from testing, have no effect

    #removed outside information unavailable
    train_store = train_store.drop(columns = ['Customers'])

    #onehot encoding of DayOfWeek
    train_store['DayMon'] = (1 == train_store["DayOfWeek"]) 
    train_store['DayTue'] = (2 == train_store["DayOfWeek"]) 
    train_store['DayWed'] = (3 == train_store["DayOfWeek"]) 
    train_store['DayThu'] = (4 == train_store["DayOfWeek"]) 
    train_store['DayFri'] = (5 == train_store["DayOfWeek"]) 
    train_store['DaySat'] = (6 == train_store["DayOfWeek"]) 
    train_store['DaySun'] = (7 == train_store["DayOfWeek"]) 
    train_store = train_store.drop(columns = ['DayOfWeek'])

    # print(train['Store'].unique())
    train_store = train_store.drop(columns=['Store'])

    return train_store
    # print("Final: \n{}".format(train_store.head().to_string()))
    # train_store.to_csv(output_file, index = True)

#~~~~~~~~Loading~~~~~~~~
def loading(train_store):
    print("#~~~~~~~~Loading~~~~~~~~")
    #separate the X and Y components
    label = train_store['Sales'].values
    train_store = train_store.drop(columns = ['Sales'])
    #convert pandas dataframe to tensor
    print(train_store.dtypes)
    target = ['Sales']
    features = [#'Store', #'Sales', 
    #'Customers', 
    'Open', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekofYear', #OG
    #'SalesPerCustomer', 
    'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', #floats except Promo2
    'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 
    'AssortA', 'AssortB', 'AssortC', 
    #'StateHolidayA', 'StateHolidayB', 'StateHolidayC',
    #'promoJan', 'promoFeb', 'promoMar', 'promoApr', 'promoMay', 'promoJun', 'promoJul', 'promoAug', 'promoSep', 'promoOct', 'promoNov', 'promoDec',
    'DayMon', 'DayTue', 'DayWed', 'DayThu', 'DayFri', 'DaySat', 'DaySun'] #Booleans
    features_dict = {features[i] : i for i in range(len(features))}
    print(features_dict)
    train_store = train_store.values

    print(train_store)

    #take 80% for training, 20% for validation
    total = len(train_store)
    idx = np.random.permutation(total)
    x,y = train_store[idx], label[idx]
    print("x:\n{}".format(x))
    print("y:\n{}".format(y))

    train_x = x[:int(total*0.8)]
    train_y = y[:int(total*0.8)]
    test_x = x[int(total*0.8):]
    test_y = y[int(total*0.8):]

    print("Train_x:\n{}".format(train_x))
    print("Train_y:\n{}".format(train_y))
    print("Test_x:\n{}".format(test_x))
    print("Test_y:\n{}".format(test_y))

    print("Lengths: train ({}, {}), test ({}, {})".format(len(train_x), len(train_y), len(test_x), len(test_y)))
    avg = np.mean(train_y)
    rmse = np.sqrt(np.mean((test_y - avg)**2))
    print("Baseline to beat w/ avg -> {}: {}\n".format(avg, rmse))
    return train_x, train_y, test_x, test_y, target, features

def getTrainBatch(train_x, train_y, features, target, BATCHSIZE):
    labels = []
    arr = np.zeroes([BATCHSIZE, features])
    for i in range(BATCHSIZE):
        arr[i] = train_x[i]
        label[i] = train_y[i]
    


#~~~~~~~~Training~~~~~~~~
def train(train_x, train_y, test_x, test_y, features, target):
    print("#~~~~~~~~Training~~~~~~~~")
    with tf.Session() as sess:
        feature_data = tf.placeholder("float32", [None, NPREDICTORS])
        target_data = tf.placeholder("float32", [None, NOUTPUTS])
        w1 = tf.Variable(tf.truncated_normal([NPREDICTORS, NOUTPUTS], stddev= 0.0))
        b1 = tf.Variable(tf.ones([NOUTPUTS]))
        logits = tf.matmul(feature_data, w1) + b1
        # preds = tf.nn.relu(logits)
        preds = logits #linear regression requires no activation
        cost = tf.losses.mean_squared_error(
            labels = target_data,
            predictions = preds           
        )
        accuracy = tf.sqrt(cost)
        training_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        init = tf.initialize_all_variables()
        sess.run(init)

        saver = tf.train.Saver({'weights' : w1, 'bias' : b1}, max_to_keep=None)
        batch_data = train_x[:BATCHSIZE]
        batch_labels = train_y[:BATCHSIZE]
        print(batch_data.shape)
        print(batch_labels.shape) 
        train_total = len(train_x)
        idx = np.random.permutation(train_total)
        train_x,train_y = train_x[idx], train_y[idx]

        current_lowest_test_error = 100000
        current_lowest_train_error = 100000
        iter_lowest_test_error = 0
        current_filename = ""
        allowed_failures = 3

        for iter in range(0, NUMITERATIONS):
            sess.run(
                training_step,
                feed_dict= {
                    feature_data : batch_data,
                    target_data : batch_labels.reshape(BATCHSIZE, NOUTPUTS)
                    # feature_data : train_x,
                    # target_data : train_y.reshape(len(train_x), NOUTPUTS)
                }
            )
            if (iter % 5000 == 0):
                cost_value, acc_value = sess.run(
                    [cost, accuracy],
                    feed_dict= {
                        feature_data : batch_data,
                        target_data : batch_labels.reshape(BATCHSIZE, NOUTPUTS)
                        # feature_data : train_x,
                        # target_data : train_y.reshape(len(train_x), NOUTPUTS)
                    })
                print("Iteration: ", iter)
                print("loss", cost_value)
                print("acc", acc_value)
            
            if iter % 10000 == 0:
                # test(test_x, test_y, NUMITERATIONS, saver, sess)
                filename = saver.save(sess, "checkpoints/trained_model.ckpt", global_step=iter)
                print("Model written to {}".format(filename))
                test_error = np.sqrt(cost.eval(feed_dict= {
                        feature_data : test_x,
                        target_data : test_y.reshape(len(test_x), NOUTPUTS)
                }))
                print("Test error: {}".format(test_error))

                if test_error < current_lowest_test_error:
                    current_lowest_test_error = test_error
                    iter_lowest_test_error = iter
                    current_lowest_train_error = acc_value
                    current_filename = filename
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~best test error yet: {} from {}".format(current_lowest_test_error, iter_lowest_test_error))
                else:
                    allowed_failures-=1
                    if allowed_failures == 0:
                        #save this model and restart
                        directory = "model_candidates/t" + str(int(current_lowest_test_error)) + "(" + str(int(acc_value)) + ")"
                        if not os.path.exists(directory + "/"):
                            os.mkdir(directory + "/")
                        shutil.copy(filename + '.index', directory)
                        shutil.copy(filename + '.meta', directory)
                        shutil.copy(filename + '.data-00000-of-00001', directory)
                        print("saved in {}".format(directory))
                        break
                    



        filename = saver.save(sess, "checkpoints/trained_model.ckpt", global_step=NUMITERATIONS)
        print("Model written to {}".format(filename))
        print("Test error: {}".format(
            np.sqrt(cost.eval(feed_dict= {
                feature_data : test_x,
                target_data : test_y.reshape(len(test_x), NOUTPUTS)
            }))
        ))
'''
features = ['Store', #'Sales', 
    #'Customers', 
    'Open', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekofYear', #OG
    #'SalesPerCustomer', 
    'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', #floats except Promo2
    'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 
    'AssortA', 'AssortB', 'AssortC', 
    #'StateHolidayA', 'StateHolidayB', 'StateHolidayC',
    #'promoJan', 'promoFeb', 'promoMar', 'promoApr', 'promoMay', 'promoJun', 'promoJul', 'promoAug', 'promoSep', 'promoOct', 'promoNov', 'promoDec',
    'DayMon', 'DayTue', 'DayWed', 'DayThu', 'DayFri', 'DaySat', 'DaySun'] #Booleans
'''
# def prediction(store, open, promo, date):
    

def eval(test_x, test_y, features, target):
    #from train
    feature_data = tf.placeholder("float32", [None, NPREDICTORS])
    target_data = tf.placeholder("float32", [None, NOUTPUTS])
    w1 = tf.Variable(tf.truncated_normal([NPREDICTORS, NOUTPUTS], stddev= 0.0))
    b1 = tf.Variable(tf.ones([NOUTPUTS]))
    logits = tf.matmul(feature_data, w1) + b1
    # preds = tf.nn.relu(logits)
    preds = logits #linear regression requires no activation
    cost = tf.losses.mean_squared_error(
        labels = target_data,
        predictions = preds           
    )
    accuracy = tf.sqrt(cost)
    training_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    with tf.Session() as sess:
        # last_check = tf.train.latest_checkpoint('./checkpoints')
        # saver = tf.train.import_meta_graph(last_check + ".meta")
        # saver.restore(sess, last_check)
        # filename = "checkpoints/trained_model.ckpt-110000"
        filename = "model_candidates/t2883(2393)/trained_model.ckpt-270000"
        saver = tf.train.Saver({'weights' : w1, 'bias' : b1})
        saver.restore(sess, filename)
        print("Model restored.")
        # Check the values of the variables
        print("w1 : %s" % w1.eval())
        print("b1 : %s" % b1.eval())
        """
        output:
        Model restored.
        w1 : [[-3.0153331e-01]
        [ 3.1417143e+01]
        [ 2.4768223e+03]
        [ 6.3471960e+02]
        [ 4.0008183e+00]
        [-1.3948508e+02]
        [ 3.8556293e+01]
        [ 2.1763359e+01]
        [-5.6020424e-02]
        [-6.5704239e+01]
        [ 4.6448222e-01]
        [-8.4330711e+01]
        [ 9.0178704e-01]
        [-2.9978019e-01]
        [-2.6656587e+03]
        [ 2.6947957e+03]
        [-2.3623589e+03]
        [-2.1456587e+03]
        [-2.2471423e+02]
        [ 0.0000000e+00]
        [ 2.8142548e+02]
        [ 5.2933624e+02]
        [ 2.2284938e+02]
        [-1.9887314e+01]
        [-8.3266464e+01]
        [-2.6087637e+02]
        [ 9.8838824e+02]
        [-1.0463618e+03]]
        b1 : [32.417385]
        test error: 2866.5537109375
        """
        print("test error: {}".format(
            np.sqrt(cost.eval(feed_dict= {
                feature_data : test_x,
                target_data : test_y.reshape(len(test_x), NOUTPUTS)
            }))
        ))

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train", "eval"])
    
    args = parser.parse_args()

    if (args.mode == "train"):
        for i in range(HYPERPARAMETERTUNING):
            train_x, train_y, test_x, test_y, target, features = loading(preprocess())
            train(train_x, train_y, test_x, test_y, features, target)
    elif (args.mode == "eval"):
        print("Evaluation run")
        _, _, test_x, test_y, target, features = loading(preprocess())
        eval(test_x, test_y, features, target)
    elif (args.mode == "preprocess"):
        loading(preprocess())


# train_x, train_y, test_x, test_y, target, features = loading(preprocess())
# train(train_x, train_y, test_x, test_y, features, target)