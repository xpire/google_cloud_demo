# Justin Or, Jan 2019

# Packaged Estimator Model for use on Google CloudML
# Contains all the model logic of the package.

# Constants will go in the __init__.py file

# Preprocessing will be made into a dataflow apache beam pipeline

# Defines the input function and the feature column functions 
# which will be called within the estimator to build the model.
# The input function takes in data from tf.Dataset and encorporates
# the shuffling and subsetting of data.
# The feature column function builds the feature columns based on
# their type (boolean, integer, float, etc) 

# Usage: python -m demo_package.task (from outside this directory)
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python packagedModel.py
from . import * 
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
# ~~~~ PREPROCESS HELPER FUNCTIONS ~~~~

def preprocess(data, store):
    # set dates
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['WeekOfYear'] = data.index.weekofyear
    # Missing values, removes zero sales stores and closed stores
    data = data[(data["Open"] != 0) & (data["Sales"] != 0)]
    # Missing values in store
    store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
    store.fillna(0,inplace = True)
    # Merging the two datasets together
    data_store = pd.merge(data, store, how = 'inner', on = 'Store')
    # Change date from [1,7] to [0,6] for efficient reading into feature column.
    data_store["DayOfWeek"] = data_store["DayOfWeek"].apply(lambda x: x - 1)
    # make DayOfWeek a string for use with categorical identity feature column
    # Removal of information with no effect
    data_store = data_store.drop(columns = ['Day'])
    data_store = data_store.drop(columns = ['PromoInterval'])
    data_store = data_store.drop(columns = ['Customers'])
    data_store = data_store.drop(columns=['Store'])
    data_store = data_store.drop(columns=['CompetitionOpenSinceMonth'])
    data_store = data_store.drop(columns=['CompetitionOpenSinceYear'])
    data_store = data_store.drop(columns=['Promo2SinceWeek'])
    data_store = data_store.drop(columns=['Promo2SinceYear'])
    # sort columns so that it matches the order specified in __init__.py
    data_store = data_store[COLUMNS]

    print("columns of data_store: {}".format(list(data_store)))
    print(data_store.head())

    #take 80% for training, 20% for validation
    total = len(data_store)
    # idx = np.random.permutation(total)
    data_store = data_store.sample(frac=1)
    train = data_store.iloc[:, :int(total*0.8)]
    test = data_store.iloc[:, int(total*0.8):]
    train.to_csv(output_train)
    test.to_csv(output_test)
    data_store.to_csv(output_file, index=True)
    return data_store


# ~~~~ INPUT FUNCTIONS ~~~~

def input_data(data_dir, store_dir):
    data =  pd.read_csv(data_dir,
                        parse_dates= True,
                        low_memory= False,
                        index_col= 'Date')
    store = pd.read_csv(store_dir,
                        low_memory= False)
    return data, store

def input_evaluation_set(stage):
    if not isinstance(stage, int):
        exit()

    def _parse_line(line):
        # Decode line into fields
        fields = tf.decode_csv(line, record_defaults=FIELD_DEFAULTS)
        # Pack results into dict
        features = dict(zip(COLUMNS, fields))
        # Separate label from features
        separatedLabel = features.pop('Sales')
        return features, separatedLabel
    
    if stage <= 0:
        print("~~~~STAGE 0~~~~")
        eval, store = input_data(train_file, store_file)
    if stage <= 1:
        print("~~~~STAGE 1~~~~")
        data_store = preprocess(eval, store)
    #separate the X and Y components
    # label = data_store['Sales'].values
    # data_store = data_store.drop(columns = ['Sales'])
    # Convert to a tensorflow Dataset
    if stage <= 2:
        print("~~~~STAGE 2~~~~")
        ds = tf.data.TextLineDataset(output_train).skip(1)
    if stage <= 3:
        print("~~~~STAGE 3~~~~")
        print(ds)
        print(FIELD_DEFAULTS)
        ds = ds.map(_parse_line)
    print("ds: {}".format(ds))
    ds = ds.shuffle(1000).repeat().batch(BATCHSIZE)
    # data_set = tf.data.Dataset.from_tensor_slices(
    #     (tf.cast(data_store[integer].values, tf.float))
    # )
    return ds

# ~~~~ FEATURE COLUMN FUNCTIONS ~~~~

def build_model_columns():
    # Builds set of feature columns
    features = []
    # integer numerical columns
    for col in integer_features:
        features.append(tf.feature_column.numeric_column(key=col))
    # boolean categorical columns
    for col in boolean_features:
        features.append(tf.feature_column.categorical_column_with_identity(key=col, num_buckets=2)) # [0, 1]
    # integer categorical columns (ranging in integers)
    for key_name, item in categorical_identity_features.items():
        # print("for {}, len {}".format(key_name, item))
        features.append(tf.feature_column.categorical_column_with_identity(
            key=key_name,
            num_buckets=len(item) #[0,7) = [Mon, Tue, Wed, ..., Sun]
        ))
    # categorical columns with vocabulary
    for key_name, item in categorical_features.items():
        features.append(tf.feature_column.categorical_column_with_vocabulary_list(
            key=key_name,
            vocabulary_list=item
        ))
    return features

"""
Old model.py code:
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
        
        # output:
        # Model restored.
        # w1 : [[-3.0153331e-01]
        # [ 3.1417143e+01]
        # [ 2.4768223e+03]
        # [ 6.3471960e+02]
        # [ 4.0008183e+00]
        # [-1.3948508e+02]
        # [ 3.8556293e+01]
        # [ 2.1763359e+01]
        # [-5.6020424e-02]
        # [-6.5704239e+01]
        # [ 4.6448222e-01]
        # [-8.4330711e+01]
        # [ 9.0178704e-01]
        # [-2.9978019e-01]
        # [-2.6656587e+03]
        # [ 2.6947957e+03]
        # [-2.3623589e+03]
        # [-2.1456587e+03]
        # [-2.2471423e+02]
        # [ 0.0000000e+00]
        # [ 2.8142548e+02]
        # [ 5.2933624e+02]
        # [ 2.2284938e+02]
        # [-1.9887314e+01]
        # [-8.3266464e+01]
        # [-2.6087637e+02]
        # [ 9.8838824e+02]
        # [-1.0463618e+03]]
        # b1 : [32.417385]
        # test error: 2866.5537109375
        
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
"""