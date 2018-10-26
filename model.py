#credits: https://www.kaggle.com/elenapetrova/time-series-analysis-and-forecasts-with-prophet

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

#~~~~~~~~VARIABLES~~~~~~~~
NPREDICTORS = 45#len(features)
NOUTPUTS = 1#len(target)
NHIDDEN = 20
NUMITERATIONS = 110000
BATCHSIZE = 240

                    
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
    train['SalesPerCustomer'] = train['Sales']/train['Customers']
    print("Sales per Customer:\n",train['SalesPerCustomer'].describe())

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
    train_store['StateHolidayA'] =  (train_store['StateHoliday'] == 'a') 
    train_store['StateHolidayB'] =  (train_store['StateHoliday'] == 'b') 
    train_store['StateHolidayC'] =  (train_store['StateHoliday'] == 'c') 
    train_store = train_store.drop(columns = ['StateHoliday'])

    #onehot encoding of promoInterval 
    train_store['promoJan'] = ("Jan" in train_store['PromoInterval']) 
    train_store['promoFeb'] = ("Feb" in train_store['PromoInterval']) 
    train_store['promoMar'] = ("Mar" in train_store['PromoInterval']) 
    train_store['promoApr'] = ("Apr" in train_store['PromoInterval']) 
    train_store['promoMay'] = ("May" in train_store['PromoInterval']) 
    train_store['promoJun'] = ("Jun" in train_store['PromoInterval']) 
    train_store['promoJul'] = ("Jul" in train_store['PromoInterval']) 
    train_store['promoAug'] = ("Aug" in train_store['PromoInterval']) 
    train_store['promoSep'] = ("Sep" in train_store['PromoInterval']) 
    train_store['promoOct'] = ("Oct" in train_store['PromoInterval']) 
    train_store['promoNov'] = ("Nov" in train_store['PromoInterval']) 
    train_store['promoDec'] = ("Dec" in train_store['PromoInterval']) 
    train_store = train_store.drop(columns = ['PromoInterval'])

    #onehot encoding of DayOfWeek
    train_store['DayMon'] = (1 == train_store["DayOfWeek"]) 
    train_store['DayTue'] = (2 == train_store["DayOfWeek"]) 
    train_store['DayWed'] = (3 == train_store["DayOfWeek"]) 
    train_store['DayThu'] = (4 == train_store["DayOfWeek"]) 
    train_store['DayFri'] = (5 == train_store["DayOfWeek"]) 
    train_store['DaySat'] = (6 == train_store["DayOfWeek"]) 
    train_store['DaySun'] = (7 == train_store["DayOfWeek"]) 
    train_store = train_store.drop(columns = ['DayOfWeek'])
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
    features = ['Store', #'Sales', 
    'Customers', 'Open', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekofYear', #OG
    'SalesPerCustomer', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', #floats except Promo2
    'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 
    'AssortA', 'AssortB', 'AssortC', 
    'StateHolidayA', 'StateHolidayB', 'StateHolidayC',
    'promoJan', 'promoFeb', 'promoMar', 'promoApr', 'promoMay', 'promoJun', 'promoJul', 'promoAug', 'promoSep', 'promoOct', 'promoNov', 'promoDec',
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

        saver = tf.train.Saver({'weights' : w1, 'bias' : b1})
        batch_data = train_x[:BATCHSIZE]
        batch_labels = train_y[:BATCHSIZE]
        print(batch_data.shape)
        print(batch_labels.shape) 
        train_total = len(train_x)
        idx = np.random.permutation(train_total)
        train_x,train_y = train_x[idx], train_y[idx]
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
            if (iter % 500 == 0):
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
            
            if iter % 1000 == 0:
                # test(test_x, test_y, NUMITERATIONS, saver, sess)
                filename = saver.save(sess, "checkpoints/trained_model.ckpt", global_step=iter)
                print("Model written to {}".format(filename))
                print("Test error: {}".format(
                    np.sqrt(cost.eval(feed_dict= {
                        feature_data : test_x,
                        target_data : test_y.reshape(len(test_x), NOUTPUTS)
                    }))
                ))

        filename = saver.save(sess, "checkpoints/trained_model.ckpt", global_step=NUMITERATIONS)
        print("Model written to {}".format(filename))
        print("Test error: {}".format(
            np.sqrt(cost.eval(feed_dict= {
                feature_data : test_x,
                target_data : test_y.reshape(len(test_x), NOUTPUTS)
            }))
        ))

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
        filename = "checkpoints/trained_model.ckpt-110000"
        saver = tf.train.Saver({'weights' : w1, 'bias' : b1})
        saver.restore(sess, filename)
        print("Model restored.")
        # Check the values of the variables
        print("w1 : %s" % w1.eval())
        print("b1 : %s" % b1.eval())
        # cost_val = sess.run(
        #     [cost],
        #     feed_dict={
        #         feature_data : test_x,
        #         target_data : test_y.reshape(len(test_x), NOUTPUTS)
        #     }
        # )
        print("test error: {}".format(
            np.sqrt(cost.eval(feed_dict= {
                feature_data : test_x,
                target_data : test_y.reshape(len(test_x), NOUTPUTS)
            }))
        ))



    # sess = tf.InteractiveSession()
    # last_check = tf.train.latest_checkpoint('./checkpoints')
    # saver = tf.train.import_meta_graph(last_check + ".meta")
    # saver.restore(sess, last_check)
    # graph = tf.get_default_graph()

    # loss = graph.get_tensor_by_name('loss:0')
    # accuracy = graph.get_tensor_by_name('accuracy:0')

    # input_data = graph.get_tensor_by_name('input_data:0')
    # labels = graph.get_tensor_by_name('labels:0')

    # num_batches = num_samples // BATCH_SIZE
    # label_list = [[1, 0]] * (num_samples // 2)  # pos always first, neg always second
    # label_list.extend([[0, 1]] * (num_samples // 2))
    # assert (len(label_list) == num_samples)
    # total_acc = 0
    # for i in range(num_batches):
    #     sample_index = i * BATCH_SIZE
    #     batch = test_data[sample_index:sample_index + BATCH_SIZE]
    #     batch_labels = label_list[sample_index:sample_index + BATCH_SIZE]
    #     lossV, accuracyV = sess.run([loss, accuracy], {input_data: batch,
    #                                                    labels: batch_labels})
    #     total_acc += accuracyV
    #     print("Accuracy %s, Loss: %s" % (accuracyV, lossV))
    # print('-' * 40)
    # print("FINAL ACC:", total_acc / num_batches)

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train", "eval"])
    
    args = parser.parse_args()

    if (args.mode == "train"):
        train_x, train_y, test_x, test_y, target, features = loading(preprocess())
        train(train_x, train_y, test_x, test_y, features, target)
    elif (args.mode == "eval"):
        print("Evaluation run")
        _, _, test_x, test_y, target, features = loading(preprocess())
        eval(test_x, test_y, features, target)
    else:
        print("Usage: python3 model.py train/eval")

# train_x, train_y, test_x, test_y, target, features = loading(preprocess())
# train(train_x, train_y, test_x, test_y, features, target)