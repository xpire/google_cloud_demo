# Justin Or, Jan 2019

# Packaged Estimator Model for use on Google CloudML
# Contains all the model logic of the package.

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
# from tf.contrib.cloud import BigQueryReader
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
    data_store["DayOfWeek"] = data_store["DayOfWeek"].apply(lambda x: int(x - 1))
    data_store["Month"] = data_store["Month"].apply(lambda x: int(x - 1))
    # Removal of information with no effect
    data_store = data_store.drop(columns = ['Day'])
    data_store = data_store.drop(columns = ['PromoInterval'])
    data_store = data_store.drop(columns = ['Customers'])
    data_store = data_store.drop(columns = ['Store'])
    data_store = data_store.drop(columns = ['CompetitionOpenSinceMonth'])
    data_store = data_store.drop(columns = ['CompetitionOpenSinceYear'])
    data_store = data_store.drop(columns = ['Promo2SinceWeek'])
    data_store = data_store.drop(columns = ['Promo2SinceYear'])
    # sort columns so that it matches the order specified in __init__.py
    data_store = data_store[COLUMNS]
    # assert the correct data types are applied in each column
    for key in integer_features + boolean_features + list(categorical_identity_features.keys()) + list(bucket_categorical_features.keys()):
        data_store[key] = data_store[key].apply(lambda x: int(x))
    for key in list(categorical_features.keys()):
        data_store[key] = data_store[key].apply(lambda x: str(x))

    print("columns of data_store: {}".format(list(data_store)))
    print(data_store.head())

    #take 80% for training, 20% for validation
    total = len(data_store)
    data_store.to_csv(output_file)
    # idx = np.random.permutation(total)
    data_store = data_store.sample(frac=1)
    train = data_store.iloc[:int(total*0.8), :]
    test = data_store.iloc[int(total*0.8):, :]
    print("Train")
    print(train.head())
    print("Test")
    print(test.head())
    # Calculating baseline to beat
    mean_sales = test["Sales"].mean()
    mse = test["Sales"].apply(lambda x: np.square(x-mean_sales)).mean()
    print("The MSE to beat is {}".format(mse))
    
    train.to_csv(output_train, index=False)
    test.to_csv(output_test, index=False)
    # data_store.to_csv(output_file)
    # return data_store


# ~~~~ INPUT FUNCTIONS ~~~~

def input_data(data_dir, store_dir):
    data =  pd.read_csv(data_dir,
                        parse_dates= True,
                        low_memory= False,
                        index_col= 'Date')
    store = pd.read_csv(store_dir,
                        low_memory= False)
    print(store.head())
    print(data.head())
    return data, store

def get_data(csv_file, preprocess_data):
    if preprocess_data:
        print("Preprocess 1. Reading in data.")
        eval, store = input_data(train_file, store_file)
        print("Preprocess 2. Preproccessing data.")
        preprocess(eval, store)
    
    if STORAGE_TYPE == "local":
        with open(csv_file, 'r') as open_input:
            dataframe = pd.read_csv(open_input)
            print(dataframe.head())
    elif STORAGE_TYPE == "gcs":
        # for use with cloud storage
        with tf.gfile.Open(csv_file, 'r') as open_input:
            dataframe = pd.read_csv(open_input)
            print(dataframe.head())
    elif STORAGE_TYPE == "bq":
        # if "train" in csv_file:
        #     TABLE_ID = "train"
        # elif "test" in csv_file:
        #     TABLE_ID = "test"
        # else:
        #     print("csv_file ({}) does not contain the word train or test, so we cannot infer whether this is the train or test dataset.".format(csv_file))
        #     exit(1)
        # reader = BigQueryReader(
        #     project_id=PROJECT_ID,
        #     dataset_id=DATASET_ID,
        #     table_id=TABLE_ID,
        #     timestamp_millis=TIME,
        #     num_partitions=NUM_PARTITIONS,
        #     features=build_model_columns()
        # )
        exit(1)
    return dataframe

def input_set(preprocess_data, csv_file=output_file):

    def _parse_line(line):
        # TODO: json input function
        # https://www.tensorflow.org/api_docs/python/tf/io/decode_csv
        # Consider changing decode csv to decode json example and then parse tensor:
        # https://www.tensorflow.org/api_docs/python/tf/io/decode_json_example
        # https://www.tensorflow.org/api_docs/python/tf/io/parse_tensor
        # Decode line into fields
        fields = tf.decode_csv(line, record_defaults=FIELD_DEFAULTS)
        # Pack results into dict
        features = dict(zip(COLUMNS, fields))
        # Separate label from features
        separatedLabel = features.pop('Sales')
        return features, separatedLabel
    
    get_data(csv_file, preprocess_data)
    ds = tf.data.TextLineDataset(csv_file).skip(1)
    print("Input 1: Parsing lines.")
    print(ds)
    print(FIELD_DEFAULTS)
    ds = ds.map(_parse_line)
    print("ds: {}".format(ds))
    ds = ds.shuffle(1000).repeat().batch(BATCHSIZE)
    return ds

def input_train_set():
    if STORAGE_TYPE == "local":
        return input_set(True, output_train)
    else:
        return input_set(False, output_train)

def input_eval_set():
    return input_set(False, output_test)

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
    for key_name, item in bucket_categorical_features.items():
        features.append(tf.feature_column.bucketized_column(
            source_column=tf.feature_column.numeric_column(key_name),
            boundaries=item
        ))

    # define feature engineering here
    # https://cloud.google.com/ml-engine/docs/tensorflow/data-prep#engineer_the_data_features
    
    return features