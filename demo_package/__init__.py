# Justin Or, Feb 2019

# Usage: python -m demo_package.task
# Make sure you are one folder outside the demo directory
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python packagedModel.py

import argparse
import time
import pandas as pd
import itertools
import datetime
import random
from numpy.random import choice
import tensorflow as tf

# 88888888ba   88888888ba     ,ad8888ba,           88  88888888888  ,ad8888ba,  888888888888  
# 88      "8b  88      "8b   d8"'    `"8b          88  88          d8"'    `"8b      88       
# 88      ,8P  88      ,8P  d8'        `8b         88  88         d8'                88       
# 88aaaaaa8P'  88aaaaaa8P'  88          88         88  88aaaaa    88                 88       
# 88""""""'    88""""88'    88          88         88  88"""""    88                 88       
# 88           88    `8b    Y8,        ,8P         88  88         Y8,                88       
# 88           88     `8b    Y8a.    .a8P  88,   ,d88  88          Y8a.    .a8P      88       
# 88           88      `8b    `"Y8888Y"'    "Y8888P"   88888888888  `"Y8888Y"'       88       

PROJECT_ID = "__sample_project_id"

# 888888888888    db         ad88888ba   88      a8P   
#      88        d88b       d8"     "8b  88    ,88'    
#      88       d8'`8b      Y8,          88  ,88"      
#      88      d8'  `8b     `Y8aaaaa,    88,d88'       
#      88     d8YaaaaY8b      `"""""8b,  8888"88,      
#      88    d8""""""""8b           `8b  88P   Y8b     
#      88   d8'        `8b  Y8a     a8P  88     "88,   
#      88  d8'          `8b  "Y88888P"   88       Y8b  

# from http://patorjk.com/software/taag
# use "Old Banner" or "Univers"

# ~~~~ PARSE ARGUMENTS ~~~~
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_paths", default='demo_package/output/trainPartitioned.csv', help='Where train data is stored.')
parser.add_argument("--eval_data_paths", default='demo_package/output/testPartitioned.csv', help="Where test data is stored.")
parser.add_argument("--store_data_path", default='demo_package/input/store.csv', help="Where the store.csv file is stored.")
parser.add_argument("--original_train_data_path", default='demo_package/input/train.csv', help="Where the original train.csv file is stored")
parser.add_argument("--output_dir", default=".\\model_check\\", help="Where the output will be stored.")
parser.add_argument("--train_steps", default=14000, type=int, help="Number of training steps.")
parser.add_argument("--batch_size", default=1000, type=int, help="Size of batches for training.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate for the optimizer.")
parser.add_argument("--job-dir", default=".\\model_check\\", help="Where the job data will be stored at.")
parser.add_argument("--storage_type", default="local", choices=["gcs", "bq", "local"])

args = parser.parse_args()

# ~~~~ CONSTANTS ~~~~
NUMITERATIONS = args.train_steps
BATCHSIZE = args.batch_size
LEARNINGRATE = args.learning_rate
STORAGE_TYPE = args.storage_type

JOBDIR = args.job_dir
# PATH = args.output_dir
# PATH = JOBDIR

# ~~~~ BQ ~~~~ (not used)
DATASET_ID = "__dataset" #"cbdsolutions_new_data_set"
TABLE_ID = "__table" # this should be where all the data is hosted at
TIME = int(round(time.time() * 1000))
NUM_PARTITIONS = 1 # choosing not to partitioned right now

# ~~~~ Cloud ML ~~~~
train_file = args.original_train_data_path
test_file = 'demo_package/input/test.csv'
store_file = args.store_data_path
output_file = 'demo_package/output/total_historical_data_set.csv'
output_train = args.train_data_paths
output_test = args.eval_data_paths

# ~~~~ Packaged Model ~~~~
unavailable_features = ['Store', 'Customers', 'PromoInterval', 'StateHoliday', 'DayOfWeek', 'Assortment', \
                        'StoreType','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Day', 'Month', 'Year']
label = ['Sales']
integer_features = ['CompetitionDistance', 'Year']
boolean_features = ['Open', 'Promo','SchoolHoliday', 'Promo2']
categorical_features = {"Assortment" : ["A", "B", "C"], \
                        "StateHoliday" : ["0","A", "B", "C"], \
                        "StoreType" : ["A", "B", "C", "D"] }
categorical_identity_features = {"DayOfWeek": [0,1,2,3,4,5,6],  #0 = mon, 1 = tue, ..., 6 = sun  
                                "Month": [0,1,2,3,4,5,6,7,8,9,10,11]}
bucket_categorical_features = {'WeekOfYear': [13,26,39]}

COLUMNS = integer_features + boolean_features + \
    sorted(list(categorical_features.keys())) + \
    sorted(list(categorical_identity_features.keys())) + \
    sorted(list(bucket_categorical_features.keys())) + label
# calculating defaults for the categorical features
FIELD_DEFAULTS = len(integer_features) * [[0]] + \
                len(boolean_features) * [[0]] + \
                [[categorical_features[x][0]] for x in sorted(list(categorical_features.keys()))] + \
                [[categorical_identity_features[x][0]] for x in sorted(list(categorical_identity_features.keys()))]+ \
                [[0] for x in sorted(list(bucket_categorical_features.keys()))] + \
                 len(label) * [[0.0]]

# Read in the store.csv, store results in a dictionary, with the key being the storeID and the item being the resulting features
# CompetitionDistance
# Promo2
# Assortment
# Storetype
with tf.gfile.Open(store_file, 'r') as open_store_input:
    store_df = pd.read_csv(open_store_input)
    store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median(), inplace = True)
    store_df.fillna(0,inplace = True)
    store_df = store_df.drop(columns = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 
                                        'Promo2SinceYear', 'PromoInterval'])
    store = store_df.set_index('Store').to_dict('index')
print(store[10])
# define a function that can read in a time, and give out the required fields
# Year
# DayOfWeek
# Month
# WeekOfYear
def parseTime(date_and_time):
    if not isinstance(date_and_time, datetime.date):
        print("{} is not a valid datetime object.".format(date_and_time))
        exit()
    results = {}
    results['Year'] = date_and_time.year
    results['DayOfWeek'] = date_and_time.weekday()
    results['Month'] = date_and_time.month-1
    results['WeekOfYear'] = date_and_time.isocalendar()[1]
    return results
# Year, Month, Date
print(parseTime(datetime.date(2015, 5, 5)))
with tf.gfile.Open(train_file, 'r') as open_train_input:
    train_df = pd.read_csv(open_train_input)
    total = train_df['Store'].count() + 1
    print("Total is {}".format(total))
    p_open = len(train_df[train_df["Open"] == 1])/total
    p_promo = len(train_df[train_df["Promo"] == 1])/total
    p_stateholiday_0 = len(train_df[train_df["StateHoliday"] == 0])/total
    p_stateholiday_a = len(train_df[train_df["StateHoliday"] == 'a'])/total
    p_stateholiday_b = len(train_df[train_df["StateHoliday"] == 'b'])/total
    p_stateholiday_c = len(train_df[train_df["StateHoliday"] == 'c'])/total
    p_schoolholiday = len(train_df[train_df["SchoolHoliday"] == 1])/total

def my_rand(i, w):
    normed = [elem/sum(w) for elem in w]
    return choice(i, p=normed)
# Usage: 
inputs = ['e', 'f', 'g', 'h']
weights = [10, 30, 50, 10]
# print(my_rand(inputs, weights))

# Given date, we can infer the values for Open, Promo and Stateholiday based on the history of the data we have.
# Given the storeID and date, we can work out if the store has a schoolHoliday at that date.
def infer_data(storeID, date):
    # we should have a calender if this were a real project
    # but for now, the stats for Open, Promo, StateHoliday and SchoolHoliday are determined by the probability
    open_val = my_rand([0, 1], [100*(1-p_open), 100*p_open])
    promo_val = my_rand([0, 1], [100*(1-p_promo), 100*p_promo])
    school_val = my_rand([0, 1], [100*(1-p_schoolholiday), 100*p_schoolholiday])
    state_val = my_rand([0, 'a', 'b', 'c'], [p_stateholiday_0, p_stateholiday_a, p_stateholiday_b, p_stateholiday_c])
    results = {}
    results["Open"] = int(open_val)
    results["Promo"] = int(promo_val)
    results["SchoolHoliday"] = int(school_val)
    results["StateHoliday"] = str(state_val)
    return results    

                                                                                      
# 88888888ba   88888888ba   88888888888  88888888ba,    88    ,ad8888ba,  888888888888  
# 88      "8b  88      "8b  88           88      `"8b   88   d8"'    `"8b      88       
# 88      ,8P  88      ,8P  88           88        `8b  88  d8'                88       
# 88aaaaaa8P'  88aaaaaa8P'  88aaaaa      88         88  88  88                 88       
# 88""""""'    88""""88'    88"""""      88         88  88  88                 88       
# 88           88    `8b    88           88         8P  88  Y8,                88       
# 88           88     `8b   88           88      .a8P   88   Y8a.    .a8P      88       
# 88           88      `8b  88888888888  88888888Y"'    88    `"Y8888Y"'       88       
                                                                                      
# From http://patorjk.com/software/taag

# batch job prediction
MODEL_ID = "__model"
VERSION_ID = "__version"

# Pub sub
TOPIC_NAME = "rossmann_real_time"

# Cloud ML
PROJECT_NAME = "projects/" + PROJECT_ID
MODEL_NAME = PROJECT_NAME + "/models/" + MODEL_ID
VERSION_NAME = MODEL_NAME + "/versions/" + VERSION_ID
BUCKET_NAME  = "__bucket_name"
SOURCE_FILE = "request.json"
LIVE_FILE = "live.csv"
DESTINATION_BLOB = "predictionOutputs/request.json"

# GCS Bucket (https://cloud.google.com/storage/docs/access-control/iam to control access to buckets)
INPUT_PATH = "__gs://path/to/request.json"
OUTPUT_DIR = "predictionOutputs/results"
OUTPUT_PATH = "gs://" + BUCKET_NAME + "/" + OUTPUT_DIR
OUTPUT_ID = "prediction.results"
OUTPUT_NAME = OUTPUT_DIR + "/" + OUTPUT_ID
# FINAL_OUTPUT = OUTPUT_DIR + "/" + OUTPUT_ID + ".output"
# for cmd:
# set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\json\json_key.json 
# for PowerShell:
# $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\json\json_key.json"
GOOGLE_APPLICATION_CREDENTIALS = "../__json_key.json"
# SET THE GOOGLE_APPLICATION_CREDENTIALS: https://cloud.google.com/docs/authentication/getting-started
