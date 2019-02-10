# Justin Or, Feb 2019

# Usage: python -m demo_package.task
# Make sure you are one folder outside the demo directory
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python packagedModel.py

import argparse
import time

# ~~~~ PARSE ARGUMENTS ~~~~
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_paths", default='demo_package/output/trainPartitioned.csv', help='Where train data is stored.')
parser.add_argument("--eval_data_paths", default='demo_package/output/testPartitioned.csv', help="Where test data is stored.")
parser.add_argument("--output_dir", default=".\\model_check\\", help="Where the output will be stored.")
parser.add_argument("--train_steps", default=12000, type=int, help="Number of training steps.")
parser.add_argument("--batch_size", default=1000, type=int, help="Size of batches for training.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate for the optimizer.")
parser.add_argument("--job-dir", default=".\\model_check\\", help="Where the job data will be stored at.")
parser.add_argument("--storage_type", default="gcs", choices=["gcs", "bq", "local"])

args = parser.parse_args()
# ~~~~ CONSTANTS ~~~~
# NPREDICTORS = 27 #len(features)
# NOUTPUTS = 1 #len(target)
# NHIDDEN = 20
NUMITERATIONS = args.train_steps
BATCHSIZE = args.batch_size
LEARNINGRATE = args.learning_rate
STORAGE_TYPE = args.storage_type

# Not used
# HYPERPARAMETERTUNING = 10

JOBDIR = args.job_dir
# PATH = args.output_dir
PATH = JOBDIR

PROJECT_ID = "rich-principle-225813"
DATASET_ID = "preprcessed"
TIME = int(round(time.time() * 1000))
NUM_PARTITIONS = 1 # choosing not to partitioned right now

train_file = 'demo_package/input/train.csv'
test_file = 'demo_package/input/test.csv'
store_file = 'demo_package/input/store.csv'
output_file = 'demo_package/output/total_historical_data_set.csv'
output_train = args.train_data_paths
output_test = args.eval_data_paths

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
