# Justin Or, Feb 2019

# Usage: python -m demo.task
# Make sure you are one folder outside the demo directory
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python packagedModel.py

# ~~~~ CONSTANTS ~~~~
NPREDICTORS = 27 #len(features)
NOUTPUTS = 1 #len(target)
NHIDDEN = 20
NUMITERATIONS = 500000
BATCHSIZE = 1000
HYPERPARAMETERTUNING = 10

train_file = 'demo_package/input/train.csv'
test_file = 'demo_package/input/test.csv'
store_file = 'demo_package/input/store.csv'
output_file = 'demo_package/output/new_rossmann_prediction.csv'
output_train = 'demo_package/output/trainPartitioned.csv'
output_test = 'demo_package/output/testPartitioned.csv'

unavailable_features = ['Store', 'Customers', 'PromoInterval', 'StateHoliday', 'DayOfWeek', 'Assortment', \
                        'StoreType','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Day', 'Month', 'Year']
label = ['Sales']
integer_features = ['CompetitionDistance', 'Year']
boolean_features = ['Open', 'Promo','SchoolHoliday', 'Promo2']
categorical_features = {"Assortment" : ["A", "B", "C"], \
                        "Month" : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], \
                        "StateHoliday" : ["A", "B", "C"], \
                        "StoreType" : ["A", "B", "C", "D"] }
categorical_identity_features = {"DayOfWeek": [0,1,2,3,4,5,6],  #0 = mon, 1 = tue, ..., 6 = sun  
                                "WeekOfYear": [0,1,2,3]}

COLUMNS = integer_features + boolean_features + sorted(list(categorical_features.keys())) + sorted(list(categorical_identity_features.keys())) + label
FIELD_DEFAULTS = len(integer_features) * [[0]] + \
                len(boolean_features) * [[0]] + \
                [[categorical_features[x][0]] for x in sorted(list(categorical_features.keys()))] + \
                [[str(categorical_identity_features[x][0])] for x in sorted(list(categorical_identity_features.keys()))]+ \
                 len(label) * [[0.0]]
# calculating defaults for the categorical features