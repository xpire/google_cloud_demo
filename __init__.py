# Justin Or, Feb 2019

# Usage: python -m demo.packagedModel
# It is not a python program, and init holds all the global variables, 
# so program will fail when called as: python packagedModel.py

# ~~~~ CONSTANTS ~~~~
NPREDICTORS = 27 #len(features)
NOUTPUTS = 1 #len(target)
NHIDDEN = 20
NUMITERATIONS = 500000
BATCHSIZE = 1000
HYPERPARAMETERTUNING = 10

train_file = 'input/train.csv'
test_file = 'input/test.csv'
store_file = 'input/store.csv'
output_file = 'output/new_rossmann_prediction.csv'
output_train = 'output/trainPartitioned.csv'
output_test = 'output/testPartitioned.csv'

unavailable_features = ['Store', 'Customers', 'PromoInterval', 'StateHoliday', 'DayOfWeek', 'Assortment', \
                        'StoreType','CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Day', 'Month', 'Year']
label = ['Sales']
integer_features = ['CompetitionDistance', 'Year']
boolean_features = ['Open', 'Promo','SchoolHoliday','WeekofYear', 'Promo2', \
                    'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', \
                    'AssortA', 'AssortB', 'AssortC', \
                    'DayMon', 'DayTue', 'DayWed', 'DayThu', 'DayFri', 'DaySat', 'DaySun', \
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', \
                    'StateHolidayA', 'StateHolidayB', 'StateHolidayC'] #"Year" #+ list(range(2013, datetime.datetime.now().year+1))

COLUMNS = integer_features + boolean_features + label
FIELD_DEFAULTS = len(integer_features) * [[0]] + len(boolean_features) * [[0]] + len(label) * [[0.0]]

