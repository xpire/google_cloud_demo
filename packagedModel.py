import tensorflow as tf
import numpy as np


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


# ~~~~ PREPROCESS HELPER FUNCTIONS ~~~~

def preprocess(data, store):
    # set dates
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    data['WeekofYear'] = data.index.weekofyear
    # Missing values, removes zero sales stores and closed stores
    data = data[(data["Open"] != 0) & (data["Sales"] != 0)]
    # Missing values in store
    store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)
    store.fillna(0,inplace = True)
    # Merging the two datasets together
    data_store = pd.merge(train, store, how = 'inner', on = 'Store')
    # One Hot Encoding of Store Type
    data_store['StoreTypeA'] = (data_store['StoreType'] == 'a')
    data_store['StoreTypeB'] =  (data_store['StoreType'] == 'b') 
    data_store['StoreTypeC'] =  (data_store['StoreType'] == 'c') 
    data_store['StoreTypeD'] =  (data_store['StoreType'] == 'd') 
    data_store = data_store.drop(columns = ['StoreType'])
    # One Hot Encoding of Assortment
    data_store['AssortA'] =  (data_store['Assortment'] == 'a') 
    data_store['AssortB'] =  (data_store['Assortment'] == 'b') 
    data_store['AssortC'] =  (data_store['Assortment'] == 'c') 
    data_store = data_store.drop(columns = ['Assortment'])
    # One Hot Encoding of Day Of Week
    data_store['DayMon'] = (1 == data_store["DayOfWeek"]) 
    data_store['DayTue'] = (2 == data_store["DayOfWeek"]) 
    data_store['DayWed'] = (3 == data_store["DayOfWeek"]) 
    data_store['DayThu'] = (4 == data_store["DayOfWeek"]) 
    data_store['DayFri'] = (5 == data_store["DayOfWeek"]) 
    data_store['DaySat'] = (6 == data_store["DayOfWeek"]) 
    data_store['DaySun'] = (7 == data_store["DayOfWeek"]) 
    data_store = data_store.drop(columns = ['DayOfWeek'])
    # Removal of information with no effect
    data_store = data_store.drop(columns = ['StateHoliday'])
    data_store = data_store.drop(columns = ['PromoInterval'])
    data_store = data_store.drop(columns = ['Customers'])
    data_store = data_store.drop(columns=['Store'])
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

def input_evaluation_set():
    eval, store = input_data(test_file, store_file)
    data_store = preprocess(eval, store)
    #separate the X and Y components
    label = data_store['Sales'].values
    data_store = data_store.drop(columns = ['Sales'])
    # Convert to a tensorflow Dataset
    # unavailable_features = ['Store', 'Sales', 'Customers', 'PromoInterval', 'StateHoliday', 'DayOfWeek', 'Assortment', 'StoreType']
    # integer_features = []
    # features = ['Open', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekofYear', #OG
    # 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', #floats except Promo2
    # 'StoreTypeA', 'StoreTypeB', 'StoreTypeC', 'StoreTypeD', 
    # 'AssortA', 'AssortB', 'AssortC', 
    # #'StateHolidayA', 'StateHolidayB', 'StateHolidayC',
    # #'promoJan', 'promoFeb', 'promoMar', 'promoApr', 'promoMay', 'promoJun', 'promoJul', 'promoAug', 'promoSep', 'promoOct', 'promoNov', 'promoDec',
    # 'DayMon', 'DayTue', 'DayWed', 'DayThu', 'DayFri', 'DaySat', 'DaySun'] #Booleans
    # data_set = tf.data.Dataset.from_tensor_slices(
    #     (tf.cast(data_store.values, tf.float))
    # )
