#Justin Feb 2018

#script to just do the preprocessing.

from . import *
import pandas as pd

data =  pd.read_csv(train_file,
                    parse_dates= True,
                    low_memory= False,
                    index_col= 'Date')
store = pd.read_csv(store_file,
                    low_memory= False)
print(store.head())
print(data.head())

# set dates
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['WeekOfYear'] = data.index.weekofyear
data['Date'] = data.index
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
# data_store = data_store.drop(columns = ['Customers'])
# data_store = data_store.drop(columns=['Store'])
data_store = data_store.drop(columns=['CompetitionOpenSinceMonth'])
data_store = data_store.drop(columns=['CompetitionOpenSinceYear'])
data_store = data_store.drop(columns=['Promo2SinceWeek'])
data_store = data_store.drop(columns=['Promo2SinceYear'])
# sort columns so that it matches the order specified in __init__.py
# data_store = data_store[COLUMNS]
# assert the correct data types are applied in each column
for key in integer_features + boolean_features + list(categorical_identity_features.keys()) + list(bucket_categorical_features.keys()):
    data_store[key] = data_store[key].apply(lambda x: int(x))
for key in list(categorical_features.keys()):
    data_store[key] = data_store[key].apply(lambda x: str(x))

print("columns of data_store: {}".format(list(data_store)))
print(data_store.head())

#take 80% for training, 20% for validation
total = len(data_store)
data_store.to_csv(output_file, index=False)
# idx = np.random.permutation(total)
# data_store = data_store.sample(frac=1)
# train = data_store.iloc[:int(total*0.8), :]
# test = data_store.iloc[int(total*0.8):, :]
# print("Train")
# print(train.head())
# print("Test")
# print(test.head())
# # Calculating baseline to beat
# mean_sales = test["Sales"].mean()
# mse = test["Sales"].apply(lambda x: np.square(x-mean_sales)).mean()
# print("The MSE to beat is {}".format(mse))

# train.to_csv(output_train, index=False)
# test.to_csv(output_test, index=False)
# data_store.to_csv(output_file)
# return data_store


