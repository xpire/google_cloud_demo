#credits: https://www.kaggle.com/elenapetrova/time-series-analysis-and-forecasts-with-prophet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline


import warnings
warnings.filterwarnings("ignore")


train_file = 'input/train.csv'
test_file = 'input/test.csv'
store_file = 'input/store.csv'
output_file = 'output/prediction.csv'

train = pd.read_csv(train_file,
                    parse_dates= True,
                    low_memory= False,
                    index_col= 'Date')
test = pd.read_csv(test_file)
store = pd.read_csv(store_file,
                    low_memory= False)
#~~~~~~~~Pre-processing~~~~~~~~
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

train_store.to_csv(output_file, index = False)
