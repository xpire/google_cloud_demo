#credits: https://www.kaggle.com/elenapetrova/time-series-analysis-and-forecasts-with-prophet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore")

                    
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
    train_store['StoreTypeB'] = (train_store['StoreType'] == 'b')
    train_store['StoreTypeC'] = (train_store['StoreType'] == 'c')
    train_store['StoreTypeD'] = (train_store['StoreType'] == 'd')
    train_store = train_store.drop(columns = ['StoreType'])

    #onehot encoding of holidays
    train_store['StateHolidayA'] = (train_store['StateHoliday'] == 'a')
    train_store['StateHolidayB'] = (train_store['StateHoliday'] == 'b')
    train_store['StateHolidayC'] = (train_store['StateHoliday'] == 'c')
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

    #produce the X and Y

    train_store.to_csv(output_file, index = True)

#~~~~~~~~Loading~~~~~~~~
    print("#~~~~~~~~Loading~~~~~~~~")
    #convert pandas dataframe to a numpy data frame

#~~~~~~~~Training~~~~~~~~
    print("#~~~~~~~~Training~~~~~~~~")

# if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("mode", choices=["preprocess", "train", "eval"])
    
    # if (args.mode == "train"):
    #     preprocess()
    #     print("Training Run")
    #     train()
    # elif (args.mode == "eval"):
    #     print("Evaluation run")
    #     eval("input/test")
    # else:
    #     print("Preprocessing")
    #     preprocess()

preprocess()
