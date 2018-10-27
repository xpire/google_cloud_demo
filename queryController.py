from flask import Flask
from flask import request
app = Flask(__name__)
#-------------------------------------------
# POST Request to /queryModel:
#   {
#       'storeName':<STORE_NAME>,
#       'storeOpen':<STORE_OPEN>,
#       'storePromo':<STORE_PROMO>,
#       'date':<DATE>,
#   }
#-------------------------------------------
@app.route('/querymodel', methods=['POST'])
def parseRequest():
    inputs = request.json
    storeName = inputs['storeName']
    storeOpen = inputs['storeOpen']
    promo = inputs['storePromo']
    date = inputs['date']
    salesPrediction = model(storeName, storeOpen, promo, date)
    return salesPrediction

#--------------------------------------------
# Justin: Model function goes here! Import
# all necessary functions from model.py if 
# needed :)
#--------------------------------------------
def model(storeName, storeOpen, promo, date):
    return 0
    